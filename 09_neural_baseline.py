"""
09: Neural baseline — well-regularized MLP with batch norm on tabular data.
Also implements feature stability analysis (SHAP across CV folds).
"""
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.base import clone
from scipy.stats import spearmanr
import lightgbm as lgb
import shap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

df = pd.read_csv("ohhr_features.csv")
ALL_HTL  = [c for c in df.columns if c.startswith("ac_htl_")]
ALL_LS   = [c for c in df.columns if c.startswith("ls_") or c.startswith("ucl_mean_")]
ALL_COG  = ["demtect","verbal_iq"]
ALL_DEMO = ["age","sex_m","school","education","net_income",
            "ha_supply_status","hl_progressing","hl_fluctuating","hl_family_often",
            "pta4_left","pta4_right","pta4_better","pta4_worse","pta4_asym",
            "slope_left","slope_right"]
BLOCK_E  = list(dict.fromkeys(ALL_HTL + ALL_LS + ALL_COG + ALL_DEMO))

y = df["dtt_srt"].values.astype(np.float32)
kf = KFold(n_splits=10, shuffle=True, random_state=42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── MLP architecture ──────────────────────────────────────────────────────────
class TabMLP(nn.Module):
    def __init__(self, in_dim, hidden=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_mlp(X_tr, y_tr, X_val, y_val, in_dim, epochs=200, lr=1e-3,
              hidden=(256,128,64), dropout=0.3, weight_decay=1e-4, patience=20):
    model = TabMLP(in_dim, hidden, dropout).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(X_tr, dtype=torch.float32).to(device)
    yt = torch.tensor(y_tr, dtype=torch.float32).to(device)
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    yv = torch.tensor(y_val, dtype=torch.float32).to(device)

    ds = TensorDataset(Xt, yt)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    best_val, best_state, wait = np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        sched.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model

# ── 10-fold CV for MLP ────────────────────────────────────────────────────────
print("Training MLP (10-fold CV)...")
X_all = df[BLOCK_E].values.astype(np.float32)
imp = SimpleImputer(strategy="median")
scl = StandardScaler()
X_all = scl.fit_transform(imp.fit_transform(X_all))

mlp_r2s, mlp_maes = [], []
for fold, (tr, te) in enumerate(kf.split(X_all)):
    X_tr, X_te = X_all[tr], X_all[te]
    y_tr, y_te = y[tr], y[te]
    model = train_mlp(X_tr, y_tr, X_te, y_te, in_dim=X_all.shape[1],
                      epochs=300, lr=5e-4, hidden=(256,128,64), dropout=0.25,
                      weight_decay=1e-3, patience=30)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_te, dtype=torch.float32).to(device)).cpu().numpy()
    mlp_r2s.append(r2_score(y_te, pred))
    mlp_maes.append(mean_absolute_error(y_te, pred))
    print(f"  Fold {fold+1:2d}: R²={mlp_r2s[-1]:.3f}  MAE={mlp_maes[-1]:.3f}")

print(f"\nMLP (full battery): R²={np.mean(mlp_r2s):.4f}±{np.std(mlp_r2s):.4f}  "
      f"MAE={np.mean(mlp_maes):.4f}±{np.std(mlp_maes):.4f}")

mlp_df = pd.DataFrame({"fold": range(10), "r2": mlp_r2s, "mae": mlp_maes})
mlp_df.to_csv("results_mlp.csv", index=False)

# ── Feature stability analysis (SHAP across folds) ────────────────────────────
print("\n=== SHAP stability analysis (LightGBM, Block E, 10 folds) ===")

def lgb_reg():
    return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8,
                              reg_lambda=1.0, random_state=42, verbose=-1)

X_raw = df[BLOCK_E].values
imp2  = SimpleImputer(strategy="median")
X_imp = imp2.fit_transform(X_raw)
X_df  = pd.DataFrame(X_imp, columns=BLOCK_E)

shap_all_folds = []
for fold, (tr, te) in enumerate(kf.split(X_imp)):
    m = lgb_reg(); m.fit(X_imp[tr], y[tr])
    expl = shap.TreeExplainer(m)
    sv   = np.abs(expl.shap_values(X_df.iloc[te]))
    shap_all_folds.append(sv.mean(axis=0))

shap_matrix = np.array(shap_all_folds)  # (10, n_features)
shap_mean   = shap_matrix.mean(axis=0)
shap_std    = shap_matrix.std(axis=0)
shap_cv     = shap_std / (shap_mean + 1e-9)   # coefficient of variation

stability_df = pd.DataFrame({
    "feature":  BLOCK_E,
    "shap_mean": shap_mean,
    "shap_std":  shap_std,
    "shap_cv":   shap_cv,
}).sort_values("shap_mean", ascending=False)

stability_df.to_csv("results_shap_stability.csv", index=False)
print("\nTop 15 features with stability (lower CV = more stable):")
print(stability_df.head(15)[["feature","shap_mean","shap_cv"]].to_string(index=False))

# ── Figure: model comparison + stability ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: model comparison (R²)
ax = axes[0]
lgb_r2 = 0.775     # from benchmark
rf_r2  = 0.805
ridge_r2 = 0.730
mlp_r2_m = np.mean(mlp_r2s)
mlp_r2_s = np.std(mlp_r2s)

model_names = ["Ridge", "LightGBM", "RF", "MLP"]
model_r2s   = [ridge_r2, lgb_r2, rf_r2, mlp_r2_m]
model_errs  = [0.0, 0.0, 0.0, mlp_r2_s]
colors_m    = ["#4C72B0","#55A868","#DD8452","#C44E52"]
bars = ax.bar(model_names, model_r2s, color=colors_m, edgecolor="k", linewidth=0.6,
              yerr=model_errs, capsize=4)
ax.set_ylabel("R² (10-fold CV, Block E, DTT SRT)")
ax.set_title("(a)  Model comparison (full battery)", fontweight="bold")
ax.set_ylim([0.6, 0.9])
for bar, val in zip(bars, model_r2s):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.004, f"{val:.3f}", ha="center", fontsize=9)

# Panel B: SHAP stability (top-12 features, mean ± std across folds)
ax2 = axes[1]
top12 = stability_df.head(12)
friendly = {
    "pta4_better":"PTA4 (better ear)", "pta4_worse":"PTA4 (worse ear)",
    "pta4_mean":"PTA4 mean","pta4_asym":"PTA4 asymmetry","age":"Age",
    "ls_mlow_left":"Loudness mlow (L)","ls_mlow_right":"Loudness mlow (R)",
    "ls_mhigh_right":"Loudness mhigh (R)","ls_mhigh_left":"Loudness mhigh (L)",
    "ls_mcut_left":"Loudness MCL (L)","ls_mcut_right":"Loudness MCL (R)",
    "ac_htl_right_2000":"AC HTL 2 kHz (R)","ac_htl_right_4000":"AC HTL 4 kHz (R)",
    "ac_htl_left_2000":"AC HTL 2 kHz (L)","ac_htl_right_1500":"AC HTL 1.5 kHz (R)",
}
def fl(f): return friendly.get(f, f.replace("ac_htl_","").replace("_"," "))

y_p = np.arange(len(top12))
labels = [fl(f) for f in top12.feature.values[::-1]]
means  = top12.shap_mean.values[::-1]
stds   = top12.shap_std.values[::-1]

def feat_color(f):
    if "pta4" in f or "ac_htl" in f or "slope" in f: return "#4C72B0"
    if "ls_" in f or "ucl" in f: return "#DD8452"
    if f in ["demtect","verbal_iq"]: return "#55A868"
    return "#C44E52"

bar_colors = [feat_color(f) for f in top12.feature.values[::-1]]
ax2.barh(y_p, means, xerr=stds, color=bar_colors, height=0.6,
         capsize=3, edgecolor="k", linewidth=0.5, alpha=0.85)
ax2.set_yticks(y_p); ax2.set_yticklabels(labels, fontsize=9)
ax2.set_xlabel("Mean |SHAP| across CV folds (± 1 SD)")
ax2.set_title("(b)  Feature importance stability across folds", fontweight="bold")

import matplotlib.patches as mpatches
patches = [mpatches.Patch(color="#4C72B0",label="Audiogram"),
           mpatches.Patch(color="#DD8452",label="Loudness"),
           mpatches.Patch(color="#55A868",label="Cognition"),
           mpatches.Patch(color="#C44E52",label="Demographics")]
ax2.legend(handles=patches, fontsize=8, loc="lower right")

plt.tight_layout()
plt.savefig("fig6_model_comparison_stability.pdf", bbox_inches="tight", dpi=200)
plt.savefig("fig6_model_comparison_stability.png", bbox_inches="tight", dpi=200)
plt.close()
print("\nSaved: fig6_model_comparison_stability.pdf/png")
print("Done.")
