"""
05: Generate publication-quality figures for the paper.
Also: re-run subgroup analysis with more stable models.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
from sklearn.base import clone
from sklearn.calibration import calibration_curve
import lightgbm as lgb
import shap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")
np.random.seed(42)

FONT = {"family": "DejaVu Sans", "size": 10}
matplotlib.rc("font", **FONT)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

df = pd.read_csv("ohhr_features.csv")

ALL_HTL  = [c for c in df.columns if c.startswith("ac_htl_")]
ALL_LS   = [c for c in df.columns if c.startswith("ls_") or c.startswith("ucl_mean_")]
ALL_COG  = ["demtect", "verbal_iq"]
ALL_DEMO = ["age","sex_m","school","education","net_income",
            "ha_supply_status","hl_progressing","hl_fluctuating","hl_family_often",
            "pta4_left","pta4_right","pta4_better","pta4_worse","pta4_asym",
            "slope_left","slope_right"]
BLOCK_E  = list(dict.fromkeys(ALL_HTL + ALL_LS + ALL_COG + ALL_DEMO))

BLOCKS = {
    "A\nPTA4-mean":       ["pta4_mean"],
    "B\nFull PTA":        ALL_HTL,
    "C\n+Loudness":       ALL_HTL + ALL_LS,
    "D\n+Cognition":      ALL_HTL + ALL_LS + ALL_COG,
    "E\nFull battery":    BLOCK_E,
}
BLOCK_LABELS = list(BLOCKS.keys())

def lgb_reg():
    return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=42, verbose=-1)

def lgb_cls():
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                               subsample=0.8, colsample_bytree=0.8,
                               class_weight="balanced", random_state=42, verbose=-1)

def pipe_reg(model):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("scl", StandardScaler()), ("model", model)])

def pipe_lgb(model):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("model", model)])

kf  = KFold(n_splits=10, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

y_reg = df["dtt_srt"].values
y_cls = df["dtt_poor_sin"].values

# ── Re-compute benchmark with MAE and R² ─────────────────────────────────────
print("Running benchmark (may take a minute)...")
res_rows = []
MODEL_DEFS = [
    ("Ridge",     pipe_reg(Ridge(alpha=1.0))),
    ("RF",        pipe_reg(RandomForestRegressor(200, max_features="sqrt",
                                                  min_samples_leaf=5, random_state=42, n_jobs=-1))),
    ("LightGBM",  pipe_lgb(lgb_reg())),
]

def cv_r2_mae(X, y, pipe, cv):
    maes, r2s = [], []
    for tr, te in cv.split(X):
        p = clone(pipe); p.fit(X[tr], y[tr])
        pred = p.predict(X[te])
        maes.append(mean_absolute_error(y[te], pred))
        r2s.append(r2_score(y[te], pred))
    return np.mean(r2s), np.std(r2s), np.mean(maes)

for block_label, feat_cols in BLOCKS.items():
    X = df[feat_cols].values
    for mname, mpipe in MODEL_DEFS:
        r2, r2_std, mae = cv_r2_mae(X, y_reg, mpipe, kf)
        res_rows.append({"block": block_label, "model": mname,
                         "r2": r2, "r2_std": r2_std, "mae": mae})

res_df = pd.DataFrame(res_rows)

# ── Bootstrap CI ─────────────────────────────────────────────────────────────
print("Bootstrap CI for LightGBM R²...")
boot_rows = []
for block_label, feat_cols in BLOCKS.items():
    X = df[feat_cols].values
    preds = np.zeros(len(y_reg))
    for tr, te in kf.split(X):
        p = clone(pipe_lgb(lgb_reg())); p.fit(X[tr], y_reg[tr])
        preds[te] = p.predict(X[te])
    r2_obs = r2_score(y_reg, preds)
    boots = [r2_score(y_reg[idx:=np.random.choice(len(y_reg),len(y_reg),True)],
                      preds[idx]) for _ in range(1000)]
    boot_rows.append({"block": block_label, "r2": r2_obs,
                      "ci_lo": np.percentile(boots, 2.5),
                      "ci_hi": np.percentile(boots, 97.5)})
boot_df = pd.DataFrame(boot_rows)

# ── SHAP importance ───────────────────────────────────────────────────────────
print("Computing SHAP values...")
imp = SimpleImputer(strategy="median")
X_imp = imp.fit_transform(df[BLOCK_E])
X_df  = pd.DataFrame(X_imp, columns=BLOCK_E)
model_s = lgb_reg(); model_s.fit(X_imp, y_reg)
expl = shap.TreeExplainer(model_s)
shap_v = expl.shap_values(X_df)
mean_shap = pd.Series(np.abs(shap_v).mean(0), index=BLOCK_E).sort_values(ascending=False)

# ── Subgroup analysis ─────────────────────────────────────────────────────────
print("Subgroup analysis...")
df["severity"] = pd.cut(df["pta4_better"],
    bins=[-np.inf, 25, 40, 60, np.inf],
    labels=["<=25 dB\n(mild)", "26-40 dB\n(mild-mod)", "41-60 dB\n(moderate)", ">60 dB\n(severe)"])

sg_rows = []
for sev, sub in df.groupby("severity", observed=True):
    n = len(sub)
    if n < 25: continue
    cv_sg = KFold(n_splits=min(5, n//10+1), shuffle=True, random_state=42)
    for block_label, feat_cols in [
        ("A\nPTA4-mean", ["pta4_mean"]),
        ("C\n+Loudness",  ALL_HTL + ALL_LS),
        ("D\n+Cognition", ALL_HTL + ALL_LS + ALL_COG),
    ]:
        X_sg = sub[feat_cols].values; y_sg = sub["dtt_srt"].values
        r2s = []
        for tr, te in cv_sg.split(X_sg):
            if len(te) < 3: continue
            p = clone(pipe_reg(Ridge(alpha=10.0)))
            p.fit(X_sg[tr], y_sg[tr])
            r2s.append(r2_score(y_sg[te], p.predict(X_sg[te])))
        sg_rows.append({"severity": str(sev), "block": block_label,
                        "n": n, "r2": np.mean(r2s)})

sg_df = pd.DataFrame(sg_rows)
print(sg_df.to_string(index=False))

# ── Calibration OOF ───────────────────────────────────────────────────────────
print("Calibration...")
X_A = df[["pta4_mean"]].values
X_E = df[BLOCK_E].values
cal_models = {
    "LR, PTA4 only": (X_A, Pipeline([("imp", SimpleImputer(strategy="median")),
                                      ("scl", StandardScaler()),
                                      ("model", LogisticRegression(C=1.0, max_iter=1000, random_state=42))])),
    "LGBM, full battery": (X_E, pipe_lgb(lgb_cls())),
}
cal_probs = {}
for label, (X_c, cpipe) in cal_models.items():
    oof = np.zeros(len(y_cls))
    for tr, te in skf.split(X_c, y_cls):
        p = clone(cpipe); p.fit(X_c[tr], y_cls[tr])
        oof[te] = p.predict_proba(X_c[te])[:, 1]
    cal_probs[label] = oof

# ── FIGURE 2: Benchmark (R², bootstrap CI) ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
colors = {"Ridge": "#4C72B0", "RF": "#DD8452", "LightGBM": "#55A868"}
markers = {"Ridge": "s", "RF": "D", "LightGBM": "o"}
x = np.arange(len(BLOCK_LABELS))
for mname, color in colors.items():
    sub = res_df[res_df.model == mname]
    ax.plot(x, sub.r2.values, marker=markers[mname], color=color,
            label=mname, linewidth=2, markersize=7)
ax.set_xticks(x); ax.set_xticklabels(BLOCK_LABELS, fontsize=9)
ax.set_ylabel("R² (10-fold cross-validation)")
ax.set_title("(a)  DTT SRT prediction performance", fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.set_ylim([0, 0.95])
ax.set_xlabel("Feature block")

# Bootstrap CI forest plot
ax2 = axes[1]
y_pos = np.arange(len(boot_df))
err_lo = boot_df.r2.values - boot_df.ci_lo.values
err_hi = boot_df.ci_hi.values - boot_df.r2.values
ax2.barh(y_pos, boot_df.r2.values,
         xerr=[err_lo, err_hi],
         color="#55A868", height=0.5, capsize=4, linewidth=0.8, edgecolor="k")
ax2.set_yticks(y_pos)
ax2.set_yticklabels([b.replace("\n", " ") for b in boot_df.block.values], fontsize=9)
ax2.set_xlabel("R² (LightGBM, 95% bootstrap CI)")
ax2.set_title("(b)  Bootstrap confidence intervals", fontsize=11, fontweight="bold")
ax2.set_xlim([0, 1.0])
for i, row in boot_df.iterrows():
    ax2.text(row.r2 + 0.02, i, f"{row.r2:.3f}", va="center", fontsize=9)

plt.tight_layout(pad=1.5)
plt.savefig("fig2_benchmark.pdf", bbox_inches="tight", dpi=200)
plt.savefig("fig2_benchmark.png", bbox_inches="tight", dpi=200)
plt.close()
print("Saved: fig2_benchmark.pdf/png")

# ── FIGURE 3: SHAP summary ────────────────────────────────────────────────────
top15 = mean_shap.head(15)
friendly = {
    "pta4_better":       "PTA4 (better ear)",
    "pta4_worse":        "PTA4 (worse ear)",
    "pta4_mean":         "PTA4 mean",
    "pta4_asym":         "PTA4 asymmetry",
    "age":               "Age",
    "ls_mlow_left":      "Loudness slope mlow (L)",
    "ls_mlow_right":     "Loudness slope mlow (R)",
    "ls_mhigh_right":    "Loudness slope mhigh (R)",
    "ls_mhigh_left":     "Loudness slope mhigh (L)",
    "ls_mcut_left":      "Loudness MCL (L)",
    "ls_mcut_right":     "Loudness MCL (R)",
    "demtect":           "DemTect (cognition)",
    "verbal_iq":         "Verbal intelligence",
    "ac_htl_right_2000": "AC HTL 2000 Hz (R)",
    "ac_htl_right_4000": "AC HTL 4000 Hz (R)",
    "ac_htl_left_2000":  "AC HTL 2000 Hz (L)",
    "ac_htl_left_4000":  "AC HTL 4000 Hz (L)",
    "ac_htl_right_1500": "AC HTL 1500 Hz (R)",
    "ac_htl_right_250":  "AC HTL 250 Hz (R)",
    "ac_htl_right_750":  "AC HTL 750 Hz (R)",
    "slope_right":       "Audiogram slope (R)",
    "slope_left":        "Audiogram slope (L)",
    "sex_m":             "Sex (male)",
}
def feat_label(f):
    return friendly.get(f, f.replace("_"," ").replace("ac htl","AC HTL").replace("ls ","LS "))

labels15 = [feat_label(f) for f in top15.index]

# Color-code by feature type
def feat_color(f):
    if f.startswith("pta4") or f.startswith("ac_htl") or f.startswith("slope"):
        return "#4C72B0"  # audiogram = blue
    elif f.startswith("ls") or f.startswith("ucl"):
        return "#DD8452"  # loudness = orange
    elif f in ["demtect","verbal_iq"]:
        return "#55A868"  # cognition = green
    else:
        return "#C44E52"  # demographic = red

colors15 = [feat_color(f) for f in top15.index]

fig, ax = plt.subplots(figsize=(7, 5.5))
y_p = np.arange(len(top15))
ax.barh(y_p, top15.values[::-1], color=[feat_color(f) for f in top15.index[::-1]])
ax.set_yticks(y_p); ax.set_yticklabels(labels15[::-1], fontsize=9)
ax.set_xlabel("Mean |SHAP value|  (DTT SRT, dB SNR)")
ax.set_title("Feature importance (LightGBM, full battery)", fontsize=11, fontweight="bold")

patches = [
    mpatches.Patch(color="#4C72B0", label="Audiogram (PTA)"),
    mpatches.Patch(color="#DD8452", label="Loudness scaling"),
    mpatches.Patch(color="#55A868", label="Cognition"),
    mpatches.Patch(color="#C44E52", label="Demographics"),
]
ax.legend(handles=patches, fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig("fig3_shap.pdf", bbox_inches="tight", dpi=200)
plt.savefig("fig3_shap.png", bbox_inches="tight", dpi=200)
plt.close()
print("Saved: fig3_shap.pdf/png")

# ── FIGURE 4: Subgroup + Calibration (combined) ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Subgroup panel
ax = axes[0]
sev_labels = sg_df["severity"].unique()
block_colors = {"A\nPTA4-mean": "#4C72B0", "C\n+Loudness": "#DD8452", "D\n+Cognition": "#55A868"}
n_sev = len(sev_labels)
bar_width = 0.22
x_sev = np.arange(n_sev)
for i, (block_label, color) in enumerate(block_colors.items()):
    vals = [sg_df[(sg_df.severity == sev) & (sg_df.block == block_label)]["r2"].values
            for sev in sev_labels]
    vals_plot = [v[0] if len(v) > 0 else 0.0 for v in vals]
    offset = (i - 1) * bar_width
    ax.bar(x_sev + offset, vals_plot, width=bar_width,
           label=block_label.replace("\n"," "), color=color, edgecolor="k", linewidth=0.5)

ax.set_xticks(x_sev)
ax.set_xticklabels([s.replace("\n"," ") for s in sev_labels], fontsize=9)
ax.set_ylabel("R² (5-fold CV, Ridge)")
ax.set_title("(a)  Performance by hearing loss severity", fontsize=11, fontweight="bold")
ax.legend(fontsize=8); ax.set_ylim([-0.1, 0.75])
ax.axhline(0, color="k", linewidth=0.6)
ax.set_xlabel("Severity group (better-ear PTA4)")

# Calibration panel
ax2 = axes[1]
line_styles = {"LR, PTA4 only": ("--", "#4C72B0"), "LGBM, full battery": ("-", "#DD8452")}
for label, oof_prob in cal_probs.items():
    frac_pos, mean_pred = calibration_curve(y_cls, oof_prob, n_bins=8, strategy="quantile")
    style, color = line_styles[label]
    auroc = roc_auc_score(y_cls, oof_prob)
    ax2.plot(mean_pred, frac_pos, f"o{style}", color=color, linewidth=2, markersize=5,
             label=f"{label}\n(AUROC={auroc:.3f})")
ax2.plot([0,1],[0,1],"k:", linewidth=1, label="Perfect calibration")
ax2.set_xlabel("Mean predicted probability")
ax2.set_ylabel("Observed fraction positive")
ax2.set_title("(b)  Calibration (poor SIN classification)", fontsize=11, fontweight="bold")
ax2.legend(fontsize=8.5); ax2.set_xlim([0,1]); ax2.set_ylim([0,1])

plt.tight_layout(pad=1.5)
plt.savefig("fig4_subgroup_calibration.pdf", bbox_inches="tight", dpi=200)
plt.savefig("fig4_subgroup_calibration.png", bbox_inches="tight", dpi=200)
plt.close()
print("Saved: fig4_subgroup_calibration.pdf/png")

# ── TABLE 1: main results ─────────────────────────────────────────────────────
print("\n=== TABLE 1: Benchmark Results ===")
print(f"{'Block':<22} {'Model':<12} {'R2':>6} {'MAE':>6}")
print("-"*48)
for _, row in res_df.iterrows():
    print(f"{row.block.replace(chr(10),' '):<22} {row.model:<12} {row.r2:6.3f} {row.mae:6.3f}")

# ── TABLE 2: Subgroup ─────────────────────────────────────────────────────────
print("\n=== TABLE 2: Subgroup R² (Ridge) ===")
pivot = sg_df.pivot(index="severity", columns="block", values="r2")
print(pivot.to_string())

# Save CSVs
res_df.to_csv("table1_benchmark.csv", index=False)
sg_df.to_csv("table2_subgroup.csv", index=False)
boot_df.to_csv("table_bootstrap_ci.csv", index=False)
print("\nAll figures and tables saved.")
