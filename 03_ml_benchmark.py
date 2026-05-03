"""
Main ML benchmark: predict DTT SRT (primary) and GST SRT (secondary)
from progressively richer audiological feature sets.

Feature blocks:
  A - PTA4 mean only (1 feature)       — simplest clinical baseline
  B - Full PTA (all HTL freq×ear)       — standard audiogram
  C - B + UCL + loudness scaling        — audiogram + loudness
  D - C + DemTect + verbal IQ           — + cognition
  E - All 42 features                   — full battery

Models: Ridge, Random Forest, LightGBM (+ classification LightGBM for Block E)
Evaluation: 10-fold CV, MAE / RMSE / R²
"""
import warnings, json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_validate, StratifiedKFold
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, average_precision_score
import lightgbm as lgb
import shap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("ohhr_features.csv")

# ── Define feature blocks ─────────────────────────────────────────────────────
BLOCK_A = ["pta4_mean"]

BLOCK_B_PTA = [c for c in df.columns if c.startswith("ac_htl_")]
BLOCK_B = BLOCK_B_PTA

BLOCK_C_EXTRA = [c for c in df.columns if c.startswith("ucl_mean_") or c.startswith("ls_")]
BLOCK_C = BLOCK_B + BLOCK_C_EXTRA

BLOCK_D_EXTRA = ["demtect", "verbal_iq"]
BLOCK_D = BLOCK_C + BLOCK_D_EXTRA

BLOCK_E_EXTRA = ["age", "sex_m", "school", "education", "net_income",
                 "ha_supply_status", "hl_progressing", "hl_fluctuating", "hl_family_often",
                 "pta4_left", "pta4_right", "pta4_better", "pta4_worse",
                 "pta4_asym", "slope_left", "slope_right"]
BLOCK_E = list(dict.fromkeys(BLOCK_D + BLOCK_E_EXTRA))   # deduplicated

BLOCKS = {
    "A: PTA4-mean": BLOCK_A,
    "B: Full PTA":  BLOCK_B,
    "C: +Loudness": BLOCK_C,
    "D: +Cognition":BLOCK_D,
    "E: Full battery": BLOCK_E,
}

print("Feature block sizes:")
for name, cols in BLOCKS.items():
    print(f"  {name:20s}: {len(cols)} features")

# ── Helper: build imputer+model pipeline ─────────────────────────────────────
def make_pipeline(model):
    return Pipeline([
        ("imp",   SimpleImputer(strategy="median")),
        ("scl",   StandardScaler()),
        ("model", model),
    ])

def lgb_regressor():
    return lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=42, verbose=-1
    )

def lgb_classifier():
    return lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        class_weight="balanced", random_state=42, verbose=-1
    )

MODELS = {
    "Ridge":  make_pipeline(Ridge(alpha=1.0)),
    "RF":     make_pipeline(RandomForestRegressor(n_estimators=200, max_features="sqrt",
                                                   min_samples_leaf=5, random_state=42, n_jobs=-1)),
    "LightGBM": Pipeline([
        ("imp",   SimpleImputer(strategy="median")),
        ("model", lgb_regressor()),
    ]),
}

CV = KFold(n_splits=10, shuffle=True, random_state=42)

# ── Run benchmark ─────────────────────────────────────────────────────────────
def cv_regression(X_df, y_series, model_pipe, cv):
    X = X_df.values
    y = y_series.values
    maes, rmses, r2s = [], [], []
    for tr, te in cv.split(X):
        pipe = make_pipeline(model_pipe.named_steps["model"].__class__(
            **model_pipe.named_steps["model"].get_params()
        )) if "model" in model_pipe.named_steps else model_pipe
        # Simpler: just clone
        from sklearn.base import clone
        p = clone(model_pipe)
        p.fit(X[tr], y[tr])
        pred = p.predict(X[te])
        maes.append(mean_absolute_error(y[te], pred))
        rmses.append(np.sqrt(np.mean((y[te]-pred)**2)))
        r2s.append(r2_score(y[te], pred))
    return np.mean(maes), np.mean(rmses), np.mean(r2s)

results = []
TARGET = "dtt_srt"
y = df[TARGET]

print(f"\n{'='*65}")
print(f"Target: {TARGET}  (mean={y.mean():.2f}, sd={y.std():.2f})")
print(f"{'='*65}")
print(f"{'Block':<22} {'Model':<12} {'MAE':>6} {'RMSE':>6} {'R²':>6}")
print("-"*65)

for block_name, feat_cols in BLOCKS.items():
    X = df[feat_cols]
    for model_name, model_pipe in MODELS.items():
        mae, rmse, r2 = cv_regression(X, y, model_pipe, CV)
        results.append({"block": block_name, "model": model_name,
                        "mae": mae, "rmse": rmse, "r2": r2})
        print(f"{block_name:<22} {model_name:<12} {mae:6.3f} {rmse:6.3f} {r2:6.3f}")
    print()

results_df = pd.DataFrame(results)
results_df.to_csv("results_dtt_regression.csv", index=False)
print("Saved: results_dtt_regression.csv")

# ── Secondary target: GST SRT ─────────────────────────────────────────────────
print(f"\n{'='*65}")
TARGET2 = "gst_srt"
y2 = df[TARGET2]
print(f"Target: {TARGET2}  (mean={y2.mean():.2f}, sd={y2.std():.2f})")
print(f"{'='*65}")
print(f"{'Block':<22} {'Model':<12} {'MAE':>6} {'RMSE':>6} {'R²':>6}")
print("-"*65)

results2 = []
for block_name, feat_cols in BLOCKS.items():
    X = df[feat_cols]
    for model_name, model_pipe in MODELS.items():
        mae, rmse, r2 = cv_regression(X, y2, model_pipe, CV)
        results2.append({"block": block_name, "model": model_name,
                         "mae": mae, "rmse": rmse, "r2": r2})
        print(f"{block_name:<22} {model_name:<12} {mae:6.3f} {rmse:6.3f} {r2:6.3f}")
    print()

pd.DataFrame(results2).to_csv("results_gst_regression.csv", index=False)
print("Saved: results_gst_regression.csv")

# ── Classification: poor SIN prediction (DTT SRT > -2 dB) ────────────────────
print(f"\n{'='*65}")
y_cls = df["dtt_poor_sin"]
print(f"Target: poor SIN (DTT SRT > -2 dB)  prevalence={y_cls.mean():.3f}")
print(f"{'='*65}")

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

cls_models = {
    "LR":       Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("scl", StandardScaler()),
                           ("model", LogisticRegression(C=1.0, max_iter=1000, random_state=42))]),
    "LightGBM": Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("model", lgb_classifier())]),
}

results_cls = []
print(f"{'Block':<22} {'Model':<12} {'AUROC':>7} {'AUPRC':>7}")
print("-"*55)
for block_name, feat_cols in BLOCKS.items():
    X = df[feat_cols].values
    y_c = y_cls.values
    for model_name, pipe in cls_models.items():
        aurocs, auprcs = [], []
        for tr, te in skf.split(X, y_c):
            p = clone(pipe)
            p.fit(X[tr], y_c[tr])
            proba = p.predict_proba(X[te])[:, 1]
            aurocs.append(roc_auc_score(y_c[te], proba))
            auprcs.append(average_precision_score(y_c[te], proba))
        results_cls.append({"block": block_name, "model": model_name,
                             "auroc": np.mean(aurocs), "auprc": np.mean(auprcs)})
        print(f"{block_name:<22} {model_name:<12} {np.mean(aurocs):7.4f} {np.mean(auprcs):7.4f}")
    print()

pd.DataFrame(results_cls).to_csv("results_classification.csv", index=False)
print("Saved: results_classification.csv")

# ── SHAP analysis on best model (LightGBM, Block E, DTT SRT) ─────────────────
print("\n=== SHAP analysis (LightGBM, all features, DTT SRT) ===")
feat_cols_E = BLOCK_E
X_E = df[feat_cols_E].copy()
imp = SimpleImputer(strategy="median")
X_imp = imp.fit_transform(X_E)
X_imp_df = pd.DataFrame(X_imp, columns=feat_cols_E)

model_shap = lgb_regressor()
model_shap.fit(X_imp, df["dtt_srt"].values)

explainer = shap.TreeExplainer(model_shap)
shap_vals = explainer.shap_values(X_imp_df)

# Summary plot
plt.figure(figsize=(8, 7))
shap.summary_plot(shap_vals, X_imp_df, show=False, max_display=20)
plt.tight_layout()
plt.savefig("fig_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig_shap_summary.png")

# Mean absolute SHAP
mean_shap = pd.Series(np.abs(shap_vals).mean(axis=0), index=feat_cols_E)
mean_shap = mean_shap.sort_values(ascending=False)
print("\nTop 15 features by mean |SHAP|:")
print(mean_shap.head(15).to_string())
mean_shap.to_csv("shap_importance.csv")

# ── Results summary figure ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

colors = {"Ridge": "#4C72B0", "RF": "#DD8452", "LightGBM": "#55A868"}

for ax, (target_label, res, metric) in zip(axes, [
    ("DTT SRT (R²)",   results_df,  "r2"),
    ("GST SRT (R²)",   pd.DataFrame(results2), "r2"),
    ("Poor SIN (AUROC)", pd.DataFrame(results_cls), "auroc"),
]):
    for model_name, color in colors.items():
        sub = res[res.model == model_name] if "model" in res.columns else res[res.model == model_name]
        if model_name not in res.model.values:
            continue
        vals = sub[metric].values
        x_labels = [b.split(":")[0].strip() for b in sub["block"].values]
        ax.plot(x_labels, vals, marker="o", label=model_name, color=color, linewidth=2)
    ax.set_title(target_label, fontsize=12)
    ax.set_xlabel("Feature block")
    ax.set_ylabel(metric.upper())
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=30)

plt.suptitle("Prediction performance by feature block and model", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("fig_benchmark_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig_benchmark_results.png")

print("\nAll done!")
