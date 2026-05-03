"""
04: Calibration analysis + minimal predictor set + bootstrap CIs + subgroup analysis
"""
import warnings, json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (mean_absolute_error, r2_score,
                              roc_auc_score, average_precision_score,
                              brier_score_loss)
from sklearn.calibration import calibration_curve
from sklearn.base import clone
import lightgbm as lgb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
np.random.seed(42)

df = pd.read_csv("ohhr_features.csv")

# Feature blocks (same as benchmark)
ALL_HTL = [c for c in df.columns if c.startswith("ac_htl_")]
ALL_LS  = [c for c in df.columns if c.startswith("ls_") or c.startswith("ucl_mean_")]
ALL_COG = ["demtect", "verbal_iq"]
ALL_DEMO= ["age","sex_m","school","education","net_income",
           "ha_supply_status","hl_progressing","hl_fluctuating","hl_family_often",
           "pta4_left","pta4_right","pta4_better","pta4_worse","pta4_asym","slope_left","slope_right"]
BLOCK_E = list(dict.fromkeys(ALL_HTL + ALL_LS + ALL_COG + ALL_DEMO))

y_reg = df["dtt_srt"].values
y_cls = df["dtt_poor_sin"].values

def lgb_cls():
    return lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                               subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                               class_weight="balanced", random_state=42, verbose=-1)

def lgb_reg():
    return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=42, verbose=-1)

def make_cls_pipe(model):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("model", model)])

def make_reg_pipe(model):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("scl", StandardScaler()), ("model", model)])

# ── 1. Calibration analysis ───────────────────────────────────────────────────
print("=== Calibration Analysis ===")
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Collect OOF probabilities for two models: LR (baseline) and LightGBM (full)
X_E = df[BLOCK_E].values
X_A = df[["pta4_mean"]].values

oof_results = {}
for label, X, pipe in [
    ("LR (PTA4 only)",       X_A, make_cls_pipe(LogisticRegression(C=1.0, max_iter=1000, random_state=42))),
    ("LightGBM (full batt)", X_E, make_cls_pipe(lgb_cls())),
]:
    oof_prob = np.zeros(len(y_cls))
    for tr, te in skf.split(X, y_cls):
        p = clone(pipe); p.fit(X[tr], y_cls[tr])
        oof_prob[te] = p.predict_proba(X[te])[:, 1]
    brier = brier_score_loss(y_cls, oof_prob)
    auroc = roc_auc_score(y_cls, oof_prob)
    oof_results[label] = {"prob": oof_prob, "brier": brier, "auroc": auroc}
    print(f"  {label:30s}  AUROC={auroc:.4f}  Brier={brier:.4f}")

# Calibration plot
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
colors = {"LR (PTA4 only)": "#4C72B0", "LightGBM (full batt)": "#DD8452"}
for ax, n_bins in zip(axes, [8, 8]):
    for label, res in oof_results.items():
        frac_pos, mean_pred = calibration_curve(y_cls, res["prob"], n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "s-", label=f"{label}\n(Brier={res['brier']:.3f})",
                color=colors[label], linewidth=2, markersize=5)
    ax.plot([0,1],[0,1],"k--", alpha=0.4, label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve: poor SIN")
    ax.legend(fontsize=9)
    ax.set_xlim([0,1]); ax.set_ylim([0,1])
    break  # just one panel

# Reliability diagram for single panel
axes[1].hist(oof_results["LightGBM (full batt)"]["prob"], bins=20, color="#DD8452", alpha=0.7)
axes[1].set_xlabel("Predicted probability"); axes[1].set_ylabel("Count")
axes[1].set_title("Distribution of predicted probabilities\n(LightGBM, full battery)")
plt.tight_layout()
plt.savefig("fig_calibration.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig_calibration.png")

# ── 2. Bootstrap confidence intervals ─────────────────────────────────────────
print("\n=== Bootstrap CIs for R² (DTT SRT, 10-fold CV) ===")
kf = KFold(n_splits=10, shuffle=True, random_state=42)

def get_r2_10fold(X, y):
    preds = np.zeros(len(y))
    for tr, te in kf.split(X):
        p = clone(make_reg_pipe(lgb_reg()))
        p.fit(X[tr], y[tr])
        preds[te] = p.predict(X[te])
    return r2_score(y, preds), preds

def bootstrap_ci_r2(y_true, y_pred, n_boot=1000):
    n = len(y_true)
    boot_r2s = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        boot_r2s.append(r2_score(y_true[idx], y_pred[idx]))
    return np.percentile(boot_r2s, 2.5), np.percentile(boot_r2s, 97.5)

ci_results = []
for block_name, feat_cols in [
    ("A: PTA4 mean",   ["pta4_mean"]),
    ("B: Full PTA",    ALL_HTL),
    ("C: +Loudness",   ALL_HTL + ALL_LS),
    ("D: +Cognition",  ALL_HTL + ALL_LS + ALL_COG),
    ("E: Full battery",BLOCK_E),
]:
    X = df[feat_cols].values
    r2, preds = get_r2_10fold(X, y_reg)
    lo, hi = bootstrap_ci_r2(y_reg, preds)
    ci_results.append({"block": block_name, "r2": r2, "ci_lo": lo, "ci_hi": hi})
    print(f"  {block_name:<22}: R²={r2:.3f}  95%CI=[{lo:.3f}, {hi:.3f}]")

ci_df = pd.DataFrame(ci_results)

# Forest plot
fig, ax = plt.subplots(figsize=(7, 4))
y_pos = range(len(ci_df))
ax.barh(list(y_pos), ci_df.r2, xerr=[ci_df.r2 - ci_df.ci_lo, ci_df.ci_hi - ci_df.r2],
        color="#55A868", height=0.5, capsize=4, edgecolor="k", linewidth=0.8)
ax.set_yticks(list(y_pos)); ax.set_yticklabels(ci_df.block)
ax.set_xlabel("R² (10-fold CV, LightGBM)")
ax.set_title("Prediction performance by feature block\n(DTT SRT, 95% bootstrap CI)")
ax.axvline(0, color="k", linewidth=0.8)
ax.set_xlim([0, 0.95])
plt.tight_layout()
plt.savefig("fig_r2_ci.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig_r2_ci.png")

# ── 3. Minimal predictor analysis ─────────────────────────────────────────────
print("\n=== Minimal Predictor Set (SHAP-ranked forward selection) ===")
import shap

X_E_imp = SimpleImputer(strategy="median").fit_transform(df[BLOCK_E])
X_E_df  = pd.DataFrame(X_E_imp, columns=BLOCK_E)
model_shap = lgb_reg()
model_shap.fit(X_E_imp, y_reg)
explainer  = shap.TreeExplainer(model_shap)
shap_vals  = explainer.shap_values(X_E_df)
mean_shap  = pd.Series(np.abs(shap_vals).mean(axis=0), index=BLOCK_E).sort_values(ascending=False)

# Progressive addition of features in SHAP importance order
shap_order = mean_shap.index.tolist()
minimal_results = []
for k in [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, len(BLOCK_E)]:
    if k > len(BLOCK_E): break
    feat_k = shap_order[:k]
    X_k = df[feat_k].values
    r2s = []
    for tr, te in kf.split(X_k):
        p = clone(make_reg_pipe(lgb_reg()))
        p.fit(X_k[tr], y_reg[tr])
        r2s.append(r2_score(y_reg[te], p.predict(X_k[te])))
    r2_k = np.mean(r2s)
    minimal_results.append({"k": k, "r2": r2_k,
                              "top_features": ", ".join(feat_k[:min(3,k)])+"..."})
    print(f"  Top-{k:2d} features: R²={r2_k:.3f}  (leading: {feat_k[0]})")

min_df = pd.DataFrame(minimal_results)
min_df.to_csv("results_minimal_predictor.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(min_df.k, min_df.r2, "o-", color="#4C72B0", linewidth=2, markersize=6)
ax.axhline(ci_df[ci_df.block.str.startswith("E")].r2.values[0],
           color="gray", linestyle="--", label="Full battery")
ax.set_xlabel("Number of features (SHAP-ranked)")
ax.set_ylabel("R² (10-fold CV)")
ax.set_title("Minimal predictor analysis: DTT SRT")
ax.legend(); ax.set_xlim([0, len(BLOCK_E)+1])
plt.tight_layout()
plt.savefig("fig_minimal_predictor.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig_minimal_predictor.png")

# ── 4. Subgroup analysis by hearing severity ──────────────────────────────────
print("\n=== Subgroup Analysis by PTA4 Severity ===")
# WHO classifications (better ear)
df["severity"] = pd.cut(df["pta4_better"],
    bins=[-np.inf, 25, 40, 60, np.inf],
    labels=["<=25 dB", "26-40 dB", "41-60 dB", ">60 dB"])

print(df.severity.value_counts().sort_index())

# Run LightGBM Block A vs E for each subgroup (leave-group-out not feasible,
# just report descriptive stats + baseline vs full R² per group)
subgroup_results = []
for sev, sub_df in df.groupby("severity", observed=True):
    n = len(sub_df)
    if n < 20:
        continue
    # Simple within-subgroup LOO estimate (too small for 10-fold, use 5-fold)
    cv5 = KFold(n_splits=5, shuffle=True, random_state=42)
    for block_name, feat_cols in [("A",["pta4_mean"]), ("E",BLOCK_E)]:
        X = sub_df[feat_cols].values
        y = sub_df["dtt_srt"].values
        r2s = []
        for tr, te in cv5.split(X):
            if len(te) < 2: continue
            p = clone(make_reg_pipe(Ridge(alpha=1.0)))
            p.fit(X[tr], y[tr])
            r2s.append(r2_score(y[te], p.predict(X[te])))
        subgroup_results.append({"severity": str(sev), "block": block_name,
                                  "n": n, "r2": np.mean(r2s),
                                  "dtt_mean": sub_df.dtt_srt.mean(),
                                  "pta4_mean": sub_df.pta4_better.mean()})

sg_df = pd.DataFrame(subgroup_results)
sg_df.to_csv("results_subgroup.csv", index=False)
print(sg_df.to_string(index=False))

print("\nAll analyses complete.")
