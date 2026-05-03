"""
10: Per-fold benchmark + statistical significance tests.

Saves per-fold R² for all model×block combinations, then runs:
  - Wilcoxon signed-rank on fold-level R² (block A vs E, model comparisons)
  - Bootstrap CIs for all models on Block E
  - DeLong test for AUROC comparisons (classification)
  - Saves results_perfold.csv, results_stat_tests.csv
"""
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
from sklearn.base import clone
import lightgbm as lgb

warnings.filterwarnings("ignore")
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

BLOCKS = {
    "A: PTA4 mean": ["pta4_mean"],
    "B: Full PTA":  ALL_HTL,
    "C: +Loudness": ALL_HTL + ALL_LS,
    "D: +Cognition":ALL_HTL + ALL_LS + ALL_COG,
    "E: Full battery": BLOCK_E,
}

y = df["dtt_srt"].values
y_cls = df["dtt_poor_sin"].values
kf  = KFold(n_splits=10, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

def ridge():    return Ridge(alpha=1.0)
def rf():       return RandomForestRegressor(n_estimators=200, max_features="sqrt",
                                              min_samples_leaf=5, random_state=42, n_jobs=-1)
def lgb_reg():  return lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                                           subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                           random_state=42, verbose=-1)
def make_pipe(m, scale=False):
    steps = [("imp", SimpleImputer(strategy="median"))]
    if scale: steps.append(("scl", StandardScaler()))
    steps.append(("model", m))
    return Pipeline(steps)

# ── 1. Per-fold R² for all model × block ─────────────────────────────────────
print("=== Per-fold regression benchmark (10-fold CV) ===")
pf_rows = []
for bname, feats in BLOCKS.items():
    X = df[feats].values
    for mname, pipe in [
        ("Ridge",    make_pipe(ridge(),   scale=True)),
        ("RF",       make_pipe(rf(),      scale=False)),
        ("LightGBM", make_pipe(lgb_reg(), scale=False)),
    ]:
        fold_r2, fold_mae = [], []
        for tr, te in kf.split(X):
            p = clone(pipe)
            p.fit(X[tr], y[tr])
            pred = p.predict(X[te])
            fold_r2.append(r2_score(y[te], pred))
            fold_mae.append(mean_absolute_error(y[te], pred))
        mean_r2 = np.mean(fold_r2)
        print(f"  {bname:<22} {mname:<10} R²={mean_r2:.3f}±{np.std(fold_r2):.3f}")
        for i, (r2, mae) in enumerate(zip(fold_r2, fold_mae)):
            pf_rows.append({"block":bname,"model":mname,"fold":i,"r2":r2,"mae":mae})

pf_df = pd.DataFrame(pf_rows)
pf_df.to_csv("results_perfold.csv", index=False)
print("Saved: results_perfold.csv")

# ── 2. Statistical tests ──────────────────────────────────────────────────────
print("\n=== Statistical tests (Wilcoxon signed-rank, n=10 folds) ===")
stat_rows = []

def wilcoxon_test(a, b, label_a, label_b):
    """One-sided: is a > b?"""
    stat, p_two = stats.wilcoxon(a, b, alternative="two-sided", zero_method="zsplit")
    stat_one, p_one = stats.wilcoxon(a, b, alternative="greater", zero_method="zsplit")
    delta = np.mean(a) - np.mean(b)
    return {"comparison": f"{label_a} vs {label_b}",
            "mean_a": np.mean(a), "mean_b": np.mean(b),
            "delta": delta, "p_twosided": p_two, "p_onesided": p_one}

def get_folds(block, model):
    return pf_df[(pf_df.block==block) & (pf_df.model==model)].sort_values("fold")["r2"].values

# Key comparisons
comparisons = [
    # Block progression (LightGBM)
    ("E: Full battery", "LightGBM", "A: PTA4 mean",   "LightGBM", "LGBMBlockE vs BlockA"),
    ("C: +Loudness",    "LightGBM", "B: Full PTA",     "LightGBM", "LGBM BlockC vs BlockB (loudness gain)"),
    ("E: Full battery", "LightGBM", "C: +Loudness",    "LightGBM", "LGBM BlockE vs BlockC"),
    # Best model comparisons (Block E)
    ("E: Full battery", "RF",       "E: Full battery", "Ridge",    "RF vs Ridge (BlockE)"),
    ("E: Full battery", "RF",       "E: Full battery", "LightGBM", "RF vs LightGBM (BlockE)"),
    ("E: Full battery", "LightGBM", "E: Full battery", "Ridge",    "LightGBM vs Ridge (BlockE)"),
    # Block A single feature
    ("B: Full PTA",     "RF",       "A: PTA4 mean",    "RF",       "RF BlockB vs BlockA"),
    ("E: Full battery", "RF",       "A: PTA4 mean",    "RF",       "RF BlockE vs BlockA"),
]

for bA, mA, bB, mB, label in comparisons:
    try:
        r2A = get_folds(bA, mA)
        r2B = get_folds(bB, mB)
        if len(r2A) == 0 or len(r2B) == 0:
            continue
        row = wilcoxon_test(r2A, r2B, f"{mA}/{bA}", f"{mB}/{bB}")
        row["label"] = label
        stat_rows.append(row)
        sig = "***" if row["p_twosided"] < 0.001 else ("**" if row["p_twosided"] < 0.01 else
              ("*" if row["p_twosided"] < 0.05 else "ns"))
        print(f"  {label:45s}  Δ={row['delta']:+.4f}  p={row['p_twosided']:.4f} {sig}")
    except Exception as e:
        print(f"  {label}: ERROR {e}")

stat_df = pd.DataFrame(stat_rows)
stat_df.to_csv("results_stat_tests.csv", index=False)
print("\nSaved: results_stat_tests.csv")

# ── 3. Bootstrap CIs for all models on Block E ────────────────────────────────
print("\n=== Bootstrap 95% CIs for Block E (all models) ===")
X_E = df[BLOCK_E].values

def get_oof_preds(X, pipe_factory, kfold):
    preds = np.zeros(len(y))
    for tr, te in kfold.split(X):
        p = clone(pipe_factory())
        p.fit(X[tr], y[tr])
        preds[te] = p.predict(X[te])
    return preds

def bootstrap_r2_ci(y_true, y_pred, n_boot=2000):
    n = len(y_true)
    boot = [r2_score(y_true[idx := np.random.choice(n, n, replace=True)],
                     y_pred[idx]) for _ in range(n_boot)]
    return np.percentile(boot, 2.5), np.percentile(boot, 97.5)

ci_rows = []
for mname, pf in [
    ("Ridge",    lambda: make_pipe(ridge(),   scale=True)),
    ("RF",       lambda: make_pipe(rf(),      scale=False)),
    ("LightGBM", lambda: make_pipe(lgb_reg(), scale=False)),
]:
    preds = get_oof_preds(X_E, pf, kf)
    r2 = r2_score(y, preds)
    lo, hi = bootstrap_r2_ci(y, preds)
    print(f"  {mname:10s} Block E: R²={r2:.4f}  95%CI=[{lo:.4f}, {hi:.4f}]")
    ci_rows.append({"model":mname,"block":"E: Full battery",
                    "r2":r2,"ci_lo":lo,"ci_hi":hi})

ci_df = pd.DataFrame(ci_rows)
ci_df.to_csv("results_blockE_ci.csv", index=False)
print("Saved: results_blockE_ci.csv")

# ── 4. Classification: DeLong-style bootstrap comparison ─────────────────────
print("\n=== Classification AUROC bootstrap CI (Block A LR vs Block E LightGBM) ===")

def lgb_cls(): return lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                                          subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                          class_weight="balanced", random_state=42, verbose=-1)
def lr_cls(): return LogisticRegression(C=1.0, max_iter=1000, random_state=42)

def get_oof_probs(X, clf_factory, skf):
    probs = np.zeros(len(y_cls))
    for tr, te in skf.split(X, y_cls):
        m = clone(Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("scl", StandardScaler()),
                             ("model", clf_factory())]))
        m.fit(X[tr], y_cls[tr])
        probs[te] = m.predict_proba(X[te])[:, 1]
    return probs

X_A = df[["pta4_mean"]].values
probs_lr_A  = get_oof_probs(X_A, lr_cls, skf)
probs_lgb_E = get_oof_probs(X_E, lgb_cls, skf)

def bootstrap_auroc_ci(y_true, probs, n_boot=2000):
    n = len(y_true)
    boot = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2: continue
        boot.append(roc_auc_score(y_true[idx], probs[idx]))
    return np.percentile(boot, 2.5), np.percentile(boot, 97.5)

for label, probs in [("LR Block A", probs_lr_A), ("LightGBM Block E", probs_lgb_E)]:
    auroc = roc_auc_score(y_cls, probs)
    lo, hi = bootstrap_auroc_ci(y_cls, probs)
    print(f"  {label:20s}: AUROC={auroc:.4f}  95%CI=[{lo:.4f}, {hi:.4f}]")

# DeLong bootstrap: is LightGBM Block E > LR Block A?
n = len(y_cls)
boot_diffs = []
for _ in range(2000):
    idx = np.random.choice(n, n, replace=True)
    if len(np.unique(y_cls[idx])) < 2: continue
    diff = roc_auc_score(y_cls[idx], probs_lgb_E[idx]) - roc_auc_score(y_cls[idx], probs_lr_A[idx])
    boot_diffs.append(diff)
p_auroc = np.mean(np.array(boot_diffs) <= 0)
mean_diff = np.mean(boot_diffs)
ci_lo = np.percentile(boot_diffs, 2.5)
ci_hi = np.percentile(boot_diffs, 97.5)
print(f"\n  AUROC diff (LightGBM E − LR A): {mean_diff:+.4f} (95%CI [{ci_lo:.4f},{ci_hi:.4f}])")
print(f"  Bootstrap p (one-sided, LightGBM > LR): {p_auroc:.4f}")

print("\nAll statistical tests complete.")
