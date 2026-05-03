"""
08: Cross-cohort validation OHHR → NHANES.

Train on OHHR (audiogram+age+sex → DTT poor SIN, calibrated LightGBM).
Apply to NHANES adults 70+ to get P(poor SIN).
Validate: does P(poor SIN) predict self-reported hearing difficulty (AUQ054, AUQ101)?
Also compare: audiogram-only vs audiogram+age models.

Key question: does the SHAP finding that age matters beyond the audiogram
replicate in an independent general-population cohort?
"""
import warnings
import numpy as np
import pandas as pd
import pyreadstat
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.base import clone
from scipy.stats import spearmanr, mannwhitneyu
import lightgbm as lgb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Load OHHR ─────────────────────────────────────────────────────────────────
ohhr = pd.read_csv("ohhr_features.csv")
ALL_HTL  = [c for c in ohhr.columns if c.startswith("ac_htl_")]
SHARED_PTA = ["pta4_better","pta4_worse","pta4_mean","pta4_asym"]

# Feature sets for transfer
FEATS_PTA_ONLY    = SHARED_PTA
FEATS_PTA_AGE     = SHARED_PTA + ["age"]
FEATS_PTA_AGE_SEX = SHARED_PTA + ["age","sex_m"]
FEATS_FULL_HTL    = ALL_HTL + SHARED_PTA + ["age","sex_m"]

y_ohhr = ohhr["dtt_poor_sin"].values

# NHANES has 500/1000/2000/4000 Hz per ear → match to OHHR column names
NHANES_AUDIO_COLS = (
    [f"ac_right_{f}" for f in ["500","1000","2000","4000"]] +
    [f"ac_left_{f}"  for f in ["500","1000","2000","4000"]]
)
OHHR_AUDIO_SHARED = (
    ["ac_htl_right_500","ac_htl_right_1000","ac_htl_right_2000","ac_htl_right_4000"] +
    ["ac_htl_left_500", "ac_htl_left_1000", "ac_htl_left_2000", "ac_htl_left_4000"]
)

# ── Load NHANES processed ─────────────────────────────────────────────────────
nhanes = pd.read_csv("nhanes_processed.csv")

# AUQ054: "How good is your hearing?"
# 1=Excellent, 2=Good, 3=A little trouble, 4=Moderate trouble,
# 5=A lot of trouble, 6=Deaf
# Positive class = any hearing trouble (>=3), including deaf (6)
nhanes["aud_difficulty"] = (nhanes["AUQ054"].isin([3.0,4.0,5.0,6.0])).astype(int)
# AUQ101: difficulty hearing in noise; NHANES coding is 1=Always … 5=Never
# (high difficulty = low value), so positive class = 1 or 2 (always/usually).
nhanes["sin_difficulty"] = (nhanes["AUQ101"] <= 2).astype(int)

# AUQ060: do you have difficulty hearing? 1=yes, 2=no
nhanes["aud_any"] = (nhanes["AUQ060"] == 1.0).astype(int)

valid_aud = nhanes["aud_difficulty"].notna()
valid_sin = nhanes["sin_difficulty"].notna()

print("=== NHANES validation targets ===")
print(f"  AUQ054 (hearing difficulty >=3): "
      f"{nhanes.loc[valid_aud,'aud_difficulty'].sum()} / {valid_aud.sum()} "
      f"({100*nhanes.loc[valid_aud,'aud_difficulty'].mean():.1f}%)")
print(f"  AUQ101 (SIN difficulty <=2):     "
      f"{nhanes.loc[valid_sin,'sin_difficulty'].sum()} / {valid_sin.sum()} "
      f"({100*nhanes.loc[valid_sin,'sin_difficulty'].mean():.1f}%)")
print(f"  AUQ060 (any hearing difficulty): "
      f"{nhanes['aud_any'].sum()} / {nhanes['aud_any'].notna().sum()} "
      f"({100*nhanes['aud_any'].mean():.1f}%)")

# ── Train on OHHR, predict on NHANES ─────────────────────────────────────────
def lgb_cls():
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                               subsample=0.8, colsample_bytree=0.8,
                               class_weight="balanced", random_state=42, verbose=-1)

def make_pipe(feats):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("model", lgb_cls())])

def transfer_predict(feats_ohhr, feats_nhanes=None):
    """Train on OHHR, get OOF probs + transfer probs on NHANES."""
    if feats_nhanes is None:
        feats_nhanes = feats_ohhr
    X_ohhr = ohhr[feats_ohhr].values
    # OOF on OHHR
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    oof_prob = np.zeros(len(y_ohhr))
    for tr, te in skf.split(X_ohhr, y_ohhr):
        p = clone(make_pipe(feats_ohhr))
        p.fit(X_ohhr[tr], y_ohhr[tr])
        oof_prob[te] = p.predict_proba(X_ohhr[te])[:, 1]
    ohhr_auroc = roc_auc_score(y_ohhr, oof_prob)

    # Full model: fit imputer on OHHR, train model, apply imputer to NHANES
    imp_train = SimpleImputer(strategy="median").fit(X_ohhr)
    X_ohhr_imp = imp_train.transform(X_ohhr)
    full_model = lgb_cls()
    full_model.fit(X_ohhr_imp, y_ohhr)

    # Apply OHHR-trained imputer medians to NHANES (true transfer: no refitting)
    X_nhanes = nhanes[feats_nhanes].values
    X_nhanes_imp = imp_train.transform(X_nhanes)
    transfer_prob = full_model.predict_proba(X_nhanes_imp)[:, 1]
    return oof_prob, ohhr_auroc, transfer_prob

print("\n=== Cross-cohort transfer results ===")
print(f"{'Feature set':<25} {'OHHR AUROC':>12} {'NHANES aud AUROC':>17} {'NHANES SIN AUROC':>17}")
print("-"*75)

transfer_rows = []
FEATS_MATCHED_HTL_O = OHHR_AUDIO_SHARED + ["age","sex_m"]
FEATS_MATCHED_HTL_N = NHANES_AUDIO_COLS + ["age","sex_m"]

feat_configs = [
    ("PTA4 only",             FEATS_PTA_ONLY,      FEATS_PTA_ONLY),
    ("PTA4 + Age",            FEATS_PTA_AGE,       FEATS_PTA_AGE),
    ("PTA4 + Age + Sex",      FEATS_PTA_AGE_SEX,   FEATS_PTA_AGE_SEX),
    ("Matched HTL + Age",     FEATS_MATCHED_HTL_O, FEATS_MATCHED_HTL_N),
]

all_transfer_probs = {}
for name, feats_o, feats_n in feat_configs:
    # Ensure NHANES has all needed columns
    missing = [f for f in feats_n if f not in nhanes.columns]
    if missing:
        print(f"  {name}: missing NHANES cols {missing}, skipping")
        continue

    oof_prob, ohhr_auc, t_prob = transfer_predict(feats_o, feats_n)
    all_transfer_probs[name] = t_prob

    # Validate against NHANES targets
    # AUQ054
    mask_a = valid_aud & nhanes["AUQ054"].notna()
    auc_aud = roc_auc_score(nhanes.loc[mask_a, "aud_difficulty"], t_prob[mask_a])
    sp_aud, p_aud = spearmanr(nhanes.loc[mask_a, "AUQ054"], t_prob[mask_a])

    # AUQ101 (SIN)
    mask_s = valid_sin & nhanes["AUQ101"].notna()
    if mask_s.sum() > 50:
        auc_sin = roc_auc_score(nhanes.loc[mask_s, "sin_difficulty"], t_prob[mask_s])
        sp_sin, p_sin = spearmanr(nhanes.loc[mask_s, "AUQ101"], t_prob[mask_s])
    else:
        auc_sin = sp_sin = p_sin = np.nan

    print(f"  {name:<25} {ohhr_auc:12.4f} {auc_aud:17.4f} {auc_sin:17.4f}")
    transfer_rows.append({"feature_set": name, "ohhr_auroc": ohhr_auc,
                          "nhanes_aud_auroc": auc_aud, "nhanes_sin_auroc": auc_sin,
                          "spearman_aud": sp_aud, "p_aud": p_aud})

transfer_df = pd.DataFrame(transfer_rows)
transfer_df.to_csv("results_crosscohort.csv", index=False)
print("\nSaved: results_crosscohort.csv")

# ── Mann-Whitney: NHANES predicted prob by difficulty level ───────────────────
print("\n=== Predicted P(poor SIN) by NHANES hearing difficulty category ===")
best_prob = all_transfer_probs.get("PTA4 + Age", all_transfer_probs.get("PTA4 only"))
aud_cats = nhanes["AUQ054"].dropna()
print(f"  Category means (AUQ054, 1=excellent..6=deaf):")
for cat in sorted(aud_cats.unique()):
    if cat > 6: continue
    mask = nhanes["AUQ054"] == cat
    if mask.sum() < 5: continue
    print(f"    {cat:.0f}: mean P={best_prob[mask].mean():.3f}  n={mask.sum()}")

# Mann-Whitney: never vs often/always
never  = nhanes["AUQ054"] <= 2
often  = nhanes["AUQ054"] >= 4
if never.sum() > 10 and often.sum() > 10:
    stat, pval = mannwhitneyu(best_prob[never], best_prob[often], alternative="less")
    print(f"\n  Mann-Whitney (never vs often): U={stat:.0f}, p={pval:.4f}")

# ── Figure: NHANES validation ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel A: bar chart of AUROC for aud_difficulty by feature set
ax = axes[0]
bar_names = [r["feature_set"] for r in transfer_rows]
bar_vals  = [r["nhanes_aud_auroc"] for r in transfer_rows]
colors    = ["#4C72B0","#55A868","#DD8452","#C44E52"][:len(bar_names)]
bars = ax.bar(range(len(bar_names)), bar_vals, color=colors, edgecolor="k", linewidth=0.6)
ax.set_xticks(range(len(bar_names)))
ax.set_xticklabels([n.replace(" + ","\n+\n").replace(" only","") for n in bar_names], fontsize=8)
ax.set_ylabel("AUROC (NHANES, hearing difficulty)")
ax.set_title("(a) Cross-cohort transfer AUROC", fontweight="bold")
ax.set_ylim([0.5, 0.9])
ax.axhline(0.5, color="k", linewidth=0.8, linestyle="--", alpha=0.4, label="Chance")
for bar, val in zip(bars, bar_vals):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.005, f"{val:.3f}", ha="center", fontsize=8)

# Panel B: predicted score distributions by AUQ054 category
ax2 = axes[1]
if best_prob is not None:
    cat_data = []
    cat_labels = []
    for cat in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        mask = nhanes["AUQ054"] == cat
        if mask.sum() < 5: continue
        cat_data.append(best_prob[mask])
        cat_labels.append(f"Cat {cat:.0f}\n(n={mask.sum()})")
    ax2.boxplot(cat_data, labels=cat_labels, patch_artist=True,
                boxprops=dict(facecolor="#55A868", alpha=0.6),
                medianprops=dict(color="k", linewidth=2))
    ax2.set_ylabel("Predicted P(poor SIN)")
    ax2.set_xlabel("NHANES AUQ054 (1=excellent, 5=a lot of trouble, 6=deaf)")
    ax2.set_title("(b) Predicted risk by difficulty category", fontweight="bold")

# Panel C: calibration plot (OHHR OOF)
ax3 = axes[2]
if "PTA4 + Age" in all_transfer_probs:
    oof_p, ohhr_auc_pa, _ = transfer_predict(FEATS_PTA_AGE, FEATS_PTA_AGE)
    oof_p0, ohhr_auc_p0, _ = transfer_predict(FEATS_PTA_ONLY, FEATS_PTA_ONLY)
    for label, oof, color, ls in [
        ("PTA4 only",   oof_p0, "#4C72B0", "--"),
        ("PTA4 + Age",  oof_p,  "#55A868", "-"),
    ]:
        frac, mean_p = calibration_curve(y_ohhr, oof, n_bins=8, strategy="quantile")
        ax3.plot(mean_p, frac, f"o{ls}", color=color, linewidth=2, markersize=5, label=label)
    ax3.plot([0,1],[0,1],"k:", linewidth=1, label="Perfect")
    ax3.set_xlabel("Mean predicted probability")
    ax3.set_ylabel("Observed fraction positive")
    ax3.set_title("(c) Calibration on OHHR (OOF)", fontweight="bold")
    ax3.legend(fontsize=9)

plt.tight_layout()
plt.savefig("fig5_crosscohort.pdf", bbox_inches="tight", dpi=200)
plt.savefig("fig5_crosscohort.png", bbox_inches="tight", dpi=200)
plt.close()
print("\nSaved: fig5_crosscohort.pdf/png")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n=== Key cross-cohort finding ===")
r0 = transfer_df[transfer_df.feature_set=="PTA4 only"].iloc[0]
r1 = transfer_df[transfer_df.feature_set=="PTA4 + Age"].iloc[0]
print(f"  PTA4 only:    NHANES AUROC = {r0['nhanes_aud_auroc']:.4f}")
print(f"  PTA4 + Age:   NHANES AUROC = {r1['nhanes_aud_auroc']:.4f}")
delta = r1['nhanes_aud_auroc'] - r0['nhanes_aud_auroc']
print(f"  Delta (adding Age): {delta:+.4f}")
print(f"\n  Interpretation: Adding age {'IMPROVES' if delta>0 else 'does not improve'} "
      f"cross-cohort hearing difficulty prediction by {abs(delta):.4f} AUROC")
