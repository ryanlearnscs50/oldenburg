"""
07: Cross-cohort validation using NHANES 2017-2018.
Strategy:
  - Load NHANES audiometry (AUX_J), hearing questionnaire (AUQ_J), demographics (DEMO_J)
  - Build shared features: PTA4 better ear, PTA4 worse ear, age, sex
  - Target: self-reported hearing condition (AUQ054: "How good is your hearing?",
            1=excellent…6=deaf) and AUQ010: "Wear hearing aid now?"
  - Train on OHHR (audiogram + demos -> binary hearing difficulty proxy)
  - Validate on NHANES (predicting self-reported hearing difficulty from audiogram)
  - Key question: does a model trained on OHHR's objective SIN data
    transfer to predict self-reported hearing difficulty in NHANES?
"""
import warnings
import numpy as np
import pandas as pd
import pyreadstat                    # SAS XPT reader
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score
from sklearn.calibration import calibration_curve
from sklearn.base import clone
import lightgbm as lgb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Install xport if needed ────────────────────────────────────────────────────
# ── Load NHANES XPT files ──────────────────────────────────────────────────────
def read_xpt(path):
    df, meta = pyreadstat.read_xport(path)
    return df

print("Loading NHANES files...")
aux  = read_xpt("nhanes_AUX_J.xpt")
auq  = read_xpt("nhanes_AUQ_J.xpt")
demo = read_xpt("nhanes_DEMO_J.xpt")
print(f"  AUX_J (audiometry):  {aux.shape}")
print(f"  AUQ_J (questionnaire): {auq.shape}")
print(f"  DEMO_J (demographics): {demo.shape}")
print(f"\nAUX columns (sample): {list(aux.columns[:20])}")
print(f"AUQ columns (sample): {list(auq.columns[:20])}")

# ── Explore questionnaire for hearing difficulty items ─────────────────────────
print("\nAUQ columns with hearing difficulty:")
for col in auq.columns:
    if "AUQ" in col:
        vc = auq[col].value_counts().head(4)
        if len(vc) > 0:
            print(f"  {col}: {vc.to_dict()}")

# ── Build NHANES audiogram features ───────────────────────────────────────────
# Right ear: AUXU500R, AUXU1K1R, AUXU2KR, AUXU4KR
# Left ear:  AUXU500L, AUXU1K1L, AUXU2KL, AUXU4KL
right_freqs = {"500":  "AUXU500R",  "1000": "AUXU1K1R",
               "2000": "AUXU2KR",   "4000": "AUXU4KR"}
left_freqs  = {"500":  "AUXU500L",  "1000": "AUXU1K1L",
               "2000": "AUXU2KL",   "4000": "AUXU4KL"}

audio = aux[["SEQN"] + list(right_freqs.values()) + list(left_freqs.values())].copy()
audio.columns = ["SEQN"] + \
    [f"ac_right_{f}" for f in right_freqs] + \
    [f"ac_left_{f}" for f in left_freqs]

# Replace 888/999 (special codes) with NaN
for col in audio.columns[1:]:
    audio[col] = pd.to_numeric(audio[col], errors="coerce")
    audio.loc[audio[col] >= 110, col] = np.nan  # >110 dB HL is out of range

# PTA4 per ear
r_cols = [f"ac_right_{f}" for f in ["500","1000","2000","4000"]]
l_cols = [f"ac_left_{f}" for f in ["500","1000","2000","4000"]]
audio["pta4_right"] = audio[r_cols].mean(axis=1)
audio["pta4_left"]  = audio[l_cols].mean(axis=1)
audio["pta4_better"] = audio[["pta4_right","pta4_left"]].min(axis=1)
audio["pta4_worse"]  = audio[["pta4_right","pta4_left"]].max(axis=1)
audio["pta4_mean"]   = audio[["pta4_right","pta4_left"]].mean(axis=1)
audio["pta4_asym"]   = (audio["pta4_right"] - audio["pta4_left"]).abs()

print(f"\nNHANES audiogram rows with valid PTA4: {audio['pta4_better'].notna().sum()}")
print(f"PTA4 better ear: mean={audio.pta4_better.mean():.1f}, sd={audio.pta4_better.std():.1f}")

# ── Demographics ───────────────────────────────────────────────────────────────
dem_sub = demo[["SEQN","RIDAGEYR","RIAGENDR"]].copy()
dem_sub.columns = ["SEQN","age","sex"]
dem_sub["sex_m"] = (dem_sub["sex"] == 1).astype(int)
dem_sub = dem_sub.drop(columns=["sex"])

# ── Hearing questionnaire ──────────────────────────────────────────────────────
# AUQ054: "How good is your hearing?" 1=Excellent, 2=Good, 3=A little trouble,
#          4=Moderate trouble, 5=A lot of trouble, 6=Deaf; 7/9=missing
# Check available columns
print("\nSearhing for key hearing difficulty columns...")
target_cols = []
for col in auq.columns:
    vals = auq[col].dropna().unique()
    n_unique = len(vals)
    if 2 <= n_unique <= 8:
        print(f"  {col}: values={sorted(vals[:6].tolist())} n={len(auq[col].dropna())}")
        target_cols.append(col)

# ── Merge ─────────────────────────────────────────────────────────────────────
nhanes = audio.merge(dem_sub, on="SEQN", how="inner")
nhanes = nhanes.merge(auq, on="SEQN", how="left")

# Keep only adults >= 20 with valid audiogram
nhanes = nhanes[(nhanes.age >= 20) & nhanes.pta4_better.notna()]
print(f"\nNHANES merged dataset: {nhanes.shape}")
print(f"Age range: {nhanes.age.min():.0f}-{nhanes.age.max():.0f}, mean={nhanes.age.mean():.1f}")

nhanes.to_csv("nhanes_processed.csv", index=False)
print("Saved: nhanes_processed.csv")

# ── Cross-cohort analysis setup ────────────────────────────────────────────────
ohhr = pd.read_csv("ohhr_features.csv")

# Shared features available in both datasets
SHARED_FEATS = ["pta4_better", "pta4_worse", "pta4_mean", "pta4_asym", "age", "sex_m"]

print("\n=== OHHR vs NHANES descriptor comparison ===")
for feat in ["pta4_better", "age"]:
    oh_m = ohhr[feat].mean(); oh_s = ohhr[feat].std()
    nh_m = nhanes[feat].mean(); nh_s = nhanes[feat].std()
    print(f"  {feat:15s}: OHHR {oh_m:.1f}±{oh_s:.1f}  |  NHANES {nh_m:.1f}±{nh_s:.1f}")

print("\nNHANES severity distribution:")
cuts = pd.cut(nhanes.pta4_better,
              bins=[-np.inf, 25, 40, 60, np.inf],
              labels=["<=25","26-40","41-60",">60"])
print(cuts.value_counts().sort_index())
print(f"\nNHANES poor hearing (PTA4 better > 40 dB): "
      f"{(nhanes.pta4_better > 40).sum()} / {len(nhanes)}")

# ── Cross-cohort: train on OHHR (PTA->poor SIN), test on NHANES ───────────────
# OHHR: label is dtt_poor_sin (DTT SRT > -2 dB)
# NHANES: need a proxy label. We'll build a PTA-based "impaired" label (PTA4 > 25 dB)
# for evaluation, and also use questionnaire items if available.

# First check what questionnaire columns are meaningful
print("\n=== Available NHANES hearing questionnaire items ===")
for col in auq.columns:
    try:
        vc = nhanes[col].value_counts().sort_index()
        if 1 < len(vc) <= 7 and nhanes[col].notna().sum() > 100:
            print(f"  {col}: {vc.to_dict()}")
    except Exception:
        pass
