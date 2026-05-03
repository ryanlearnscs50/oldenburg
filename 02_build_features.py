"""
Build the flat per-participant feature matrix from OHHR JSON files.
Outputs: ohhr_features.csv  (581 rows × N feature columns + outcome columns)
"""
import json, os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA = "ohhr_data/data"

def load(fname):
    with open(os.path.join(DATA, fname)) as f:
        return pd.DataFrame(json.load(f))

# ── Load raw tables ───────────────────────────────────────────────────────────
anamnesis = load("anamnesis.json")
audiogram = load("audiogram.json")
ag_line   = load("audiogram_line.json")
ag_point  = load("audiogram_point.json")
dtt       = load("digit_triplets_test.json")
gst       = load("goettingen_sentence_test.json")
demtect   = load("demtect.json")
verbal    = load("verbal_intell.json")
ls_data   = load("loudness_scaling_data.json")
ls_fit    = load("loudness_scaling_fit_brand.json")
ls_id     = load("loudness_scaling.json")
soc       = load("soc_econ_stat.json")
homeqsn   = load("homeqsn.json")

# ── BLOCK 1: Audiogram features ───────────────────────────────────────────────
FREQS = [250, 500, 750, 1000, 1500, 2000, 4000]

def extract_thresholds(line_type, transducer="ac", masking_filter=None):
    """Return wide table: clientid + {transducer}_{side}_{freq} columns."""
    filt = ag_line[(ag_line.type == line_type) &
                   (ag_line.transducertype == transducer)]
    if masking_filter is not None:
        filt = filt[filt.masking == masking_filter]
    pts = ag_point.merge(filt[["audiogramlineid","audiogramid","side"]],
                         on="audiogramlineid")
    pts = pts[pts.frequency.isin(FREQS)]
    pts = pts.merge(audiogram[["audiogramid","clientid"]], on="audiogramid")
    wide = pts.pivot_table(index="clientid", columns=["side","frequency"],
                           values="level", aggfunc="mean")
    wide.columns = [f"{transducer}_{line_type}_{s}_{f}" for s, f in wide.columns]
    return wide.reset_index()

# Air-conduction HTL (unmasked) — primary audiogram
htl = extract_thresholds("htl", "ac", masking_filter=False)

# Air-conduction UCL (uncomfortable loudness level)
ucl = extract_thresholds("ucl", "ac")

# Derived PTA summaries
for side in ["left","right"]:
    cols4 = [f"ac_htl_{side}_{f}" for f in [500,1000,2000,4000] if f"ac_htl_{side}_{f}" in htl]
    if cols4:
        htl[f"pta4_{side}"] = htl[cols4].mean(axis=1)
    cols_hf = [f"ac_htl_{side}_{f}" for f in [2000,4000] if f"ac_htl_{side}_{f}" in htl]
    cols_lf = [f"ac_htl_{side}_{f}" for f in [250,500,1000] if f"ac_htl_{side}_{f}" in htl]
    if cols_hf and cols_lf:
        htl[f"slope_{side}"] = htl[cols_hf].mean(axis=1) - htl[cols_lf].mean(axis=1)

htl["pta4_better"] = htl[["pta4_left","pta4_right"]].min(axis=1)
htl["pta4_worse"]  = htl[["pta4_left","pta4_right"]].max(axis=1)
htl["pta4_mean"]   = htl[["pta4_left","pta4_right"]].mean(axis=1)
htl["pta4_asym"]   = (htl["pta4_right"] - htl["pta4_left"]).abs()

# Average UCL per side (across frequencies)
for side in ["left","right"]:
    ucl_cols = [c for c in ucl.columns if f"ac_ucl_{side}_" in c]
    if ucl_cols:
        ucl[f"ucl_mean_{side}"] = ucl[ucl_cols].mean(axis=1)

# ── BLOCK 2: Loudness scaling fit parameters ──────────────────────────────────
ls_merge = ls_fit.merge(
    ls_data[["loudnessscalingdataid","loudnessscalingid","side"]], on="loudnessscalingdataid"
).merge(
    ls_id[["loudnessscalingid","clientid"]], on="loudnessscalingid"
)
ls_wide = ls_merge.pivot_table(
    index="clientid", columns="side", values=["mlow","mhigh","mcut"], aggfunc="mean"
)
ls_wide.columns = [f"ls_{v}_{s}" for v, s in ls_wide.columns]
ls_wide = ls_wide.reset_index()

# Derived: inter-ear asymmetry in mcut
if "ls_mcut_left" in ls_wide and "ls_mcut_right" in ls_wide:
    ls_wide["ls_mcut_asym"] = (ls_wide["ls_mcut_right"] - ls_wide["ls_mcut_left"]).abs()

# ── BLOCK 3: Cognition ────────────────────────────────────────────────────────
dem = demtect.groupby("clientid")["sum_pnt"].mean().rename("demtect").reset_index()
vi  = verbal.groupby("clientid")["score"].mean().rename("verbal_iq").reset_index()

# ── BLOCK 4: Demographics ──────────────────────────────────────────────────────
ana = anamnesis[["clientid","sex","birthday_year","ha_supply_status",
                 "hl_progressing","hl_fluctuating","hl_family_often"]].copy()
ana["age"] = 2014 - ana["birthday_year"]
ana["sex_m"] = (ana["sex"] == "m").astype(int)
ana = ana.drop(columns=["sex","birthday_year"]).groupby("clientid").first().reset_index()

soc_sub = soc[["clientid","school","education","net_income"]].groupby("clientid").first().reset_index()

# ── OUTCOMES ──────────────────────────────────────────────────────────────────
# Primary: Digit Triplet Test SRT (binaural, free-field) — LOWER = BETTER
dtt_y = (dtt[(dtt.side=="binaural") & (dtt.transducertype=="ff")]
         .groupby("clientid")["SRT"].median().rename("dtt_srt").reset_index())

# Secondary: Göttingen Sentence Test SRT (binaural, free-field)
gst_y = (gst[(gst.side=="binaural") & (gst.transducertype=="ff")]
         .groupby("clientid")["SRT"].median().rename("gst_srt").reset_index())

# Self-report: hearing problems in noise (1=never, 5=always)
hq = homeqsn[["clientid","hp_quiet","hp_noise","ha_use_curr"]].copy()
hq = hq.groupby("clientid").first().reset_index()

# ── MERGE ─────────────────────────────────────────────────────────────────────
clients = pd.DataFrame({"clientid": sorted(anamnesis.clientid.unique())})
df = (clients
      .merge(htl,     on="clientid", how="left")
      .merge(ucl[["clientid"] + [c for c in ucl.columns if c.startswith("ucl_mean")]],
             on="clientid", how="left")
      .merge(ls_wide,  on="clientid", how="left")
      .merge(dem,      on="clientid", how="left")
      .merge(vi,       on="clientid", how="left")
      .merge(ana,      on="clientid", how="left")
      .merge(soc_sub,  on="clientid", how="left")
      .merge(dtt_y,    on="clientid", how="left")
      .merge(gst_y,    on="clientid", how="left")
      .merge(hq,       on="clientid", how="left")
      )

# Binary classification target: "poor SIN" if DTT SRT > median + 0.5*SD
# (above -2 dB SNR is clearly impaired range for the Oldenburg DTT)
dtt_cutoff = -2.0   # dB SNR; clinical convention for "poor" SIN
df["dtt_poor_sin"] = (df["dtt_srt"] > dtt_cutoff).astype(int)

print(f"Feature matrix shape: {df.shape}")
print(f"\nOutcome distributions:")
print(f"  DTT SRT: mean={df.dtt_srt.mean():.2f}, sd={df.dtt_srt.std():.2f}, "
      f"range=[{df.dtt_srt.min():.1f}, {df.dtt_srt.max():.1f}]")
print(f"  GST SRT: mean={df.gst_srt.mean():.2f}, sd={df.gst_srt.std():.2f}")
print(f"  Poor SIN (DTT SRT > {dtt_cutoff}): {df.dtt_poor_sin.sum()} / {len(df)} "
      f"({100*df.dtt_poor_sin.mean():.1f}%)")

print(f"\nFeature missingness (% missing):")
feat_cols = [c for c in df.columns if c not in
             ["clientid","dtt_srt","gst_srt","dtt_poor_sin","hp_quiet","hp_noise","ha_use_curr"]]
miss = (df[feat_cols].isna().mean() * 100).sort_values(ascending=False)
print(miss[miss > 0].to_string())
print(f"\nFeatures with 0% missing: {(miss==0).sum()}")

df.to_csv("ohhr_features.csv", index=False)
print("\nSaved: ohhr_features.csv")
print(f"Total features: {len(feat_cols)}")
print(f"Feature columns: {feat_cols}")
