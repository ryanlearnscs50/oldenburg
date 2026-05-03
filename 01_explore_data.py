"""
OHHR Data Exploration
Loads all relevant tables, joins them into a flat per-participant feature matrix,
and computes descriptive statistics.
"""
import json
import os
import numpy as np
import pandas as pd

DATA_DIR = "ohhr_data/data"

def load(fname):
    with open(os.path.join(DATA_DIR, fname)) as f:
        return pd.DataFrame(json.load(f))

# ── Core tables ──────────────────────────────────────────────────────────────
anamnesis   = load("anamnesis.json")
audiogram   = load("audiogram.json")
ag_line     = load("audiogram_line.json")
ag_point    = load("audiogram_point.json")
dtt         = load("digit_triplets_test.json")
gst         = load("goettingen_sentence_test.json")
demtect     = load("demtect.json")
verbal      = load("verbal_intell.json")
ls_fit      = load("loudness_scaling_fit_brand.json")
ls_data     = load("loudness_scaling_data.json")
soc         = load("soc_econ_stat.json")
homeqsn     = load("homeqsn.json")

print("=== TABLE SIZES ===")
for name, df in [("anamnesis",anamnesis),("audiogram",audiogram),("ag_line",ag_line),
                 ("ag_point",ag_point),("dtt",dtt),("gst",gst),("demtect",demtect),
                 ("verbal",verbal),("ls_fit",ls_fit),("ls_data",ls_data),
                 ("soc",soc),("homeqsn",homeqsn)]:
    print(f"  {name:20s}: {len(df):5d} rows  cols={list(df.columns)[:6]}")

# ── Build audiogram features ──────────────────────────────────────────────────
# Keep only air-conduction (ac), HTL type, unmasked
ac_lines = ag_line[
    (ag_line.transducertype == "ac") &
    (ag_line.type == "htl") &
    (ag_line.masking == False)
][["audiogramlineid", "audiogramid", "side"]].copy()

# Merge audiogram_point onto ac_lines
ac_pts = ag_point.merge(ac_lines, on="audiogramlineid")
# Keep standard audiometric frequencies
std_freqs = [250, 500, 1000, 2000, 3000, 4000, 6000, 8000]
ac_pts = ac_pts[ac_pts.frequency.isin(std_freqs)]

# Merge audiogram → clientid
ac_pts = ac_pts.merge(audiogram[["audiogramid","clientid"]], on="audiogramid")

# Pivot: one row per client, columns = {side}_{freq}
ac_wide = ac_pts.pivot_table(
    index="clientid", columns=["side","frequency"], values="level", aggfunc="mean"
)
ac_wide.columns = [f"ac_{s}_{f}" for s, f in ac_wide.columns]
ac_wide = ac_wide.reset_index()

print(f"\n=== Audiogram wide table: {ac_wide.shape} ===")
print("  Columns:", list(ac_wide.columns))

# ── 4-frequency PTA (0.5, 1, 2, 4 kHz) per ear ───────────────────────────────
for side in ["left", "right"]:
    cols = [f"ac_{side}_{f}" for f in [500, 1000, 2000, 4000] if f"ac_{side}_{f}" in ac_wide.columns]
    if cols:
        ac_wide[f"pta4_{side}"] = ac_wide[cols].mean(axis=1)
ac_wide["pta4_better"] = ac_wide[["pta4_left","pta4_right"]].min(axis=1)
ac_wide["pta4_worse"]  = ac_wide[["pta4_left","pta4_right"]].max(axis=1)
ac_wide["pta4_mean"]   = ac_wide[["pta4_left","pta4_right"]].mean(axis=1)

# ── DTT: binaural free-field SRT ─────────────────────────────────────────────
dtt_bin = dtt[(dtt.side == "binaural") & (dtt.transducertype == "ff")][["clientid","SRT"]].rename(columns={"SRT":"dtt_srt"})
print(f"\n=== DTT binaural ff: {len(dtt_bin)} records ===")
print(dtt_bin.dtt_srt.describe())

# Some clients may have multiple DTT records — take median
dtt_bin = dtt_bin.groupby("clientid")["dtt_srt"].median().reset_index()

# ── GST: binaural free-field SRT ─────────────────────────────────────────────
gst_bin = gst[(gst.side == "binaural") & (gst.transducertype == "ff")][["clientid","SRT"]].rename(columns={"SRT":"gst_srt"})
print(f"\n=== GST binaural ff: {len(gst_bin)} records ===")
print(gst_bin.gst_srt.describe())
gst_bin = gst_bin.groupby("clientid")["gst_srt"].median().reset_index()

# ── Cognitive: DemTect ────────────────────────────────────────────────────────
dem = demtect[["clientid","sum_pnt"]].rename(columns={"sum_pnt":"demtect_score"})
dem = dem.groupby("clientid")["demtect_score"].mean().reset_index()

# ── Verbal intelligence ───────────────────────────────────────────────────────
vi = verbal[["clientid","score"]].rename(columns={"score":"verbal_iq"})
vi = vi.groupby("clientid")["verbal_iq"].mean().reset_index()

# ── Loudness scaling: per-ear fit parameters ──────────────────────────────────
ls_data_sub = ls_data[["loudnessscalingdataid","loudnessscalingid","side"]].copy()
ls_fit_m = ls_fit.merge(ls_data_sub, on="loudnessscalingdataid")
ls_fit_m = ls_fit_m.merge(
    load("loudness_scaling.json")[["loudnessscalingid","clientid"]], on="loudnessscalingid"
)
ls_wide = ls_fit_m.pivot_table(
    index="clientid", columns="side", values=["mlow","mhigh","mcut"], aggfunc="mean"
)
ls_wide.columns = [f"ls_{v}_{s}" for v, s in ls_wide.columns]
ls_wide = ls_wide.reset_index()
print(f"\n=== Loudness scaling wide: {ls_wide.shape} ===")

# ── Anamnesis: age, sex, HA status ────────────────────────────────────────────
ana = anamnesis[["clientid","sex","birthday_year","ha_supply_status","hl_progressing","hl_fluctuating","hl_family_often"]].copy()
ana["age"] = 2014 - ana["birthday_year"]  # approximate; most data 2013-2015
ana["sex_m"] = (ana["sex"] == "m").astype(int)
ana = ana.drop(columns=["sex","birthday_year"])
ana = ana.groupby("clientid").first().reset_index()

# ── Socioeconomic ─────────────────────────────────────────────────────────────
soc_sub = soc[["clientid","school","education","net_income"]].groupby("clientid").first().reset_index()

# ── Home questionnaire: key self-report outcomes ──────────────────────────────
# hp_quiet, hp_noise: hearing problems (1=never .. 5=always)
# hh: hearing handicap composite
hq = homeqsn[["clientid","hp_quiet","hp_noise","hh","hh_s","hh_sn"]].groupby("clientid").first().reset_index()
print(f"\n=== Hearing handicap (hh) distribution ===")
print(hq["hh"].describe())
print(f"  Missing: {hq['hh'].isna().sum()}")

# ── Merge all into a flat feature table ──────────────────────────────────────
clients = pd.DataFrame({"clientid": sorted(anamnesis.clientid.unique())})
df = (clients
      .merge(ac_wide, on="clientid", how="left")
      .merge(dtt_bin,  on="clientid", how="left")
      .merge(gst_bin,  on="clientid", how="left")
      .merge(dem,      on="clientid", how="left")
      .merge(vi,       on="clientid", how="left")
      .merge(ls_wide,  on="clientid", how="left")
      .merge(ana,      on="clientid", how="left")
      .merge(soc_sub,  on="clientid", how="left")
      .merge(hq,       on="clientid", how="left")
      )

print(f"\n=== Final merged table: {df.shape} ===")
print(f"  Non-null counts per key column:")
for col in ["dtt_srt","gst_srt","demtect_score","verbal_iq","pta4_mean","age","hh"]:
    print(f"    {col:20s}: {df[col].notna().sum():3d} / {len(df)}")

# Save
df.to_csv("ohhr_flat.csv", index=False)
print("\nSaved: ohhr_flat.csv")
