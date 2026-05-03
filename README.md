# Predicting Speech-in-Noise Performance Beyond the Pure-Tone Audiogram

**A Machine Learning Benchmark on the Oldenburg Hearing Health Record (OHHR)**

Ryan Yi Sheng Neo — Independent Researcher — realryanneo@gmail.com

---

## Overview

This repository contains all analysis code for the paper:

> Neo, R.Y.S. "Predicting Speech-in-Noise Performance Beyond the Pure-Tone Audiogram: A Machine Learning Benchmark on the Oldenburg Hearing Health Record." *Scientific Reports* (submitted 2026).

We present the first systematic ML benchmark on the OHHR dataset, evaluating Ridge regression, Random Forest, LightGBM, and MLP across five progressively richer feature blocks for predicting Digit Triplet Test speech-reception threshold (DTT SRT).

**Key results:**
- Single binaural PTA4: R² = 0.71 · Full battery (RF): R² = 0.81 (MAE = 1.23 dB SNR)
- Benefit of extended assessment concentrated in mild hearing loss (26–40 dB HL)
- Age and loudness-growth slope are most informative beyond audiogram (SHAP)
- Poor SIN classification: AUROC = 0.978, Brier = 0.053
- Cross-cohort NHANES validation: AUROC 0.747 → 0.801 with age added

---

## Data

**OHHR dataset** — download directly from Zenodo (not included here):
- DOI: [10.5281/zenodo.16919812](https://doi.org/10.5281/zenodo.16919812)
- Licence: CC BY 4.0
- Extract to `ohhr_data/` in this directory

**NHANES 2017–2018** — download from CDC (not included here):
- Audiometry: [AUX_J.XPT](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/AUX_J.XPT)
- Questionnaire: [AUQ_J.XPT](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/AUQ_J.XPT)
- Demographics: [DEMO_J.XPT](https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT)
- Place `.XPT` files in `nhanes_raw/`

---

## Reproduction

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run scripts in order

```bash
python 01_explore_data.py        # Parse OHHR JSON → ohhr_flat.csv
python 02_build_features.py      # Feature engineering → ohhr_features.csv
python 03_ml_benchmark.py        # Main benchmark table + figures
python 04_calibration_and_minimal.py  # Calibration, minimal predictor, subgroup
python 05_paper_figures.py       # Final publication figures
python 06_nested_cv_tuning.py    # Nested CV / hyperparameter sensitivity
python 07_nhanes_validation.py   # NHANES preprocessing → nhanes_processed.csv
python 08_crosscohort_validation.py   # Cross-cohort transfer results
python 09_neural_baseline.py     # MLP + SHAP stability
python 10_statistical_tests.py   # Wilcoxon tests, bootstrap CIs
```

All results are saved as `results_*.csv`; all figures as `fig_*.png/pdf`.

### 3. Compile the paper

```bash
cd paper
pdflatex srep_main.tex
pdflatex srep_main.tex   # twice for references
```

---

## Repository structure

```
.
├── 01_explore_data.py
├── 02_build_features.py
├── 03_ml_benchmark.py
├── 04_calibration_and_minimal.py
├── 05_paper_figures.py
├── 06_nested_cv_tuning.py
├── 07_nhanes_validation.py
├── 08_crosscohort_validation.py
├── 09_neural_baseline.py
├── 10_statistical_tests.py
├── requirements.txt
├── ohhr_features.csv          # Derived feature matrix (581 × 41+)
├── results_*.csv              # All numerical results
├── shap_importance.csv
└── paper/
    ├── srep_main.tex          # Main manuscript (Scientific Reports format)
    ├── cover_letter.tex
    ├── fig2_benchmark.png
    ├── fig3_shap.png
    ├── fig4_subgroup_calibration.png
    ├── fig5_crosscohort.png
    └── fig6_model_comparison_stability.png
```

---

## Licence

Code: MIT Licence  
Results/figures: CC BY 4.0
