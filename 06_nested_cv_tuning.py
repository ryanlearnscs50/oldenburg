"""
06: Nested cross-validation with Optuna hyperparameter tuning.
Outer 10-fold: unbiased performance estimate.
Inner 5-fold: HP search per outer fold.
Models: LightGBM, RF, Ridge (Ridge alpha tuned too).
Also computes paired Wilcoxon signed-rank tests between models.
"""
import warnings, json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.base import clone
from scipy.stats import wilcoxon
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")
np.random.seed(42)

df = pd.read_csv("ohhr_features.csv")
ALL_HTL  = [c for c in df.columns if c.startswith("ac_htl_")]
ALL_LS   = [c for c in df.columns if c.startswith("ls_") or c.startswith("ucl_mean_")]
ALL_COG  = ["demtect", "verbal_iq"]
ALL_DEMO = ["age","sex_m","school","education","net_income",
            "ha_supply_status","hl_progressing","hl_fluctuating","hl_family_often",
            "pta4_left","pta4_right","pta4_better","pta4_worse","pta4_asym",
            "slope_left","slope_right"]
BLOCK_E  = list(dict.fromkeys(ALL_HTL + ALL_LS + ALL_COG + ALL_DEMO))
BLOCK_A  = ["pta4_mean"]

y = df["dtt_srt"].values
outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)

# ── Optuna objective factories ─────────────────────────────────────────────────
def make_lgb_objective(X_tr, y_tr, inner_cv):
    def objective(trial):
        params = dict(
            n_estimators    = trial.suggest_int("n_estimators", 100, 800),
            learning_rate   = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves      = trial.suggest_int("num_leaves", 16, 96),
            min_child_samples = trial.suggest_int("min_child_samples", 5, 40),
            subsample       = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree= trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda      = trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
            reg_alpha       = trial.suggest_float("reg_alpha", 0.0, 5.0),
        )
        imp = SimpleImputer(strategy="median")
        X_imp = imp.fit_transform(X_tr)
        model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
        scores = cross_val_score(model, X_imp, y_tr, cv=inner_cv,
                                 scoring="r2", n_jobs=-1)
        return scores.mean()
    return objective

def make_rf_objective(X_tr, y_tr, inner_cv):
    def objective(trial):
        params = dict(
            n_estimators   = trial.suggest_int("n_estimators", 100, 600),
            max_depth      = trial.suggest_int("max_depth", 3, 20),
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 20),
            max_features   = trial.suggest_float("max_features", 0.3, 1.0),
        )
        imp = SimpleImputer(strategy="median")
        scl = StandardScaler()
        X_proc = scl.fit_transform(imp.fit_transform(X_tr))
        model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        scores = cross_val_score(model, X_proc, y_tr, cv=inner_cv,
                                 scoring="r2", n_jobs=-1)
        return scores.mean()
    return objective

def make_ridge_objective(X_tr, y_tr, inner_cv):
    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.001, 100.0, log=True)
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("scl", StandardScaler()),
                         ("model", Ridge(alpha=alpha))])
        scores = cross_val_score(pipe, X_tr, y_tr, cv=inner_cv,
                                 scoring="r2", n_jobs=-1)
        return scores.mean()
    return objective

# ── Nested CV ─────────────────────────────────────────────────────────────────
N_TRIALS = 80   # Optuna trials per outer fold

results_nested = {m: {"r2": [], "mae": [], "best_params": []}
                  for m in ["LightGBM", "RF", "Ridge"]}

X_full = df[BLOCK_E].values
X_base = df[BLOCK_A].values

print(f"Nested CV: 10 outer folds × 5 inner folds × {N_TRIALS} Optuna trials")
print("This may take several minutes...\n")

for fold_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X_full)):
    X_tr, X_te = X_full[tr_idx], X_full[te_idx]
    y_tr, y_te = y[tr_idx],      y[te_idx]
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=fold_idx)

    # ── LightGBM ──
    study_lgb = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=fold_idx))
    study_lgb.optimize(make_lgb_objective(X_tr, y_tr, inner_cv),
                       n_trials=N_TRIALS, show_progress_bar=False)
    bp = study_lgb.best_params
    imp = SimpleImputer(strategy="median")
    X_tr_imp = imp.fit_transform(X_tr)
    X_te_imp = imp.transform(X_te)
    m_lgb = lgb.LGBMRegressor(**bp, random_state=42, verbose=-1)
    m_lgb.fit(X_tr_imp, y_tr)
    pred_lgb = m_lgb.predict(X_te_imp)
    results_nested["LightGBM"]["r2"].append(r2_score(y_te, pred_lgb))
    results_nested["LightGBM"]["mae"].append(mean_absolute_error(y_te, pred_lgb))
    results_nested["LightGBM"]["best_params"].append(bp)

    # ── RF ──
    study_rf = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=fold_idx))
    study_rf.optimize(make_rf_objective(X_tr, y_tr, inner_cv),
                      n_trials=N_TRIALS, show_progress_bar=False)
    bp_rf = study_rf.best_params
    scl = StandardScaler(); imp2 = SimpleImputer(strategy="median")
    X_tr_p = scl.fit_transform(imp2.fit_transform(X_tr))
    X_te_p = scl.transform(imp2.transform(X_te))
    m_rf = RandomForestRegressor(**bp_rf, random_state=42, n_jobs=-1)
    m_rf.fit(X_tr_p, y_tr)
    pred_rf = m_rf.predict(X_te_p)
    results_nested["RF"]["r2"].append(r2_score(y_te, pred_rf))
    results_nested["RF"]["mae"].append(mean_absolute_error(y_te, pred_rf))
    results_nested["RF"]["best_params"].append(bp_rf)

    # ── Ridge ──
    study_ridge = optuna.create_study(direction="maximize",
                                       sampler=optuna.samplers.TPESampler(seed=fold_idx))
    study_ridge.optimize(make_ridge_objective(X_tr, y_tr, inner_cv),
                         n_trials=40, show_progress_bar=False)
    alpha_best = study_ridge.best_params["alpha"]
    pipe_r = Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("scl", StandardScaler()),
                       ("model", Ridge(alpha=alpha_best))])
    pipe_r.fit(X_tr, y_tr)
    pred_r = pipe_r.predict(X_te)
    results_nested["Ridge"]["r2"].append(r2_score(y_te, pred_r))
    results_nested["Ridge"]["mae"].append(mean_absolute_error(y_te, pred_r))
    results_nested["Ridge"]["best_params"].append({"alpha": alpha_best})

    print(f"  Fold {fold_idx+1:2d}/10  LGB R²={results_nested['LightGBM']['r2'][-1]:.3f}  "
          f"RF R²={results_nested['RF']['r2'][-1]:.3f}  "
          f"Ridge R²={results_nested['Ridge']['r2'][-1]:.3f}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n=== Nested CV Results (Block E, DTT SRT) ===")
summary_rows = []
for mname, res in results_nested.items():
    r2_arr  = np.array(res["r2"])
    mae_arr = np.array(res["mae"])
    row = {"model": mname,
           "r2_mean": r2_arr.mean(), "r2_std": r2_arr.std(),
           "mae_mean": mae_arr.mean(), "mae_std": mae_arr.std()}
    summary_rows.append(row)
    print(f"  {mname:12s}: R²={r2_arr.mean():.4f}±{r2_arr.std():.4f}  "
          f"MAE={mae_arr.mean():.4f}±{mae_arr.std():.4f}")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("results_nested_cv.csv", index=False)

# ── Wilcoxon signed-rank tests ─────────────────────────────────────────────────
print("\n=== Wilcoxon Signed-Rank Tests (fold-level R², paired) ===")
models = list(results_nested.keys())
test_rows = []
for i in range(len(models)):
    for j in range(i+1, len(models)):
        a = np.array(results_nested[models[i]]["r2"])
        b = np.array(results_nested[models[j]]["r2"])
        stat, pval = wilcoxon(a, b, alternative="two-sided")
        direction = ">" if a.mean() > b.mean() else "<"
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
        print(f"  {models[i]} vs {models[j]}: W={stat:.1f}, p={pval:.4f} {sig} "
              f"({models[i]} {direction} {models[j]})")
        test_rows.append({"model_a": models[i], "model_b": models[j],
                          "W": stat, "p": pval, "sig": sig})

pd.DataFrame(test_rows).to_csv("results_wilcoxon.csv", index=False)

# Also test Block E vs Block A for best model (LightGBM)
print("\n=== Block A vs E: LightGBM (default params, 10-fold) ===")
from sklearn.base import clone as sk_clone
def cv_r2_folds(X, y, model_factory, cv):
    fold_r2 = []
    for tr, te in cv.split(X):
        m = model_factory()
        m.fit(X[tr], y[tr])
        fold_r2.append(r2_score(y[te], m.predict(X[te])))
    return np.array(fold_r2)

imp_full = SimpleImputer(strategy="median")
X_A_imp = imp_full.fit_transform(df[BLOCK_A].values.reshape(-1,1))
X_E_imp = imp_full.fit_transform(df[BLOCK_E].values)

def lgb_default():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                              num_leaves=31, random_state=42, verbose=-1))])

r2_A = cv_r2_folds(df[BLOCK_A].values, y,
                   lambda: Pipeline([("imp", SimpleImputer(strategy="median")),
                                     ("scl", StandardScaler()),
                                     ("m", Ridge(alpha=1.0))]),
                   outer_cv)
r2_E = cv_r2_folds(df[BLOCK_E].values, y, lgb_default, outer_cv)

stat_AE, p_AE = wilcoxon(r2_E, r2_A, alternative="greater")
print(f"  Block E (LGB) vs Block A (Ridge):  "
      f"R²_E={r2_E.mean():.4f}, R²_A={r2_A.mean():.4f}, "
      f"W={stat_AE:.1f}, p={p_AE:.4f}")

pd.DataFrame({"fold": range(10), "r2_A": r2_A, "r2_E": r2_E}).to_csv(
    "results_blockA_vs_E.csv", index=False)

print("\nDone. Saved: results_nested_cv.csv, results_wilcoxon.csv, results_blockA_vs_E.csv")
