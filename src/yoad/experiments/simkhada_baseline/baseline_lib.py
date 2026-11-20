"""Faithful re-implementation of the Simkhada et al. 2026 training protocol (no code copied; the reference repository has no
licence). Pure functions on an in-memory sample table from src/cohort/simkhada_dataset.build_dataset. Synthetic-tested in
tests/test_simkhada_baseline.py; the guarded real-data entry point is run_simkhada_baseline.py (researcher only).

Provenance: [A] article, [C] public code at commit c67358dbb0960b41c2b1fd775114d000bae13081, [D] our documented deviation.

Pipeline (one estimator; everything learned is fitted on training rows only)       [C]
  numeric:     standardise -> KNN imputation (k = 5) -> back-transform unless the model is linear (LR, SVC)
               -> variance threshold -> greedy correlation filter (|r| > t, keep the earlier column)
  categorical: most-frequent imputation (PTGENDER, PTHAND), placed LAST so SMOTENC sees their indices
  resampling:  none | SMOTENC | SMOTENC + Tomek links, applied only while fitting (all four algorithms)
  estimator:   LR (balanced, max_iter 5000) | RF (balanced_subsample) | SVC rbf (balanced, no probabilities) | XGBoost (hist)
Final deployment model (the reported result)                                      [A][C]
  test split   first fold of StratifiedGroupKFold(5, shuffle, seed 11): ~80/20, participants never shared
  search       RandomizedSearchCV, 1,000 draws, RepeatedStratifiedGroupKFold(5 x 5), scores {average precision, macro F1,
               balanced accuracy}, refit on balanced accuracy
  threshold    Youden's J on out-of-fold CALIBRATED predictions from grouped 5-fold CV inside the training set
  calibration  CalibratedClassifierCV(sigmoid) with grouped 5-fold splits on the training set
  test         metrics at that threshold; percentile bootstrap, 2,000 resamples of TEST SAMPLES (reference) and, as a labelled
               addition [D], of test PARTICIPANTS; calibration intercept and slope added [D]
Nested CV (historical performance annotation withheld)                              [A][C]
  outer RepeatedStratifiedGroupKFold(5 x 5) on the training set; inner StratifiedGroupKFold(5), 250 draws; per outer fold the
  same threshold and calibration steps, evaluated on the outer test fold
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, log_loss,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.metaestimators import available_if

from yoad.benchmark.metrics import select_threshold
from yoad.benchmark.pipeline import CorrelationFilter, FitGuard, ResampledClassifier

REFERENCE = {"seed": 11, "test_folds": 5, "final_n_iter": 1000, "nested_n_iter": 250, "outer_splits": 5, "outer_repeats": 5,
             "inner_splits": 5, "calibration": "sigmoid", "calibration_splits": 5, "threshold": "youden_oof_calibrated",
             "bootstrap_B": 2000, "refit": "balanced_accuracy", "knn_k": 5,
             "selected_configuration": {"algorithm": "xgboost", "sampling": "smotenc", "feature_set": "ALL"}}
ALGORITHMS = ("logistic", "random_forest", "svc", "xgboost")
SAMPLING = ("none", "smotenc", "smotenc_tomek")
LINEAR = {"logistic", "svc"}


# ------------------------------------------------------------------------------------------------ pipeline pieces
class ScaledKNNImputer(BaseEstimator, TransformerMixin):
    """KNN imputation on standardised values; returns standardised values for linear models, original units otherwise."""

    def __init__(self, n_neighbors: int = 5, return_scaled: bool = False):
        self.n_neighbors = n_neighbors
        self.return_scaled = return_scaled

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler().fit(X)
        self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors).fit(self.scaler_.transform(X))
        return self

    def transform(self, X):
        Z = self.imputer_.transform(self.scaler_.transform(X))
        return Z if self.return_scaled else self.scaler_.inverse_transform(Z)


class SmotencSampler(BaseEstimator):
    """SMOTE-NC on the training rows (categorical columns are the LAST n_cat columns), optionally followed by Tomek links.
    Plain SMOTE when there is no categorical column. Requires imbalanced-learn."""

    def __init__(self, n_cat: int = 0, k_neighbors: int = 5, random_state: int = 0, tomek: bool = False):
        self.n_cat = n_cat
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.tomek = tomek

    def fit_resample(self, X, y):
        from imblearn.over_sampling import SMOTE, SMOTENC
        from imblearn.under_sampling import TomekLinks
        X = np.asarray(X, float)
        if self.n_cat <= 0:
            sm = SMOTE(sampling_strategy="minority", random_state=self.random_state, k_neighbors=self.k_neighbors)
        else:
            if self.n_cat >= X.shape[1]:
                raise ValueError("n_cat must be smaller than the number of columns after preprocessing")
            sm = SMOTENC(categorical_features=list(range(X.shape[1] - self.n_cat, X.shape[1])), sampling_strategy="minority",
                         random_state=self.random_state, k_neighbors=self.k_neighbors)
        Xr, yr = sm.fit_resample(X, np.asarray(y))
        if self.tomek:
            Xr, yr = TomekLinks().fit_resample(Xr, yr)
        return Xr, yr


class _Resampled(ResampledClassifier):
    """ResampledClassifier that also exposes decision_function, but only when the wrapped estimator has one (SVC has no
    probabilities in the reference; XGBoost has no decision_function), so scorers and calibration pick the right method."""

    @available_if(lambda self: hasattr(self.estimator, "decision_function"))
    def decision_function(self, X):
        return self.estimator_.decision_function(X)

    @available_if(lambda self: hasattr(self.estimator, "predict_proba") and getattr(self.estimator, "probability", True))
    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def make_estimator(algorithm: str, seed: int):
    if algorithm == "logistic":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(class_weight="balanced", random_state=seed, max_iter=5000)
    if algorithm == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=seed, class_weight="balanced_subsample", n_jobs=1)
    if algorithm == "svc":
        from sklearn.svm import SVC
        return SVC(class_weight="balanced", random_state=seed, probability=False)
    if algorithm == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(objective="binary:logistic", eval_metric="auc", tree_method="hist", random_state=seed, n_jobs=1)
    raise ValueError(f"algorithm must be one of {ALGORITHMS}")


def make_pipeline(algorithm: str, sampling: str, num_cols: list, cat_cols: list, seed: int) -> Pipeline:
    if sampling not in SAMPLING:
        raise ValueError(f"sampling must be one of {SAMPLING}")
    num = Pipeline([("impute", ScaledKNNImputer(5, return_scaled=algorithm in LINEAR)), ("var", VarianceThreshold(0.0)),
                    ("corr", CorrelationFilter(0.95))])
    tr = [("num", num, list(num_cols))]
    if cat_cols:
        tr.append(("cat", SimpleImputer(strategy="most_frequent"), list(cat_cols)))
    sampler = None if sampling == "none" else SmotencSampler(len(cat_cols), 5, seed, tomek=sampling == "smotenc_tomek")
    return Pipeline([("fit_guard", FitGuard()), ("prep", ColumnTransformer(tr, remainder="drop")),
                     ("model", _Resampled(make_estimator(algorithm, seed), sampler))])


# Search distributions [C], with the reference step names mapped to ours:
#   preprocess__num__var -> prep__num__var, preprocess__num__corr -> prep__num__corr, smote__ -> model__sampler__,
#   clf__ -> model__estimator__
_PRUNE = {"prep__num__var__threshold": [0.0, 1e-6, 1e-5, 1e-4], "prep__num__corr__threshold": [0.90, 0.95, 0.98]}
_K = {"model__sampler__k_neighbors": randint(3, 8)}
_E = "model__estimator__"
_SPACE = {
    "logistic": {f"{_E}C": loguniform(1e-3, 1e2), f"{_E}solver": ["lbfgs"]},
    "svc": {f"{_E}kernel": ["rbf"], f"{_E}C": loguniform(1e-3, 1e3), f"{_E}gamma": loguniform(1e-4, 1e1)},
    "random_forest": {f"{_E}n_estimators": randint(100, 1001), f"{_E}max_depth": [None] + list(range(3, 31)),
                      f"{_E}min_samples_split": randint(2, 21), f"{_E}min_samples_leaf": randint(1, 11),
                      f"{_E}max_features": ["sqrt", "log2", None], f"{_E}bootstrap": [True, False]},
    "xgboost": {f"{_E}max_depth": randint(3, 8), f"{_E}min_child_weight": randint(1, 8),
                f"{_E}learning_rate": loguniform(1e-3, 3e-1), f"{_E}n_estimators": randint(100, 1001),
                f"{_E}subsample": uniform(0.6, 0.4), f"{_E}colsample_bytree": uniform(0.6, 0.4), f"{_E}gamma": uniform(0.0, 2.0),
                f"{_E}reg_lambda": loguniform(1e-2, 1e1), f"{_E}reg_alpha": loguniform(1e-4, 1e0)},
}
_CW_NONE = {"logistic": [None, "balanced"], "svc": [None, "balanced"], "random_forest": [None, "balanced", "balanced_subsample"]}


def search_space(algorithm: str, sampling: str) -> dict:
    sp = {**_PRUNE, **_SPACE[algorithm]}
    if sampling != "none":
        sp.update(_K)
        if algorithm in _CW_NONE:
            sp[f"{_E}class_weight"] = [None]          # with resampling the reference fixes class_weight to None
    elif algorithm in _CW_NONE:
        sp[f"{_E}class_weight"] = _CW_NONE[algorithm]
    return sp


# ------------------------------------------------------------------------------------------------ splits
def repeated_sgkf(y, groups, n_splits: int, n_repeats: int, seed: int) -> list:
    """Repeated StratifiedGroupKFold driven by one RandomState, as sklearn's repeated splitters do."""
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_repeats):
        out += list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rng).split(np.zeros(len(y)), y, groups))
    return out


def sgkf(y, groups, n_splits: int, seed: int) -> list:
    return list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))


def grouped_test_split(y, groups, seed: int = REFERENCE["seed"]):
    tr, te = sgkf(y, groups, REFERENCE["test_folds"], seed)[0]
    if set(np.asarray(groups)[tr]) & set(np.asarray(groups)[te]):
        raise AssertionError("participant in both training and test sets")
    return tr, te


# ------------------------------------------------------------------------------------------------ fitting steps
SCORING = {"pr_auc": "average_precision", "f1_macro": "f1_macro", "balanced_accuracy": "balanced_accuracy"}


def run_search(pipe, space, X, y, groups, n_iter, cv, seed, n_jobs=1):
    s = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring=SCORING, refit="balanced_accuracy", cv=cv,
                           random_state=seed, n_jobs=n_jobs, error_score=np.nan)
    s.fit(X, y)
    failed = int(np.isnan(s.cv_results_["mean_test_balanced_accuracy"]).sum())
    return s, failed


def grouped_calibrated(model, X, y, groups, seed):
    cal = CalibratedClassifierCV(estimator=clone(model), method="sigmoid", cv=sgkf(y, groups, REFERENCE["calibration_splits"], seed))
    return cal.fit(X, y)


def oof_threshold(model, X, y, groups, seed):
    """Youden threshold on out-of-fold calibrated predictions; every fit uses only the training rows passed in."""
    oof = np.full(len(y), np.nan)
    for itr, iva in sgkf(y, groups, REFERENCE["inner_splits"], seed):
        cal = grouped_calibrated(model, X.iloc[itr], y[itr], groups[itr], seed)
        oof[iva] = cal.predict_proba(X.iloc[iva])[:, 1]
    return select_threshold("youden", None, y, oof), oof


def fitted_rows(calibrated) -> set:
    """Row labels seen by every pipeline inside a CalibratedClassifierCV (for the leakage check)."""
    rows = set()
    for cc in calibrated.calibrated_classifiers_:
        rows |= set(cc.estimator.named_steps["fit_guard"].fitted_rows_)
    return rows


# ------------------------------------------------------------------------------------------------ evaluation
def _cal_int_slope(y, p):
    from yoad.experiments.today_audit.audit_core import cal_intercept, cal_slope
    return cal_intercept(y, p), cal_slope(y, p)


def point_metrics(y, p, t) -> dict:
    y = np.asarray(y).astype(int); p = np.asarray(p, float); d = (p >= t).astype(int)
    tp = int(((d == 1) & (y == 1)).sum()); tn = int(((d == 0) & (y == 0)).sum())
    fp = int(((d == 1) & (y == 0)).sum()); fn = int(((d == 0) & (y == 1)).sum())
    m = {"balanced_accuracy": balanced_accuracy_score(y, d), "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
         "specificity": tn / (tn + fp) if tn + fp else np.nan, "ppv": tp / (tp + fp) if tp + fp else np.nan,
         "npv": tn / (tn + fn) if tn + fn else np.nan, "f1_macro": f1_score(y, d, average="macro"),
         "mcc": matthews_corrcoef(y, d), "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
         "prevalence": float(y.mean()), "brier": brier_score_loss(y, p), "log_loss": log_loss(y, np.clip(p, 1e-15, 1 - 1e-15))}
    try:
        m["calibration_intercept"], m["calibration_slope"] = _cal_int_slope(y, p)       # [D] addition
    except Exception:
        m["calibration_intercept"] = m["calibration_slope"] = np.nan
    return {k: float(v) for k, v in m.items()}


def bootstrap(y, p, t, groups, unit: str, B: int, seed: int) -> pd.DataFrame:
    """Percentile 95% CIs. unit='sample' resamples test samples (reference [C]); unit='participant' resamples participants
    with all their samples [D]. Conditional on the fitted model; single-class resamples are skipped and counted."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int); p = np.asarray(p, float); g = np.asarray(groups)
    if unit == "participant":
        ids = np.unique(g); idx_of = {k: np.where(g == k)[0] for k in ids}
    reps, skipped = [], 0
    for _ in range(B):
        idx = rng.integers(0, len(y), len(y)) if unit == "sample" else np.concatenate([idx_of[k] for k in rng.choice(ids, len(ids))])
        if len(np.unique(y[idx])) < 2:
            skipped += 1; continue
        reps.append(point_metrics(y[idx], p[idx], t))
    r = pd.DataFrame(reps)
    out = pd.DataFrame({"metric": r.columns, "ci_low": r.quantile(0.025).to_numpy(), "ci_high": r.quantile(0.975).to_numpy(),
                        "bootstrap_mean": r.mean().to_numpy()})
    out.insert(0, "unit", unit); out["B_used"] = len(reps); out["B_skipped_single_class"] = skipped
    return out


# ------------------------------------------------------------------------------------------------ protocols
def final_model(samples: pd.DataFrame, num_cols, cat_cols, algorithm="xgboost", sampling="smotenc", seed=REFERENCE["seed"],
                n_iter=REFERENCE["final_n_iter"], outer=(REFERENCE["outer_splits"], REFERENCE["outer_repeats"]),
                B=REFERENCE["bootstrap_B"], n_jobs=1) -> dict:
    """The reported final deployment model, evaluated once on the untouched test participants."""
    t0 = time.time()
    X = samples[list(num_cols) + list(cat_cols)]
    y = samples["y"].to_numpy().astype(int); g = samples["participant_id"].to_numpy()
    tr, te = grouped_test_split(y, g, seed)
    Xtr, ytr, gtr = X.iloc[tr], y[tr], g[tr]
    pipe = make_pipeline(algorithm, sampling, num_cols, cat_cols, seed)
    search, failed = run_search(pipe, search_space(algorithm, sampling), Xtr, ytr, gtr, n_iter,
                                repeated_sgkf(ytr, gtr, outer[0], outer[1], seed), seed, n_jobs)
    best = search.best_estimator_
    thr, _ = oof_threshold(best, Xtr, ytr, gtr, seed)
    cal = grouped_calibrated(best, Xtr, ytr, gtr, seed)
    seen = fitted_rows(cal) | set(best.named_steps["fit_guard"].fitted_rows_)
    if seen & set(X.index[te]):
        raise AssertionError("a test row was used for fitting")
    p = cal.predict_proba(X.iloc[te])[:, 1]
    pm = point_metrics(y[te], p, thr)
    boots = pd.concat([bootstrap(y[te], p, thr, g[te], "sample", B, seed), bootstrap(y[te], p, thr, g[te], "participant", B, seed)])
    cv = search.cv_results_; i = search.best_index_
    return {"algorithm": algorithm, "sampling": sampling, "threshold": float(thr), "best_params": {k: (v if np.isscalar(v) or v is None else str(v)) for k, v in search.best_params_.items()},
            "search_cv_best": {k: float(cv[f"mean_test_{k}"][i]) for k in SCORING} | {"balanced_accuracy_sd": float(cv["std_test_balanced_accuracy"][i])},
            "search_draws_failed": failed, "n_train_samples": int(len(tr)), "n_test_samples": int(len(te)),
            "n_train_participants": int(len(set(gtr))), "n_test_participants": int(len(set(g[te]))),
            "train_events": int(ytr.sum()), "test_events": int(y[te].sum()), "test_metrics": pm, "test_bootstrap": boots,
            "seconds": round(time.time() - t0, 1),
            "_model": cal, "_test_positions": te}            # underscore keys: local saving only, never written to aggregates


def nested_cv(samples: pd.DataFrame, num_cols, cat_cols, algorithm="xgboost", sampling="smotenc", seed=REFERENCE["seed"],
              n_iter=REFERENCE["nested_n_iter"], outer=(REFERENCE["outer_splits"], REFERENCE["outer_repeats"]),
              inner=REFERENCE["inner_splits"], n_jobs=1) -> pd.DataFrame:
    """Per-outer-fold metrics of the nested CV on the training set only (the test participants are never touched)."""
    X = samples[list(num_cols) + list(cat_cols)]
    y = samples["y"].to_numpy().astype(int); g = samples["participant_id"].to_numpy()
    tr, _ = grouped_test_split(y, g, seed)
    X, y, g = X.iloc[tr], y[tr], g[tr]
    rows = []
    for k, (otr, ote) in enumerate(repeated_sgkf(y, g, outer[0], outer[1], seed)):
        try:
            pipe = make_pipeline(algorithm, sampling, num_cols, cat_cols, seed)
            s, failed = run_search(pipe, search_space(algorithm, sampling), X.iloc[otr], y[otr], g[otr], n_iter,
                                   sgkf(y[otr], g[otr], inner, seed), seed, n_jobs)
            thr, _ = oof_threshold(s.best_estimator_, X.iloc[otr], y[otr], g[otr], seed)
            cal = grouped_calibrated(s.best_estimator_, X.iloc[otr], y[otr], g[otr], seed)
            if fitted_rows(cal) & set(X.index[ote]):
                raise AssertionError("an outer test row was used for fitting")
            rows.append({"outer_fold": k, "threshold": thr, "search_draws_failed": failed,
                         **point_metrics(y[ote], cal.predict_proba(X.iloc[ote])[:, 1], thr)})
        except Exception as e:                                          # recorded, never dropped silently
            rows.append({"outer_fold": k, "error": f"{type(e).__name__}: {str(e)[:200]}"})
    return pd.DataFrame(rows)
