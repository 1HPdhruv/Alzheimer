"""Pipeline builder.

Every learned step (imputation, encoding, scaling, filtering, feature selection, resampling, the estimator) is a step of
one scikit-learn Pipeline, so `pipeline.fit(X_train, y_train)` is the only place anything is learned. Resampling is done
inside `ResampledClassifier.fit`, so it can only ever see the rows passed to fit; prediction never resamples.
Unimplemented options raise NotImplementedError instead of being replaced by a different method.
"""
from __future__ import annotations

from functools import partial
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (activates IterativeImputer)
from sklearn.feature_selection import SelectFromModel, SelectKBest, VarianceThreshold, f_classif, mutual_info_classif
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from yoad.benchmark.data import FeatureInfo
from yoad.benchmark.models import build_estimator


# ------------------------------------------------------------------------------------------------ custom steps
class FitGuard(BaseEstimator, TransformerMixin):
    """Always the first pipeline step: remembers (in memory only) which row labels the pipeline was fitted on, so the
    runner can verify at run time that no evaluation/validation row was used for fitting. Passes data through unchanged."""

    def fit(self, X, y=None):
        if not hasattr(X, "index"):
            raise TypeError("FitGuard needs a pandas DataFrame so fitted rows can be verified")
        self.fitted_rows_ = frozenset(X.index)
        return self

    def transform(self, X):
        return X


def assert_fit_rows(pipe: Pipeline, allowed, forbidden=()) -> None:
    """Raise LeakageError unless the pipeline was fitted on exactly `allowed` and on none of `forbidden` (row labels)."""
    from yoad.evaluation.safeguards import LeakageError
    rows = pipe.named_steps["fit_guard"].fitted_rows_
    if rows & frozenset(forbidden):
        raise LeakageError(f"pipeline was fitted on {len(rows & frozenset(forbidden))} held-out row(s)")
    if rows != frozenset(allowed):
        raise LeakageError("pipeline was not fitted on exactly the intended fitting rows")


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drops the later column of any pair with |Pearson r| > threshold, judged on the training data only."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        keep = []
        if X.shape[1]:
            c = np.abs(np.nan_to_num(np.corrcoef(X, rowvar=False), nan=0.0)) if X.shape[1] > 1 else np.zeros((1, 1))
            for j in range(X.shape[1]):
                if all(c[j, k] <= self.threshold for k in keep):
                    keep.append(j)
        self.keep_ = np.array(keep, dtype=int)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X, float)[:, self.keep_]


class KBest(SelectKBest):
    """SelectKBest that fails clearly when k exceeds the number of available features (no silent 'all')."""

    def fit(self, X, y):
        if self.k != "all" and self.k > np.asarray(X).shape[1]:
            raise ValueError(f"feature_selection.k={self.k} exceeds the {np.asarray(X).shape[1]} features available after preprocessing")
        return super().fit(X, y)


class RandomOverSampler:
    """Duplicates minority-class rows (with replacement) until classes are balanced. Seeded; numpy only."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit_resample(self, X, y):
        y = np.asarray(y); rng = np.random.default_rng(self.seed)
        cls, cnt = np.unique(y, return_counts=True)
        idx = [np.where(y == c)[0] for c in cls]
        out = [np.concatenate([i, rng.choice(i, size=cnt.max() - len(i), replace=True)]) for i in idx]
        sel = np.concatenate(out)
        return np.asarray(X)[sel], y[sel]


class RandomUnderSampler:
    """Drops majority-class rows (without replacement) until classes are balanced. Seeded; numpy only."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit_resample(self, X, y):
        y = np.asarray(y); rng = np.random.default_rng(self.seed)
        cls, cnt = np.unique(y, return_counts=True)
        sel = np.concatenate([rng.choice(np.where(y == c)[0], size=cnt.min(), replace=False) for c in cls])
        return np.asarray(X)[np.sort(sel)], y[np.sort(sel)]


SAMPLERS: dict[str, Callable | None] = {
    "random_oversample": RandomOverSampler,
    "random_undersample": RandomUnderSampler,
    "smotenc": None,            # planned: requires imbalanced-learn (not installed) and categorical-index handling
}


class ResampledClassifier(BaseEstimator, ClassifierMixin):
    """Final pipeline step: resamples the rows it is fitted on, then fits the estimator. Prediction is untouched."""

    def __init__(self, estimator=None, sampler=None):
        self.estimator = estimator
        self.sampler = sampler

    def fit(self, X, y):
        self.n_input_rows_ = len(y)
        Xr, yr = self.sampler.fit_resample(X, y) if self.sampler is not None else (X, np.asarray(y))
        self.n_resampled_rows_ = len(yr)
        self.resampled_class_counts_ = {int(k): int(v) for k, v in zip(*np.unique(yr, return_counts=True))}
        self.estimator_ = clone(self.estimator).fit(Xr, yr)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.estimator_.predict(X)


# ------------------------------------------------------------------------------------------------ builder
NUMERIC_IMPUTATION = {"mean": dict(strategy="mean"), "median": dict(strategy="median"),
                      "median_with_indicator": dict(strategy="median", add_indicator=True)}
IMPUTER_FACTORIES = {"knn": lambda seed: KNNImputer(n_neighbors=5),
                     "iterative": lambda seed: IterativeImputer(max_iter=10, random_state=seed, sample_posterior=False)}
PLANNED_IMPUTATION = {"complete_case"}   # complete-case changes the population; must be decided at cohort level, not here
STEP_HOOKS: dict[str, Callable[[], object]] = {}     # optional extra first steps (e.g. audit spies in tests)


def build_pipeline(cfg: dict, features: list[FeatureInfo], params: dict, seed: int) -> Pipeline:
    pre, fs, samp, model = cfg["preprocessing"], cfg["feature_selection"], cfg["sampling"], cfg["model"]
    num = [f.name for f in features if f.kind == "numeric"]
    cat = [f.name for f in features if f.kind == "categorical"]
    if pre["numeric_imputation"] in PLANNED_IMPUTATION:
        raise NotImplementedError(f"numeric_imputation={pre['numeric_imputation']!r} is planned, not implemented")
    ni = pre["numeric_imputation"]
    num_steps = [("impute", IMPUTER_FACTORIES[ni](seed) if ni in IMPUTER_FACTORIES else SimpleImputer(**NUMERIC_IMPUTATION[ni]))]
    if pre["scaling"] == "standard":
        num_steps.append(("scale", StandardScaler()))
    transformers = []
    if num:
        transformers.append(("num", Pipeline(num_steps), num))
    if cat:
        transformers.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                              ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat))
    steps = [("fit_guard", FitGuard())] + [(f"hook_{h}", STEP_HOOKS[h]()) for h in pre.get("extra_steps", [])]
    steps.append(("columns", ColumnTransformer(transformers, remainder="drop")))
    if pre.get("low_variance_threshold") is not None:
        steps.append(("low_variance", VarianceThreshold(pre["low_variance_threshold"])))
    if pre.get("correlation_threshold") is not None:
        steps.append(("correlation", CorrelationFilter(pre["correlation_threshold"])))
    if fs["method"] == "anova":
        steps.append(("select", KBest(f_classif, k=fs["k"])))
    elif fs["method"] == "mutual_info":
        steps.append(("select", KBest(partial(mutual_info_classif, random_state=seed), k=fs["k"])))
    elif fs["method"] == "l1":
        steps.append(("select", SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", C=fs["l1_C"],
                                                                   random_state=seed, max_iter=5000))))
    elif fs["method"] == "pca":
        steps.append(("select", PCA(n_components=fs["n_components"], random_state=seed)))
    elif fs["method"] != "none":
        raise NotImplementedError(f"feature_selection.method={fs['method']!r} is planned, not implemented")
    cw = "balanced" if samp["method"] == "class_weight" else None
    est = build_estimator(model["id"], cfg["task"], params, seed, class_weight=cw)
    if samp["method"] in SAMPLERS:
        if SAMPLERS[samp["method"]] is None:
            raise NotImplementedError(f"sampling.method={samp['method']!r} is planned, not implemented")
        est = ResampledClassifier(est, SAMPLERS[samp["method"]](seed=seed))
    else:
        est = ResampledClassifier(est, None)
    steps.append(("model", est))
    return Pipeline(steps)
