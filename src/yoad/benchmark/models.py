"""Model registry: stable identifiers -> estimator builders, with an explicit implementation status.

status:
  implemented - builder present and covered by synthetic tests
  optional    - builder needs a package that may be absent; fails clearly if missing
  planned     - named for later stages; selecting it is a configuration error

New models are added with `register_model(...)`; the runner never needs editing for a new model.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC


class ModelNotAvailable(ValueError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    id: str
    status: str                                  # implemented | optional | planned
    tasks: tuple = ("binary_classification",)
    builder: Callable | None = None              # builder(params: dict, seed: int, class_weight: str | None) -> estimator
    allowed_params: frozenset = field(default_factory=frozenset)
    supports_class_weight: bool = False
    probabilistic: bool = True
    requires: tuple = ()                         # importable package names
    note: str = ""

    def available(self) -> bool:
        return self.builder is not None and all(importlib.util.find_spec(r) is not None for r in self.requires)


REGISTRY: dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec, replace: bool = False) -> None:
    if spec.status not in {"implemented", "optional", "planned"}:
        raise ValueError(f"invalid status {spec.status!r}")
    if spec.status != "planned" and spec.builder is None:
        raise ValueError(f"model {spec.id!r} with status {spec.status} needs a builder")
    if spec.id in REGISTRY and not replace:
        raise ValueError(f"model id {spec.id!r} already registered")
    REGISTRY[spec.id] = spec


def get_model(model_id: str, task: str) -> ModelSpec:
    if model_id not in REGISTRY:
        raise ModelNotAvailable(f"unknown model id {model_id!r}; registered: {sorted(REGISTRY)}")
    spec = REGISTRY[model_id]
    if spec.status == "planned":
        raise ModelNotAvailable(f"model {model_id!r} is planned but not implemented ({spec.note})")
    if task not in spec.tasks:
        raise ModelNotAvailable(f"model {model_id!r} does not support task {task!r}")
    if not spec.available():
        raise ModelNotAvailable(f"model {model_id!r} requires missing package(s) {spec.requires}")
    return spec


def build_estimator(model_id: str, task: str, params: dict, seed: int, class_weight: str | None = None):
    spec = get_model(model_id, task)
    bad = set(params) - spec.allowed_params
    if bad:
        raise ValueError(f"parameters {sorted(bad)} not allowed for model {model_id!r}; allowed {sorted(spec.allowed_params)}")
    if class_weight and not spec.supports_class_weight:
        raise ValueError(f"model {model_id!r} does not support class weighting")
    return spec.builder(dict(params), seed, class_weight)


# ------------------------------------------------------------------------------------------------ score-only wrapper
class ScoreClassifier(BaseEstimator, ClassifierMixin):
    """Wraps a margin classifier (decision_function only). predict_proba returns sigmoid(decision score): a monotone SCORE in
    (0, 1), NOT a calibrated probability. Models using it are registered probabilistic=False, and the configuration validator
    then requires post-hoc calibration on the participant-separate validation partition. (Built-in Platt scaling in SVC /
    CalibratedClassifierCV would use internal splits that ignore participant grouping.)"""

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def decision_function(self, X):
        return self.estimator_.decision_function(X)

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-np.clip(self.decision_function(X), -50, 50)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.estimator_.predict(X)


def _tuple(p, key):
    if key in p and isinstance(p[key], list):
        p[key] = tuple(p[key])
    return p


# ------------------------------------------------------------------------------------------------ registry entries
# Statistical baselines
register_model(ModelSpec(
    "dummy_prior", "implemented",
    builder=lambda p, s, cw: DummyClassifier(strategy="prior", random_state=s),
    note="predicts the training prevalence; a floor for every comparison"))
register_model(ModelSpec(
    "logistic_l2", "implemented",
    builder=lambda p, s, cw: LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000, random_state=s,
                                                class_weight=cw, **p),
    allowed_params=frozenset({"C"}), supports_class_weight=True,
    note="L2-penalised logistic regression (scikit-learn)"))
register_model(ModelSpec(
    "logistic_l1", "implemented",
    builder=lambda p, s, cw: LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, random_state=s,
                                                class_weight=cw, **p),
    allowed_params=frozenset({"C"}), supports_class_weight=True, note="L1 (lasso) logistic regression"))
register_model(ModelSpec(
    "logistic_elasticnet", "implemented",
    builder=lambda p, s, cw: LogisticRegression(penalty="elasticnet", solver="saga", max_iter=10000, random_state=s,
                                                class_weight=cw, **{"l1_ratio": 0.5, **p}),
    allowed_params=frozenset({"C", "l1_ratio"}), supports_class_weight=True, note="elastic-net logistic regression (saga)"))
register_model(ModelSpec(
    "lda", "implemented",
    builder=lambda p, s, cw: LinearDiscriminantAnalysis(solver="lsqr", **{"shrinkage": "auto", **p}),
    allowed_params=frozenset({"shrinkage"}), note="shrinkage LDA (Ledoit-Wolf by default)"))
register_model(ModelSpec(
    "qda", "implemented",
    builder=lambda p, s, cw: QuadraticDiscriminantAnalysis(**{"reg_param": 0.5, **p}),
    allowed_params=frozenset({"reg_param"}),
    note="regularised QDA; only defensible with few features relative to events per class"))
register_model(ModelSpec(
    "gaussian_nb", "implemented",
    builder=lambda p, s, cw: GaussianNB(**p), allowed_params=frozenset({"var_smoothing"}),
    note="Gaussian naive Bayes on preprocessed (one-hot + scaled) inputs"))
# Margin-based (score-only; must be calibrated on the validation partition)
register_model(ModelSpec(
    "linear_svm", "implemented", probabilistic=False,
    builder=lambda p, s, cw: ScoreClassifier(LinearSVC(dual="auto", max_iter=20000, random_state=s, class_weight=cw, **p)),
    allowed_params=frozenset({"C"}), supports_class_weight=True, note="linear SVM; scores only"))
register_model(ModelSpec(
    "svm_rbf", "implemented", probabilistic=False,
    builder=lambda p, s, cw: ScoreClassifier(SVC(kernel="rbf", random_state=s, class_weight=cw, **p)),
    allowed_params=frozenset({"C", "gamma"}), supports_class_weight=True, note="RBF SVM; scores only"))
# Tree ensembles and boosting
_RF = frozenset({"n_estimators", "max_depth", "min_samples_leaf", "max_features"})
register_model(ModelSpec(
    "random_forest", "implemented",
    builder=lambda p, s, cw: RandomForestClassifier(**{"n_estimators": 300, **p}, n_jobs=1, random_state=s, class_weight=cw),
    allowed_params=_RF, supports_class_weight=True))
register_model(ModelSpec(
    "extra_trees", "implemented",
    builder=lambda p, s, cw: ExtraTreesClassifier(**{"n_estimators": 300, **p}, n_jobs=1, random_state=s, class_weight=cw),
    allowed_params=_RF, supports_class_weight=True))
register_model(ModelSpec(
    "hist_gb", "implemented",
    builder=lambda p, s, cw: HistGradientBoostingClassifier(random_state=s, class_weight=cw, **p),
    allowed_params=frozenset({"learning_rate", "max_depth", "max_iter", "min_samples_leaf", "l2_regularization"}),
    supports_class_weight=True))
register_model(ModelSpec(
    "xgboost", "optional", requires=("xgboost",),
    builder=lambda p, s, cw: __import__("xgboost").XGBClassifier(
        **{"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, **p}, tree_method="hist", n_jobs=1, random_state=s,
        eval_metric="logloss"),
    allowed_params=frozenset({"n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "reg_lambda",
                              "min_child_weight"}),
    note="reference-paper family; class imbalance via sampling, not class_weight"))
register_model(ModelSpec(
    "lightgbm", "optional", requires=("lightgbm",),
    builder=lambda p, s, cw: __import__("lightgbm").LGBMClassifier(**p, random_state=s, n_jobs=1, class_weight=cw, verbose=-1),
    allowed_params=frozenset({"n_estimators", "num_leaves", "learning_rate", "min_child_samples", "reg_lambda"}),
    supports_class_weight=True, note="package not installed in .venv-data; selecting it fails clearly"))
register_model(ModelSpec(
    "catboost", "optional", requires=("catboost",),
    builder=lambda p, s, cw: __import__("catboost").CatBoostClassifier(**p, random_seed=s, verbose=0, thread_count=1),
    allowed_params=frozenset({"iterations", "depth", "learning_rate", "l2_leaf_reg"}),
    note="package not installed in .venv-data; selecting it fails clearly"))
# Neural
register_model(ModelSpec(
    "mlp", "implemented",
    builder=lambda p, s, cw: MLPClassifier(**{"hidden_layer_sizes": (32,), "alpha": 1e-3, **_tuple(dict(p), "hidden_layer_sizes")},
                                           max_iter=1000, early_stopping=False, random_state=s),
    allowed_params=frozenset({"hidden_layer_sizes", "alpha", "learning_rate_init"}),
    note="small MLP; early stopping disabled because its internal split ignores participant grouping"))

for _id, _req, _note in [
    ("tabnet", ("pytorch_tabnet",), "not installed; needs a sample size that the planned cohorts are unlikely to reach"),
    ("ft_transformer", ("torch",), "not installed; same caveat"),
    ("learned_fusion_net", (), "later stage; simple late fusion is available via fusion.type: late (not a model id)"),
    ("cox_ph", (), "survival task (not implemented)"),
    ("gru_landmark", ("torch",), "longitudinal task (not implemented); use src/cohort landmark builders first"),
]:
    register_model(ModelSpec(_id, "planned", requires=_req, note=_note))
