"""Candidate estimators of the multimodal benchmark (PROTOCOL.md §6). Every learned step (imputation, scaling, interaction
expansion, stacking, resampling, the estimator) is fitted inside `fit`, on the rows passed to it only.

* lr_l2_ref / xgb_ref with median imputation are built by the FROZEN factory experiments/visit_anchor_audit/design_lib.
  make_pipeline_factory, i.e. exactly the 2 x 2 pipelines (L2 C = 1; XGBoost 200 / 2 / 0.05; median imputation + scaling).
* Other registry models reuse src/benchmark/pipeline.build_pipeline (FitGuard first; the scikit-learn / XGBoost builders of
  src/benchmark/models.py).
* lr_interact: median + scaling -> all pairwise products -> scaling -> ridge logistic regression.
* late_fusion: one L2 logistic model per modality group; their out-of-fold (inner stratified K-fold inside fit) log-odds are
  stacked by a logistic regression; base models are then refitted on all fitting rows.
* cox_l2: ridge Cox proportional-hazards model (Breslow ties and baseline hazard), fitted on right-censored follow-up of ALL
  fitting participants (REF-indeterminate included); predicted 24-month risk = 1 - S0(730.5 d)^exp(x'beta).
* xgb_simkhada: experiments/simkhada_baseline/baseline_lib (the re-implemented reference pipeline), driven by the engine.
Pure; no file access. Synthetic-tested in multimodal_benchmark/tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from yoad.experiments.visit_anchor_audit import design_lib as frozen
from yoad.benchmark.pipeline import FitGuard

SB_KEY = {"lr_l2_ref": "logistic_l2", "xgb_ref": "xgboost", "lr_enet": "logistic_elasticnet", "svm_rbf": "svm_rbf",
          "random_forest": "random_forest", "extra_trees": "extra_trees", "hist_gb": "hist_gb", "xgb_tuned": "xgboost",
          "mlp": "mlp"}
IMPUTATION_KEY = {"median": "median", "iterative": "iterative", "median_indicator": "median_with_indicator",
                  "complete_case": "median"}       # complete case restricts rows; the (inactive) median step is kept


def _logit(p):
    q = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def sb_pipeline(model_id: str, params: dict, features: list, seed: int, imputation: str = "median"):
    """src/benchmark pipeline; for (lr_l2_ref | xgb_ref, median) this is the frozen design_lib factory itself."""
    if model_id in ("lr_l2_ref", "xgb_ref") and imputation in ("median", "complete_case"):
        return frozen.make_pipeline_factory(SB_KEY[model_id], params, features)("benchmark", seed)
    from yoad.benchmark.config import load_config
    from yoad.benchmark.data import FeatureInfo
    from yoad.benchmark.pipeline import build_pipeline
    feats = [FeatureInfo(f, "clinical", "numeric") for f in features]
    raw = {"experiment_id": "multimodal_benchmark", "model": {"id": SB_KEY[model_id], "params": dict(params)},
           "preprocessing": {"numeric_imputation": IMPUTATION_KEY[imputation], "scaling": "standard"}, "metrics": ["roc_auc"]}
    if model_id == "svm_rbf":       # score-only: calibrated by the engine (Platt on inner out-of-fold scores)
        raw.update(calibration={"method": "sigmoid"}, validation={"validation_fraction": 0.2})
        raw["model"]["search_metric"] = "roc_auc"
    return build_pipeline(load_config(raw), feats, dict(params), seed)


def interaction_pipeline(features: list, C: float, seed: int) -> Pipeline:
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    return Pipeline([("fit_guard", FitGuard()), ("columns", ColumnTransformer([("num", num, list(features))], remainder="drop")),
                     ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
                     ("scale2", StandardScaler()),
                     ("model", LogisticRegression(C=C, max_iter=5000, random_state=seed))])


class LateFusionClassifier(BaseEstimator, ClassifierMixin):
    """Decision-level fusion. groups: {name: [feature, ...]} (columns of the DataFrame passed to fit/predict)."""

    def __init__(self, groups=None, C=1.0, stack_C=1.0, inner_splits=3, seed=0):
        self.groups = groups
        self.C = C
        self.stack_C = stack_C
        self.inner_splits = inner_splits
        self.seed = seed

    def _base(self):
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                         ("lr", LogisticRegression(C=self.C, max_iter=5000, random_state=self.seed))])

    def fit(self, X: pd.DataFrame, y):
        if not hasattr(X, "index"):
            raise TypeError("LateFusionClassifier needs a DataFrame")
        self.fitted_rows_ = frozenset(X.index)
        y = np.asarray(y).astype(int)
        names = list(self.groups)
        oof = np.zeros((len(y), len(names)))
        skf = StratifiedKFold(n_splits=self.inner_splits, shuffle=True, random_state=self.seed)
        for tr, va in skf.split(np.zeros(len(y)), y):
            for j, g in enumerate(names):
                m = self._base().fit(X.iloc[tr][self.groups[g]], y[tr])
                oof[va, j] = m.predict_proba(X.iloc[va][self.groups[g]])[:, 1]
        self.stacker_ = LogisticRegression(C=self.stack_C, max_iter=5000, random_state=self.seed).fit(_logit(oof), y)
        self.bases_ = {g: self._base().fit(X[self.groups[g]], y) for g in names}
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        Z = np.column_stack([_logit(self.bases_[g].predict_proba(X[self.groups[g]])[:, 1]) for g in self.groups])
        return self.stacker_.predict_proba(Z)


class CoxRidge:
    """Cox PH with an L2 penalty 0.5 * penalty * ||beta||^2 on standardised covariates; Breslow handling of ties."""

    def __init__(self, penalty: float = 1.0):
        self.penalty = penalty

    @staticmethod
    def _risk_sets(time):
        order = np.argsort(-time, kind="mergesort")              # descending time: cumulative sums give risk-set sums
        t_sorted = time[order]
        # last position (in descending order) of each tie group -> risk set {j: time_j >= t}
        last = np.r_[np.flatnonzero(np.diff(t_sorted) != 0), len(t_sorted) - 1]
        grp_end = np.repeat(last, np.diff(np.r_[-1, last]))
        return order, grp_end

    def fit(self, X, time, event):
        X = np.asarray(X, float); time = np.asarray(time, float); event = np.asarray(event, int)
        if event.sum() == 0:
            raise ValueError("Cox model needs at least one event")
        order, grp_end = self._risk_sets(time)
        Xs, es = X[order], event[order]

        def f(beta):
            eta = Xs @ beta
            m = eta.max()
            w = np.exp(eta - m)
            cs = np.cumsum(w)[grp_end]                               # risk-set sums (scaled by exp(-m))
            cx = np.cumsum(w[:, None] * Xs, axis=0)[grp_end]
            ll = np.sum(es * (eta - m - np.log(cs)))
            grad = np.sum(es[:, None] * (Xs - cx / cs[:, None]), axis=0)
            pen = 0.5 * self.penalty * beta @ beta
            return -(ll - pen), -(grad - self.penalty * beta)
        res = minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": 1000})
        self.coef_ = res.x
        self.converged_ = bool(res.success)
        w = np.exp(Xs @ self.coef_)
        cs = np.cumsum(w)[grp_end]
        ev_t = time[order]
        # Breslow: H0(t) = sum over distinct event times <= t of d(t) / risk-set sum(t)
        uniq = np.unique(ev_t[es == 1])
        d = np.array([es[ev_t == u].sum() for u in uniq])
        rs = np.array([cs[np.flatnonzero(ev_t == u)[0]] for u in uniq])
        self.event_times_, self.cum_hazard_ = uniq, np.cumsum(d / rs)
        return self

    def cumulative_hazard(self, t: float) -> float:
        k = np.searchsorted(self.event_times_, t, side="right")
        return float(self.cum_hazard_[k - 1]) if k > 0 else 0.0

    def risk(self, X, horizon: float) -> np.ndarray:
        return 1.0 - np.exp(-self.cumulative_hazard(horizon) * np.exp(np.asarray(X, float) @ self.coef_))


class CoxPipeline:
    """median imputation + scaling + CoxRidge, fitted on the participants passed to fit; predict_proba gives 24-month risk."""

    def __init__(self, features: list, penalty: float, horizon: float):
        self.features, self.penalty, self.horizon = list(features), penalty, horizon

    def fit(self, X: pd.DataFrame, time, event):
        self.fitted_rows_ = frozenset(X.index)
        self.prep_ = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]).fit(X[self.features])
        self.cox_ = CoxRidge(self.penalty).fit(self.prep_.transform(X[self.features]), time, event)
        return self

    def predict_proba(self, X: pd.DataFrame):
        p = np.clip(self.cox_.risk(self.prep_.transform(X[self.features]), self.horizon), 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])


def fitted_rows(est) -> frozenset:
    """Row labels an estimator was fitted on (FitGuard, LateFusion, Cox), for the run-time leakage check."""
    if hasattr(est, "fitted_rows_"):
        return est.fitted_rows_
    if hasattr(est, "named_steps") and "fit_guard" in est.named_steps:
        return est.named_steps["fit_guard"].fitted_rows_
    raise TypeError(f"cannot verify fitted rows of {type(est).__name__}")


def build(model_id: str, params: dict, features: list, seed: int, imputation: str = "median", groups: dict | None = None):
    """A new unfitted binary estimator for one candidate configuration (cox_l2 and xgb_simkhada are driven by the engine)."""
    if model_id == "lr_interact":
        return interaction_pipeline(features, params["C"], seed)
    if model_id == "late_fusion":
        return LateFusionClassifier(groups=groups, C=params["C"], stack_C=params["stack_C"], seed=seed)
    if model_id in SB_KEY:
        return sb_pipeline(model_id, params, features, seed, imputation)
    raise ValueError(f"model {model_id!r} is not built by models.build")
