"""Source-only development and enforced freezing of Model A (penalised logistic) / Model B (XGBoost).

Synthetic-tested only; never run on real participants.
* fit_source() takes SOURCE data only (cohort_role must be "source") - no target argument exists.
* imputation, scaling, grouped-CV tuning and optional calibration are fitted on the source.
* The returned FrozenModel carries a FreezeManifest; every prediction re-verifies the fitted-artifact checksum, the
  feature list/order, the domains (units/coding) and the manifest itself. Attribute assignment is blocked.
* Using target outcomes to recalibrate returns a labelled UpdatedModel (model updating, not frozen validation).
"""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from yoad.evaluation.safeguards import group_kfold, LeakageError
from yoad.models.freeze import FreezeManifest, IntegrityError, make_manifest

SEED = 11
PREPROCESSING_VERSION = "preproc-1.0"


class UnitCodingError(ValueError):
    pass


class TargetLeakageError(LeakageError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    name: str                       # "A_logistic" | "B_xgboost"
    features: tuple[str, ...]       # prespecified, ordered; never chosen from target performance
    grid: tuple[tuple[str, tuple], ...]
    horizon_months: int = 24
    outcome: str = "any_dementia"
    cv_folds: int = 5
    calibrate_on_source: bool = True
    seed: int = SEED
    domains: tuple[tuple[str, float, float, str], ...] = ()   # (feature, lo, hi, unit) plausible domain/coding

    def spec_hash(self) -> str:
        return hashlib.sha256(json.dumps([self.name, self.features, self.grid, self.horizon_months, self.outcome,
                                          self.cv_folds, self.calibrate_on_source, self.seed, self.domains], default=list).encode()).hexdigest()[:16]


def _estimator(name: str, params: dict, seed: int):
    if name == "A_logistic":
        return LogisticRegression(penalty="l2", solver="lbfgs", max_iter=2000, random_state=seed, **params)
    if name == "B_xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=seed, n_jobs=1, tree_method="hist", **params)
    raise ValueError(name)


def _artifact_sha(name, est, imp, sc, platt) -> str:
    h = hashlib.sha256()
    for a in (imp.statistics_, sc.mean_, sc.scale_):
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    if name == "A_logistic":
        for a in (est.coef_, est.intercept_):
            h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    else:
        h.update(bytes(est.get_booster().save_raw("json")))
    h.update(json.dumps(platt, default=float).encode())
    return h.hexdigest()


def validate_domains(X: pd.DataFrame, domains) -> None:
    bad = {}
    for f, lo, hi, unit in domains:
        v = X[f].dropna().astype(float)
        n_bad = int(((v < lo) | (v > hi)).sum())
        if n_bad:
            bad[f] = f"{n_bad} values outside [{lo}, {hi}] {unit}"
    if bad:
        raise UnitCodingError(bad)


class FrozenModel:
    """Immutable after construction; verified on every use."""

    def __init__(self, spec, params, imputer, scaler, estimator, platt, cv_table, source_n, manifest):
        for k, v in dict(_spec=spec, _params=params, _imp=imputer, _sc=scaler, _est=estimator, _platt=platt,
                         cv_table=cv_table, source_n=source_n, manifest=manifest, status="FROZEN_SOURCE_ONLY").items():
            object.__setattr__(self, k, v)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, k, v):
        raise AttributeError("FrozenModel is immutable; create a new version via fit_source(..., version=...)")

    def __delattr__(self, k):
        raise AttributeError("FrozenModel is immutable")

    @property
    def spec(self): return self._spec

    @property
    def params(self): return dict(self._params)

    def verify_integrity(self) -> None:
        self.manifest.verify()
        if _artifact_sha(self._spec.name, self._est, self._imp, self._sc, self._platt) != self.manifest.artifact_sha256:
            raise IntegrityError("fitted artifact differs from the frozen checksum")
        if tuple(self._spec.features) != tuple(self.manifest.features):
            raise IntegrityError("feature list/order differs from manifest")

    def _lp(self, X) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("scoring requires a named DataFrame; positional arrays can silently reorder predictors")
        self.verify_integrity()
        missing = [f for f in self._spec.features if f not in X.columns]
        if missing:
            raise KeyError(f"target lacks prespecified features: {missing}")
        validate_domains(X, self._spec.domains)
        Z = self._sc.transform(self._imp.transform(X[list(self._spec.features)].astype(float)))
        p = np.clip(self._est.predict_proba(Z)[:, 1], 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        lp = self._lp(X)
        if self._platt is not None:
            a, b = self._platt
            lp = a + b * lp
        return 1 / (1 + np.exp(-np.clip(lp, -30, 30)))

    def fingerprint(self) -> str:
        return self.manifest.manifest_sha256[:16]


def _fit_platt(y, lp):
    X = np.c_[np.ones_like(lp), lp]; beta = np.array([0.0, 1.0]); y = np.asarray(y, float)
    for _ in range(50):
        mu = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30))); W = mu * (1 - mu)
        step = np.linalg.solve(X.T @ (X * W[:, None]) + 1e-9 * np.eye(2), X.T @ (y - mu)); beta += step
        if np.abs(step).max() < 1e-8: break
    return float(beta[0]), float(beta[1])


def fit_source(spec: ModelSpec, X_src: pd.DataFrame, y_src: np.ndarray, participant_ids: pd.Series, *,
               version: str = "1.0.0", training_cohort_id: str = "source", cohort_role: str = "source",
               selection_result: dict | None = None, validation_plan: dict | None = None,
               freeze_date: str | None = None, parent_manifest_sha256: str | None = None) -> FrozenModel:
    """Tune by participant-grouped CV on the source, refit on all source rows, freeze. SOURCE DATA ONLY."""
    if cohort_role != "source":
        raise TargetLeakageError(f"model fitting/selection is allowed on the source cohort only (got role {cohort_role!r})")
    y = np.asarray(y_src)
    if np.isnan(y.astype(float)).any():
        raise ValueError("unlabelled outcomes must be removed (and counted) before fitting, not coded 0")
    X = X_src[list(spec.features)].astype(float).reset_index(drop=True)
    validate_domains(X, spec.domains)
    ids = pd.Series(np.asarray(participant_ids)).reset_index(drop=True)
    names = [g[0] for g in spec.grid]; combos = [dict(zip(names, c)) for c in itertools.product(*[g[1] for g in spec.grid])]
    folds = group_kfold(ids, pd.Series(y), k=spec.cv_folds, seed=spec.seed)
    rows = []
    for params in combos:
        aucs = []
        for tr, te in folds:
            imp = SimpleImputer(strategy="median").fit(X.iloc[tr]); sc = StandardScaler().fit(imp.transform(X.iloc[tr]))
            est = _estimator(spec.name, params, spec.seed).fit(sc.transform(imp.transform(X.iloc[tr])), y[tr])
            aucs.append(roc_auc_score(y[te], est.predict_proba(sc.transform(imp.transform(X.iloc[te])))[:, 1]))
        rows.append({**params, "cv_auc": float(np.mean(aucs))})
    table = pd.DataFrame(rows).sort_values("cv_auc", ascending=False, kind="mergesort").reset_index(drop=True)
    best = {k: table.loc[0, k] for k in names}; best = {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()}
    imp = SimpleImputer(strategy="median").fit(X); sc = StandardScaler().fit(imp.transform(X))
    est = _estimator(spec.name, best, spec.seed).fit(sc.transform(imp.transform(X)), y)
    platt = None
    if spec.calibrate_on_source:
        oof = np.full(len(y), np.nan)
        for tr, te in folds:
            i2 = SimpleImputer(strategy="median").fit(X.iloc[tr]); s2 = StandardScaler().fit(i2.transform(X.iloc[tr]))
            e2 = _estimator(spec.name, best, spec.seed).fit(s2.transform(i2.transform(X.iloc[tr])), y[tr])
            p = np.clip(e2.predict_proba(s2.transform(i2.transform(X.iloc[te])))[:, 1], 1e-6, 1 - 1e-6)
            oof[te] = np.log(p / (1 - p))
        platt = _fit_platt(y, oof)
    for a in (imp.statistics_, sc.mean_, sc.scale_):
        a.flags.writeable = False
    if spec.name == "A_logistic":
        est.coef_.flags.writeable = False; est.intercept_.flags.writeable = False
    manifest = make_manifest(
        model_id=spec.name, version=version, training_cohort_id=training_cohort_id, outcome=spec.outcome,
        horizon_months=spec.horizon_months, features=tuple(spec.features),
        preprocessing={"imputer": "median", "scaler": "standard", "preprocessing_version": PREPROCESSING_VERSION,
                       "domains": [list(d) for d in spec.domains], "calibrate_on_source": spec.calibrate_on_source,
                       "grid": [[g[0], list(g[1])] for g in spec.grid], "spec_hash": spec.spec_hash(), "chosen_params": best},
        artifact_sha256=_artifact_sha(spec.name, est, imp, sc, platt),
        selection_result=selection_result or {}, validation_plan=validation_plan or {},
        parent_manifest_sha256=parent_manifest_sha256, **({"freeze_date": freeze_date} if freeze_date else {}))
    return FrozenModel(spec, best, imp, sc, est, platt, table, len(y), manifest)


def evaluate_frozen_on_target(model: FrozenModel, X_t: pd.DataFrame, y_t, *, target_ids, source_ids=None,
                              expected_manifest_sha256: str | None = None, n_boot: int = 500) -> dict:
    """The ONLY sanctioned target evaluation. Re-verifies the freeze, refuses NaN labels, reports overlap status honestly."""
    from yoad.evaluation.frozen_validation import evaluate_frozen
    model.verify_integrity()
    if expected_manifest_sha256 is not None and model.manifest.manifest_sha256 != expected_manifest_sha256:
        raise IntegrityError("model is not the pre-registered frozen version")
    y = np.asarray(y_t, float)
    if np.isnan(y).any():
        raise ValueError("unlabelled target outcomes must be removed and reported, not scored")
    if source_ids is None:
        overlap = "UNVERIFIED (metadata cannot prove independence; authorised ID-level verification required)"
    else:
        both = set(source_ids) & set(target_ids)
        if both:
            raise LeakageError(f"{len(both)} participants appear in both source and target")
        overlap = "VERIFIED_DISJOINT_BY_AUTHORISED_ID_COMPARISON"
    res = evaluate_frozen(y.astype(int), model.predict_proba(X_t), n_boot=n_boot)
    return {**res, "evaluation_mode": "FROZEN", "model_version": model.manifest.version,
            "manifest_sha256": model.manifest.manifest_sha256, "overlap_status": overlap}


@dataclass
class UpdatedModel:
    """Result of using target outcomes to recalibrate. NOT a frozen external validation."""
    base_fingerprint: str
    intercept_shift: float
    slope: float
    status: str = "MODEL_UPDATING_NOT_FROZEN_VALIDATION"
    evaluation_mode: str = "UPDATED"
    note: str = "target outcomes were used; report separately from frozen results"

    def predict_proba(self, frozen: FrozenModel, X: pd.DataFrame) -> np.ndarray:
        p = np.clip(frozen.predict_proba(X), 1e-6, 1 - 1e-6)
        lp = np.log(p / (1 - p))
        return 1 / (1 + np.exp(-np.clip(self.intercept_shift + self.slope * lp, -30, 30)))


def update_calibration_on_target(frozen: FrozenModel, X_t: pd.DataFrame, y_t: np.ndarray) -> UpdatedModel:
    p = np.clip(frozen.predict_proba(X_t), 1e-6, 1 - 1e-6)
    a, b = _fit_platt(y_t, np.log(p / (1 - p)))
    return UpdatedModel(frozen.fingerprint(), a, b)
