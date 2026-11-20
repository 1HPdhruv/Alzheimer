"""Synthetic-testable interfaces for frozen external validation.

Nothing here touches real participants. A frozen model is represented by a spec
(feature list, preprocessing, horizon, version) plus a scoring callable; the
target cohort is only ever *scored*, never refit.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from yoad.evaluation.sample_size import newcombe_se


class HarmonizationError(ValueError):
    pass


@dataclass(frozen=True)
class FrozenModelSpec:
    name: str
    version: str
    horizon_months: int
    features: tuple[str, ...]
    preprocessing: dict = field(default_factory=dict)  # fixed constants only (e.g. means/SDs from source cohort)

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, default=list).encode()).hexdigest()[:16]


def check_harmonization(spec: FrozenModelSpec, target_cols: list[str], mapping_status: dict[str, str]) -> dict:
    """Fail loudly: every required feature must be present AND mapped as usable.

    mapping_status values: supported | transform | uncertain | incompatible.
    """
    missing = [f for f in spec.features if f not in target_cols]
    bad = [f for f in spec.features if mapping_status.get(f, "uncertain") in ("incompatible", "uncertain")]
    unmapped = [f for f in spec.features if f not in mapping_status]
    if missing or bad or unmapped:
        raise HarmonizationError({"missing_in_target": missing, "incompatible_or_uncertain": bad, "unmapped": unmapped})
    return {"ok": True, "needs_transform": [f for f in spec.features if mapping_status[f] == "transform"]}


def apply_locked_preprocessing(spec: FrozenModelSpec, X: pd.DataFrame) -> pd.DataFrame:
    """Standardise with the *source* constants stored in the spec; never target statistics."""
    out = X[list(spec.features)].astype(float).copy()
    for f, (mu, sd) in spec.preprocessing.get("standardize", {}).items():
        out[f] = (out[f] - mu) / sd
    return out


def _auc(y, s):
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def _recal(y, p):
    """Calibration slope (logistic of y on logit p, Newton) and calibration-in-the-large (intercept, slope fixed at 1)."""
    lp = _logit(p); y = np.asarray(y, float)
    X = np.c_[np.ones_like(lp), lp]; beta = np.array([0.0, 1.0])
    for _ in range(50):
        mu = 1 / (1 + np.exp(-X @ beta)); W = mu * (1 - mu)
        step = np.linalg.solve(X.T @ (X * W[:, None]) + 1e-9 * np.eye(2), X.T @ (y - mu))
        beta = beta + step
        if np.abs(step).max() < 1e-8: break
    a = 0.0
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(a + lp))); step = (y - mu).sum() / max((mu * (1 - mu)).sum(), 1e-12); a += step
        if abs(step) < 1e-10: break
    return float(beta[1]), float(a)


def evaluate_frozen(y: np.ndarray, p: np.ndarray, n_boot: int = 500, seed: int = 0,
                    min_events: int = 20, min_nonevents: int = 20) -> dict:
    """Metrics for already-scored predictions. y must contain only labelled (0/1) rows.

    Returns status='insufficient' (no metrics) instead of silently proceeding when counts are too low.
    """
    y = np.asarray(y); p = np.asarray(p, float)
    if np.isnan(y.astype(float)).any():
        raise ValueError("unlabelled (NaN) outcomes must be removed and reported by the caller, not scored as 0")
    ne, nn = int(y.sum()), int((1 - y).sum())
    res = {"n": len(y), "events": ne, "nonevents": nn}
    if ne < min_events or nn < min_nonevents:
        return {**res, "status": "insufficient", "reason": f"events={ne}, nonevents={nn} below {min_events}/{min_nonevents}"}
    slope, citl = _recal(y, p)
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); boots = []
    for _ in range(n_boot):
        b = rng.choice(idx, len(y))
        if y[b].sum() in (0, len(b)): continue
        boots.append((_auc(y[b], p[b]), float(np.mean((p[b] - y[b]) ** 2)), y[b].mean() / p[b].mean()))
    lo, hi = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
    return {**res, "status": "ok", "auroc": float(_auc(y, p)), "auroc_ci": (float(lo[0]), float(hi[0])),
            "brier": float(np.mean((p - y) ** 2)), "brier_ci": (float(lo[1]), float(hi[1])),
            "oe_ratio": float(y.mean() / p.mean()), "oe_ci": (float(lo[2]), float(hi[2])),
            "cal_slope": slope, "cal_in_large": citl, "auroc_se_newcombe": float(newcombe_se(_auc(y, p), len(y), y.mean()))}


def run_record(spec: FrozenModelSpec, config: dict, metrics: dict, synthetic: bool) -> dict:
    return {"model": spec.name, "model_version": spec.version, "spec_fingerprint": spec.fingerprint(),
            "features": list(spec.features), "preprocessing": spec.preprocessing, "config": config,
            "metrics": metrics, "SYNTHETIC": bool(synthetic)}
