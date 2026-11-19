"""Task-aware metrics, post-hoc calibration and threshold selection (binary classification only).

Probability metrics take predicted risks; threshold metrics take a 0/1 decision derived from a threshold chosen on a
validation partition (never on the evaluation partition). A metric that cannot be computed is recorded as failed with
its reason; it is not silently dropped. Survival tasks must define their own metrics (not implemented).
Calibration intercept/slope reuse experiments/today_audit/audit_core.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, log_loss,
                             roc_auc_score)

from yoad.experiments.today_audit.audit_core import cal_intercept, cal_slope


@dataclass(frozen=True)
class MetricSpec:
    name: str
    kind: str                     # "probability" | "threshold"
    fn: Callable
    tasks: tuple = ("binary_classification",)
    needs_both_classes: bool = True


def _log_oe(y, p):
    """log(observed / expected events); > 0 means the model under-predicts. Undefined without events."""
    o, e = float(np.sum(y)), float(np.sum(p))
    if o <= 0 or e <= 0:
        raise ValueError("log O:E undefined with zero observed or expected events")
    return float(np.log(o / e))


def _sens(y, d):
    return float(np.mean(d[y == 1] == 1))


def _spec(y, d):
    return float(np.mean(d[y == 0] == 0))


METRICS: dict[str, MetricSpec] = {m.name: m for m in [
    MetricSpec("roc_auc", "probability", lambda y, p: float(roc_auc_score(y, p))),
    MetricSpec("average_precision", "probability", lambda y, p: float(average_precision_score(y, p))),
    MetricSpec("brier", "probability", lambda y, p: float(brier_score_loss(y, p)), needs_both_classes=False),
    MetricSpec("log_loss", "probability", lambda y, p: float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])),
               needs_both_classes=False),
    MetricSpec("calibration_intercept", "probability", lambda y, p: cal_intercept(y, p)),
    MetricSpec("log_oe", "probability", lambda y, p: _log_oe(y, p), needs_both_classes=False),
    MetricSpec("calibration_slope", "probability", lambda y, p: cal_slope(y, p)),
    MetricSpec("balanced_accuracy", "threshold", lambda y, d: float(balanced_accuracy_score(y, d))),
    MetricSpec("sensitivity", "threshold", _sens),
    MetricSpec("specificity", "threshold", _spec),
    MetricSpec("f1_macro", "threshold", lambda y, d: float(f1_score(y, d, average="macro"))),
]}


def compute_metrics(names, y, p, threshold: float) -> tuple[dict, dict]:
    """Return (values, failures) for the requested metrics on one evaluation partition."""
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    vals, fails = {}, {}
    if p.shape != y.shape or not np.all(np.isfinite(p)) or p.min() < 0 or p.max() > 1:
        return {}, {n: "predictions are not valid probabilities aligned with labels" for n in names}
    d = (p >= threshold).astype(int)
    for n in names:
        m = METRICS[n]
        if m.needs_both_classes and len(np.unique(y)) < 2:
            fails[n] = "evaluation partition contains a single class"
            continue
        try:
            vals[n] = m.fn(y, p if m.kind == "probability" else d)
        except Exception as e:                      # recorded, never hidden
            fails[n] = f"{type(e).__name__}: {e}"
    return vals, fails


# ------------------------------------------------------------------------------------------------ calibration
def _logit(p):
    q = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def fit_calibrator(method: str, y_val, p_val, seed: int):
    """Fit a post-hoc calibrator on VALIDATION predictions. Returns a function p -> calibrated p."""
    y_val = np.asarray(y_val).astype(int)
    if method == "none":
        return lambda p: np.asarray(p, float)
    if len(np.unique(y_val)) < 2:
        raise ValueError("calibration requires both classes in the validation partition")
    if method == "sigmoid":
        lr = LogisticRegression(C=1e6, max_iter=5000, random_state=seed).fit(_logit(p_val).reshape(-1, 1), y_val)
        return lambda p: lr.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(np.asarray(p_val, float), y_val)
        return lambda p: iso.predict(np.asarray(p, float))
    raise NotImplementedError(f"calibration method {method!r} not implemented")


# ------------------------------------------------------------------------------------------------ thresholds
def select_threshold(method: str, value, y_val=None, p_val=None) -> float:
    """'fixed' uses `value`; 'youden' and 'target_sensitivity' use VALIDATION data only."""
    if method == "fixed":
        return float(value)
    y = np.asarray(y_val).astype(int); p = np.asarray(p_val, float)
    if len(np.unique(y)) < 2:
        raise ValueError("threshold selection requires both classes in the validation partition")
    cand = np.unique(p)
    sens = np.array([np.mean(p[y == 1] >= t) for t in cand])
    spec = np.array([np.mean(p[y == 0] < t) for t in cand])
    if method == "youden":
        return float(cand[int(np.argmax(sens + spec - 1))])
    if method == "target_sensitivity":
        ok = np.where(sens >= float(value))[0]
        return float(cand[ok.max()]) if ok.size else float(cand.min())
    raise NotImplementedError(f"threshold method {method!r} not implemented")
