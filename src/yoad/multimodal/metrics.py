"""Metrics and uncertainty for the multimodal benchmark (PROTOCOL.md §7).

Definitions are those of the frozen study (src/benchmark/metrics.py, experiments/today_audit/audit_core.py): AUROC (ties count
one half), step-wise average precision, Brier score, log loss, log O:E = log(sum observed / sum predicted) (negative = aggregate
over-prediction), calibration intercept (offset logistic model, slope fixed at 1; logit clipped at 1e-6) and calibration slope
(logistic model with free intercept, maximum likelihood). Fast exact implementations are used inside the bootstrap; their
agreement with the scikit-learn / audit_core versions is a synthetic test.

Nested-CV point estimates: in each outer repeat the out-of-fold predictions of all participants are pooled and the metric is
computed; the reported value is the mean over repeats (with the across-repeat SD as a split-variability summary). Intervals:
participant bootstrap (B resamples of participants with replacement; every repeat is scored on the same resample and averaged),
percentile 95% limits. Paired contrasts use the same resample for both candidates, on the participants that have predictions
from both. These intervals condition on the fitted fold models (no refitting inside the bootstrap) and are therefore approximate
(Bates, Hastie & Tibshirani 2024); this is stated with every result. Counts 1..10 are suppressed in every exported table.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm, rankdata

from yoad.multimodal import registry as R

CORE_METRICS = ("roc_auc", "average_precision", "brier", "log_loss", "log_oe", "calibration_intercept", "calibration_slope")
CONTRAST_METRICS = ("roc_auc", "average_precision", "brier", "log_loss", "log_oe", "calibration_slope")


def sup(n):
    n = int(n)
    return n if n == 0 or n >= R.K_SUPPRESS else f"<{R.K_SUPPRESS}"


def _logit(p):
    q = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def auroc(y, p):
    y = np.asarray(y, int); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    if y.sum() == 0:
        return np.nan
    o = np.argsort(-p, kind="mergesort"); y, p = y[o], p[o]
    last = np.r_[np.flatnonzero(np.diff(p) != 0), len(p) - 1]
    tp = np.cumsum(y)[last]; prec = tp / (last + 1); rec = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def cal_intercept(y, p):
    y = np.asarray(y, float); off = _logit(p); a = 0.0
    for _ in range(100):
        mu = expit(a + off); h = np.sum(mu * (1 - mu))
        if h <= 0:
            return np.nan
        step = np.sum(y - mu) / h; a += step
        if abs(step) < 1e-12:
            break
    return float(a) if np.isfinite(a) else np.nan


def cal_slope(y, p):
    y = np.asarray(y, float); X = np.column_stack([np.ones(len(y)), _logit(p)]); b = np.array([0.0, 1.0])
    for _ in range(100):
        mu = expit(X @ b); W = mu * (1 - mu)
        H = X.T @ (X * W[:, None])
        try:
            step = np.linalg.solve(H, X.T @ (y - mu))
        except np.linalg.LinAlgError:
            return np.nan
        b = b + step
        if not np.all(np.isfinite(b)) or np.abs(b).max() > 1e6:
            return np.nan
        if np.abs(step).max() < 1e-10:
            break
    return float(b[1])


def point_metrics(y, p, names=CORE_METRICS) -> dict:
    y = np.asarray(y, int); p = np.asarray(p, float)
    out = {}
    for n in names:
        if n == "roc_auc":
            out[n] = auroc(y, p)
        elif n == "average_precision":
            out[n] = average_precision(y, p)
        elif n == "brier":
            out[n] = float(np.mean((p - y) ** 2))
        elif n == "log_loss":
            q = np.clip(p, 1e-15, 1 - 1e-15); out[n] = float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
        elif n == "log_oe":
            out[n] = float(np.log(y.sum() / p.sum())) if y.sum() > 0 and p.sum() > 0 else np.nan
        elif n == "calibration_intercept":
            out[n] = cal_intercept(y, p)
        elif n == "calibration_slope":
            out[n] = cal_slope(y, p)
        else:
            raise ValueError(f"unknown metric {n}")
    return out


def threshold_metrics(y, p, t) -> dict:
    """t: one threshold per participant (the threshold of the fold that scored it, chosen on inner out-of-fold data)."""
    y = np.asarray(y, int); d = (np.asarray(p, float) >= np.asarray(t, float)).astype(int)
    tp, tn = int(((d == 1) & (y == 1)).sum()), int(((d == 0) & (y == 0)).sum())
    fp, fn = int(((d == 1) & (y == 0)).sum()), int(((d == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan; spec = tn / (tn + fp) if tn + fp else np.nan
    return {"sensitivity": sens, "specificity": spec, "balanced_accuracy": (sens + spec) / 2,
            "ppv": tp / (tp + fp) if tp + fp else np.nan, "npv": tn / (tn + fn) if tn + fn else np.nan}


def repeat_summary(y, P, T=None, names=CORE_METRICS) -> dict:
    """P: repeats x participants (NaN = no prediction); y: participants. Returns {metric: (mean over repeats, SD over repeats)}."""
    vals = {n: [] for n in names}
    thr = {k: [] for k in ("sensitivity", "specificity", "balanced_accuracy", "ppv", "npv")}
    for r in range(P.shape[0]):
        ok = ~np.isnan(P[r])
        m = point_metrics(y[ok], P[r, ok], names)
        for n in names:
            vals[n].append(m[n])
        if T is not None:
            tm = threshold_metrics(y[ok], P[r, ok], T[r, ok])
            for k in thr:
                thr[k].append(tm[k])
    out = {n: (float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if len(v) > 1 else np.nan) for n, v in vals.items()}
    if T is not None:
        out.update({k: (float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if len(v) > 1 else np.nan) for k, v in thr.items()})
    return out


def _boot_stats(y, P, idx, names):
    return np.array([[point_metrics(y[idx], P[r, idx], names)[n] for n in names] for r in range(P.shape[0])]).mean(axis=0)


def bootstrap(y, P, names=CORE_METRICS, B: int = R.BOOT_B, seed: int = R.SEED, alpha: float = R.ALPHA) -> dict:
    """Marginal participant-bootstrap percentile intervals of the repeat-averaged metrics (participants predicted in every
    repeat)."""
    ok = ~np.isnan(P).any(axis=0)
    y, P = np.asarray(y, int)[ok], P[:, ok]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(B):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        draws.append(_boot_stats(y, P, idx, names))
    D = np.array(draws)
    lo, hi = np.nanquantile(D, alpha / 2, axis=0), np.nanquantile(D, 1 - alpha / 2, axis=0)
    return {n: (float(lo[i]), float(hi[i])) for i, n in enumerate(names)} | {"_B_used": len(draws)}


def paired_bootstrap(y, Pa, Pb, names=CONTRAST_METRICS, B: int = R.BOOT_B, seed: int = R.SEED, alpha: float = R.ALPHA) -> dict:
    """Difference a - b of repeat-averaged metrics on the participants predicted by both in every repeat. Returns per metric
    {difference, ci_low, ci_high, se} plus n and events (raw ints; the caller suppresses)."""
    ok = ~(np.isnan(Pa).any(axis=0) | np.isnan(Pb).any(axis=0))
    y, Pa, Pb = np.asarray(y, int)[ok], Pa[:, ok], Pb[:, ok]
    full = np.arange(len(y))
    point = _boot_stats(y, Pa, full, names) - _boot_stats(y, Pb, full, names)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(B):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        draws.append(_boot_stats(y, Pa, idx, names) - _boot_stats(y, Pb, idx, names))
    D = np.array(draws)
    out = {"n": int(len(y)), "events": int(y.sum()), "B_used": len(draws)}
    for i, n in enumerate(names):
        col = D[:, i][np.isfinite(D[:, i])]
        out[n] = {"difference": float(point[i]), "ci_low": float(np.quantile(col, alpha / 2)) if len(col) else np.nan,
                  "ci_high": float(np.quantile(col, 1 - alpha / 2)) if len(col) else np.nan,
                  "se": float(np.std(col, ddof=1)) if len(col) > 1 else np.nan}
    return out


def bonferroni(diff: float, se: float, m: int, alpha: float = R.ALPHA) -> tuple[float, float]:
    """Normal-approximation family-wise interval for one of m contrasts (bootstrap SE)."""
    if not (np.isfinite(diff) and np.isfinite(se)) or m < 1:
        return np.nan, np.nan
    z = norm.ppf(1 - alpha / (2 * m))
    return float(diff - z * se), float(diff + z * se)


def calibration_bins(y, p, n_bins: int = 5) -> pd.DataFrame:
    """Quantile bins of predicted risk with suppression: n is suppressed below 11; the observed proportion is withheld when the
    bin's event or non-event count is 1..10."""
    y = np.asarray(y, int); p = np.asarray(p, float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for k in range(len(edges) - 1):
        m = b == k
        n, e = int(m.sum()), int(y[m].sum())
        ok = n >= R.K_SUPPRESS and (e == 0 or e >= R.K_SUPPRESS) and (n - e == 0 or n - e >= R.K_SUPPRESS)
        rows.append({"bin": k + 1, "n": sup(n), "mean_predicted": float(p[m].mean()) if n else np.nan,
                     "observed_proportion": float(y[m].mean()) if ok else np.nan})
    return pd.DataFrame(rows)


def classify(lo, hi, margin) -> str:
    """The frozen interval rule (design_lib.classify_interval), shortened."""
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return "not estimable"
    if lo > margin or hi < -margin:
        return "outside margin"
    if lo >= -margin and hi <= margin:
        return "inside margin"
    return "inconclusive"


def youden(y, p) -> float:
    from yoad.benchmark.metrics import select_threshold
    return select_threshold("youden", None, y, p)


def finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)
