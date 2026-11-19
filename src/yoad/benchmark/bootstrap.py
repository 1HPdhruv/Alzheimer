"""Participant-level (cluster) bootstrap for metric uncertainty and paired model differences.

Resampling unit = participant: every resample draws participants with replacement and keeps ALL of their observations, so
repeated visits of one person are never treated as independent (the reference paper's public code resampled visits).
Paired differences use the SAME resampled participants for both models. The intervals are conditional on the fitted models
(they do not include model-refitting variability; for that use experiments/today_audit/pipeline_bootstrap.py).
Resamples lacking one class are skipped and counted, never silently replaced.
"""
from __future__ import annotations

import numpy as np

from yoad.benchmark.metrics import METRICS


def cluster_bootstrap(y, preds: dict, groups, metrics, B: int = 1000, seed: int = 0, reference: str | None = None,
                      alpha: float = 0.05) -> dict:
    """preds: {model_name: probability array aligned with y}. Returns point estimates, percentile CIs and, if `reference` is
    given, paired differences (model - reference) with CIs, plus the number of skipped resamples."""
    y = np.asarray(y).astype(int); g = np.asarray(groups)
    for n, p in preds.items():
        if len(p) != len(y):
            raise ValueError(f"predictions for {n!r} not aligned with labels")
    bad = [m for m in metrics if METRICS[m].kind != "probability"]
    if bad:
        raise ValueError(f"cluster_bootstrap supports probability metrics only; got {bad}")
    if reference is not None and reference not in preds:
        raise ValueError(f"reference {reference!r} not among predictions")
    uniq, inv = np.unique(g, return_inverse=True)
    rows_of = [np.where(inv == k)[0] for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    draws = {(n, m): [] for n in preds for m in metrics}
    diffs = {(n, m): [] for n in preds if n != reference for m in metrics} if reference else {}
    skipped = 0
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_of[k] for k in pick])
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            skipped += 1
            continue
        vals = {(n, m): METRICS[m].fn(yb, np.asarray(p)[idx]) for n, p in preds.items() for m in metrics}
        for k, v in vals.items():
            draws[k].append(v)
        for (n, m) in diffs:
            diffs[(n, m)].append(vals[(n, m)] - vals[(reference, m)])
    q = lambda a: (float(np.quantile(a, alpha / 2)), float(np.quantile(a, 1 - alpha / 2))) if len(a) else (None, None)
    out = {"unit": "participant", "B_requested": B, "B_used": B - skipped, "skipped_single_class": skipped,
           "n_participants": int(len(uniq)), "n_observations": int(len(y)), "estimates": [], "paired_differences": []}
    for n, p in preds.items():
        for m in metrics:
            lo, hi = q(draws[(n, m)])
            out["estimates"].append({"model": n, "metric": m, "point": float(METRICS[m].fn(y, np.asarray(p))),
                                     "ci_low": lo, "ci_high": hi})
    for (n, m), d in diffs.items():
        lo, hi = q(d)
        point = float(METRICS[m].fn(y, np.asarray(preds[n])) - METRICS[m].fn(y, np.asarray(preds[reference])))
        out["paired_differences"].append({"model": n, "reference": reference, "metric": m, "difference": point,
                                          "ci_low": lo, "ci_high": hi})
    return out
