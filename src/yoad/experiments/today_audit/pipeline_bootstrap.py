"""Full-pipeline (out-of-bag) bootstrap, overlap and leakage checks (2026-10-15 execution mission).

Pure functions; no file access. Synthetic-tested in tests/test_execution_mission.py.

The earlier E03 bootstrap resampled predictions of models that had already been fitted (conditional on the fitted models), so its
intervals ignored model-fitting variability. Here each replicate re-runs the WHOLE pipeline: resample participants, refit the
determinacy model, the weights and the learner under every training label rule, then score on the participants left out of the
resample (out-of-bag, OOB). Differences between rules are computed within a replicate, so they are paired.
Limitation: OOB sets are ~36.8% of participants, so absolute metrics are slightly pessimistic and noisier than 5-fold CV; the
paired DIFFERENCES between rules (the quantity of interest) are the primary output.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from yoad.experiments.today_audit import audit_core as ac


def default_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def oob_pipeline_bootstrap(X, XD, y_rules: dict, ref: str, det, comp, B=200, seed=20261015, make_model=default_model,
                           min_events=10, min_nonevents=10, w_clip=(0.01, 0.99)):
    """X: predictors (n, p); XD: determinacy-model covariates (n, q); y_rules[r]: float array (n,) with 1/0 and NaN where the rule
    leaves the participant unlabelled; ref: name of the reference rule in y_rules; det: bool, reference-determinate; comp: bool,
    competing event (excluded from evaluation and from the determinacy model).
    Returns (long DataFrame of per-replicate metrics, summary DataFrame with percentile CIs of metrics and paired differences)."""
    X = np.asarray(X, float); XD = np.asarray(XD, float); det = np.asarray(det, bool); comp = np.asarray(comp, bool)
    n = len(X); rng = np.random.default_rng(seed); rows = []
    for b in range(B):
        idx = rng.integers(0, n, n)
        oob = np.setdiff1d(np.arange(n), idx)
        tr_nc = idx[~comp[idx]]
        dm = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(XD[tr_nc], det[tr_nc].astype(int))
        pbar = det[tr_nc].mean()
        pi_tr = dm.predict_proba(XD[idx])[:, 1]; w_tr = pbar / pi_tr
        lo, hi = np.quantile(w_tr[det[idx]], w_clip); w_tr = np.clip(w_tr, lo, hi)
        E = oob[det[oob] & ~comp[oob]]
        yE = np.asarray(y_rules[ref], float)[E]
        if yE.sum() < min_events or (len(yE) - yE.sum()) < min_nonevents:
            continue
        wE = np.clip(pbar / dm.predict_proba(XD[E])[:, 1], lo, hi)
        for r, y in y_rules.items():
            y = np.asarray(y, float)
            lab = ~np.isnan(y[idx])
            mdl = make_model()
            if r == ref:
                try:
                    mdl.fit(X[idx][lab], y[idx][lab], **{mdl.steps[-1][0] + "__sample_weight": w_tr[lab]})
                except Exception:
                    mdl.fit(X[idx][lab], y[idx][lab])
            else:
                mdl.fit(X[idx][lab], y[idx][lab])
            p = mdl.predict_proba(X[E])[:, 1]
            rows.append({"rep": b, "rule": r, "n_eval": len(E), "events_eval": int(yE.sum()), "AUROC": ac.wauc(yE, p, wE),
                         "cal_intercept": ac.cal_intercept(yE, p, wE), "cal_slope": ac.cal_slope(yE, p, wE),
                         "logOE": float(np.log(ac.oe_ratio(yE, p, wE))), "Brier": ac.brier(yE, p, wE), "mean_pred": float(np.average(p, weights=wE))})
    long = pd.DataFrame(rows)
    if long.empty:
        return long, pd.DataFrame()
    piv = {m: long.pivot(index="rep", columns="rule", values=m) for m in ["AUROC", "cal_intercept", "cal_slope", "logOE", "Brier", "mean_pred"]}
    out = []
    for r in y_rules:
        for m, P in piv.items():
            v = P[r].dropna()
            rec = {"rule": r, "metric": m, "n_rep": len(v), "mean": v.mean(), "lo": v.quantile(.025), "hi": v.quantile(.975)}
            if r != ref:
                d = (P[r] - P[ref]).dropna()
                rec.update(diff_vs_ref_mean=d.mean(), diff_lo=d.quantile(.025), diff_hi=d.quantile(.975))
            out.append(rec)
    return long, pd.DataFrame(out)


# ----------------------------------------------------------------------------------------------------------- overlap (counts only)
def suppress(n, k=11):
    n = int(n)
    return n if (n == 0 or n >= k) else f"<{k}"


def candidate_overlap_count(a: pd.DataFrame, b: pd.DataFrame, keys, k=11):
    """Number of participants of `a` whose quasi-identifier tuple (e.g. birth year, sex, education, APOE genotype, baseline year-month)
    is found exactly once in `b` AND exactly once in `a` (a 1:1 candidate match). Returns (suppressed count, share of a). A
    non-zero count flags POSSIBLE overlap; it cannot prove it (no shared identifier exists). Never returns or prints row-level data."""
    ka = a[list(keys)].dropna(); kb = b[list(keys)].dropna()
    ca = ka.value_counts(); cb = kb.value_counts()
    uniq_a = ca[ca == 1].index; both = [t for t in uniq_a if cb.get(t, 0) == 1]
    return suppress(len(both), k), len(both) / max(len(a), 1)


# ----------------------------------------------------------------------------------------------------------- leakage / dates
def check_predictor_windows(pred_dates: pd.DataFrame, t0: pd.Series, lo_days=-90, hi_days=14):
    """pred_dates: columns = predictor names, values = measurement dates (NaT allowed); t0: baseline date per row (same index).
    Returns a dict {predictor: number of rows whose measurement is outside [t0+lo, t0+hi]} (0 = no leakage / window violations).
    A measurement after t0 + hi_days is potential future information."""
    out = {}
    for c in pred_dates.columns:
        d = (pd.to_datetime(pred_dates[c]) - pd.to_datetime(t0)).dt.days
        out[c] = int(((d < lo_days) | (d > hi_days)).sum())
    return out


def baseline_date_consistency(index_dates: pd.DataFrame, tol_days=0):
    """index_dates: columns = label-rule names, each the baseline/index date used by that rule. Returns the number of participants
    for whom the rules do not share one index date (differences > tol_days). 0 = harmonised."""
    d = index_dates.apply(pd.to_datetime)
    spread = (d.max(axis=1) - d.min(axis=1)).dt.days
    return int((spread > tol_days).sum())
