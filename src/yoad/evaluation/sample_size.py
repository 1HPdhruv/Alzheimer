"""Precision / sample-size planning for validating a binary-outcome prediction model.

PLANNING ONLY: every number returned depends on assumed prevalence and assumed spread of the model's linear
predictor (sigma); nothing here is estimated from real participant data and no model is trained.
Framework follows the precision-based approach of Riley et al. (Stat Med 2021, PMID 34031906); the formulas
below are standard approximations (Hanley-McNeil AUC variance; SE of ln(O/E)); calibration-slope precision is
obtained by simulation under the stated assumptions.
"""
from __future__ import annotations

import math

import numpy as np


def se_ln_oe(n: int, prevalence: float) -> float:
    """Approximate SE of ln(Observed/Expected) when the model is well calibrated: sqrt((1-phi)/(n*phi))."""
    _check(n, prevalence)
    return math.sqrt((1 - prevalence) / (n * prevalence))


def hanley_mcneil_se(auc: float, n_events: int, n_nonevents: int) -> float:
    """Hanley & McNeil (1982) SE of the C-statistic (classic approximation; recalled, not re-verified in this session)."""
    if not 0.5 <= auc < 1 or n_events < 2 or n_nonevents < 2:
        raise ValueError("need 0.5<=auc<1 and >=2 events and non-events")
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_events - 1) * (q1 - auc ** 2) + (n_nonevents - 1) * (q2 - auc ** 2)) / (n_events * n_nonevents)
    return math.sqrt(var)


def newcombe_se(c: float, n: int, prevalence: float) -> float:
    """SE of the C-statistic in the form attributed to Newcombe in Whittle et al. 2025 (PMC12210805, criterion 3):
    SE = sqrt( C(1-C) [1 + (N/2 - 1)(1-C)/(2-C) + (N/2 - 1) C/(1+C)] / (N^2 * phi * (1-phi)) ).
    CAUTION: the equation was read from a text extraction in which sub-expressions were garbled; this transcription must be
    confirmed against the PDF before it is cited. It is cross-checked against Hanley-McNeil and simulation in the tests."""
    if not (0.5 <= c < 1) or n <= 4 or not 0 < prevalence < 1:
        raise ValueError("need 0.5<=C<1, n>4, 0<prevalence<1")
    num = c * (1 - c) * (1 + (n / 2 - 1) * (1 - c) / (2 - c) + (n / 2 - 1) * c / (1 + c))
    return math.sqrt(num / (n ** 2 * prevalence * (1 - prevalence)))


def events_per_variable(n_events: int, n_candidate_params: int) -> float:
    if n_candidate_params <= 0:
        raise ValueError("n_candidate_params must be > 0")
    return n_events / n_candidate_params


def riley_shrinkage_n(n_params: int, r2_cs: float, shrinkage: float = 0.9) -> float:
    """Development-sample-size criterion (expected shrinkage >= S): n = P / ((S-1) ln(1 - R2cs/S)).
    UNVERIFIED closed form (recalled from Riley et al.; the BMJ 2020 and Stat Med 2019 full texts were not accessible).
    Verified qualitatively in PMC12106283 (Statistical primer): target S >= 0.9; if no information, use R2cs = 15% of its maximum;
    the C-statistic can substitute for R2cs. Treat outputs as illustrative floors only."""
    if not (0 < r2_cs < 1 and 0 < shrinkage < 1 and n_params > 0):
        raise ValueError("invalid arguments")
    return n_params / ((shrinkage - 1) * math.log(1 - r2_cs / shrinkage))


def max_r2_cs(prevalence: float) -> float:
    """Upper bound of Cox-Snell R2 for a binary outcome: 1 - (phi^phi (1-phi)^(1-phi))^2."""
    if not 0 < prevalence < 1:
        raise ValueError("prevalence in (0,1)")
    return 1 - (prevalence ** prevalence * (1 - prevalence) ** (1 - prevalence)) ** 2


def _check(n, p):
    if n <= 0 or not 0 < p < 1:
        raise ValueError("need n>0 and 0<prevalence<1")


def _expit(x):
    return 1 / (1 + np.exp(-x))


def _intercept_for_prevalence(sigma: float, prevalence: float) -> float:
    z = np.random.default_rng(0).standard_normal(200_000) * sigma
    lo, hi = -15.0, 15.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _expit(mid + z).mean() < prevalence:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _auc(y, s):
    r = np.argsort(np.argsort(s)) + 1.0
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _slope_se(y, lp, iters=25):
    X = np.column_stack([np.ones_like(lp), lp]); b = np.zeros(2)
    for _ in range(iters):
        p = _expit(X @ b); W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, X.T @ (y - p)); b = b + step
        if np.abs(step).max() < 1e-8:
            break
    p = _expit(X @ b); W = p * (1 - p)
    cov = np.linalg.inv(X.T @ (X * W[:, None]) + 1e-9 * np.eye(2))
    return float(b[1]), float(math.sqrt(cov[1, 1]))


def simulate_validation_precision(n: int, prevalence: float, sigma: float, n_sim: int = 300, seed: int = 1) -> dict:
    """Under assumptions (true linear predictor ~ N(mu, sigma^2), model perfectly calibrated), return expected
    events, C-statistic, and typical half-widths of 95% CIs for C, calibration slope and ln(O/E)."""
    _check(n, prevalence)
    rng = np.random.default_rng(seed); mu = _intercept_for_prevalence(sigma, prevalence)
    aucs, slopes, ses, ev = [], [], [], []
    for _ in range(n_sim):
        lp = mu + sigma * rng.standard_normal(n)
        y = (rng.random(n) < _expit(lp)).astype(float)
        if y.sum() < 3 or y.sum() > n - 3:
            continue
        aucs.append(_auc(y, lp)); b, se = _slope_se(y, lp); slopes.append(b); ses.append(se); ev.append(y.sum())
    aucs, slopes = np.array(aucs), np.array(slopes)
    return {"n": n, "prevalence": prevalence, "sigma": sigma, "mean_events": float(np.mean(ev)),
            "mean_auc": float(aucs.mean()), "auc_ci_halfwidth": float(1.96 * aucs.std(ddof=1)),
            "slope_ci_halfwidth": float(1.96 * np.mean(ses)), "slope_empirical_sd": float(slopes.std(ddof=1)),
            "ln_oe_ci_halfwidth": float(1.96 * se_ln_oe(n, prevalence)), "n_sim": len(aucs)}
