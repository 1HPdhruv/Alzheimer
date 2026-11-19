"""Today's audit core (2026-10-15): calibration metrics, Definition B labels, censoring-aware incidence estimators.

Pure functions; no file access. Unit-tested on synthetic data (tests/test_today_audit.py). The real-data runner
(run_today_audit.py) must only be executed in an environment approved under the ADNI/NACC agreements (see METHODS_AND_ENDPOINTS.md).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


# ----------------------------------------------------------------------------------------------------------- calibration metrics
def _logit(p):
    q = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def wauc(y, p, w=None):
    """Weighted Mann-Whitney AUROC, ties = 1/2."""
    y = np.asarray(y, float); p = np.asarray(p, float); w = np.ones_like(y) if w is None else np.asarray(w, float)
    o = np.argsort(p, kind="mergesort"); y, p, w = y[o], p[o], w[o]
    pos = y == 1
    wp, wn = w * pos, w * (~pos)
    _, start = np.unique(p, return_index=True)
    cp = np.add.reduceat(wp, start); cn = np.add.reduceat(wn, start)
    before = np.r_[0, np.cumsum(cn)[:-1]]
    return float((cp * (before + 0.5 * cn)).sum() / (wp.sum() * wn.sum()))


def oe_ratio(y, p, w=None):
    """Observed / expected events: sum(w*y) / sum(w*p). > 1 means the model under-predicts."""
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    return float((w * np.asarray(y)).sum() / (w * np.asarray(p)).sum())


def cal_intercept(y, p, w=None):
    """Calibration-in-the-large: intercept a of logit P(y=1) = a + 1*logit(p) (offset model). 0 = calibrated; > 0 under-prediction."""
    y = np.asarray(y, float); off = _logit(p); w = np.ones_like(y) if w is None else np.asarray(w, float)
    def nll(a):
        eta = a[0] + off
        return -np.sum(w * (y * eta - np.logaddexp(0, eta)))
    return float(minimize(nll, x0=[0.0], method="BFGS").x[0])


def cal_slope(y, p, w=None):
    """Calibration slope b of logit P(y=1) = a + b*logit(p). 1 = ideal; < 1 predictions too extreme; > 1 too moderate."""
    y = np.asarray(y, float); w = np.ones_like(y) if w is None else np.asarray(w, float)
    m = LogisticRegression(C=1e8, max_iter=5000).fit(_logit(p).reshape(-1, 1), y, sample_weight=w)
    return float(m.coef_[0, 0])


def brier(y, p, w=None):
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    return float((w * (np.asarray(p) - np.asarray(y)) ** 2).sum() / w.sum())


def pr_auc(y, p, w=None):
    """Weighted average precision (step-wise, tie-invariant: one threshold per distinct score, as in scikit-learn)."""
    y = np.asarray(y, float); p = np.asarray(p, float); w = np.ones_like(y) if w is None else np.asarray(w, float)
    o = np.argsort(-p, kind="mergesort"); y, p, w = y[o], p[o], w[o]
    last = np.r_[np.flatnonzero(np.diff(p) != 0), len(p) - 1]  # last index of each tie group
    tp = np.cumsum(w * y)[last]; fp = np.cumsum(w * (1 - y))[last]
    prec = tp / (tp + fp); rec = tp / (w * y).sum()
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def ici(y, p, w=None):
    """Integrated calibration index with an isotonic (monotone) calibration curve: weighted mean |p - smoothed observed|."""
    y = np.asarray(y, float); p = np.asarray(p, float); w = np.ones_like(y) if w is None else np.asarray(w, float)
    iso = IsotonicRegression(out_of_bounds="clip").fit(p, y, sample_weight=w)
    return float((w * np.abs(p - iso.predict(p))).sum() / w.sum())


def calibration_curve(y, p, w=None, bins=10):
    """Quantile-binned observed vs predicted risk (weighted)."""
    y = np.asarray(y, float); p = np.asarray(p, float); w = np.ones_like(y) if w is None else np.asarray(w, float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"bin": b, "n": int(m.sum()), "pred": float((w[m] * p[m]).sum() / w[m].sum()), "obs": float((w[m] * y[m]).sum() / w[m].sum())})
    return pd.DataFrame(rows)


def metric_set(y, p, w=None):
    return {"AUROC": wauc(y, p, w), "PR_AUC": pr_auc(y, p, w), "Brier": brier(y, p, w), "cal_intercept": cal_intercept(y, p, w),
            "cal_slope": cal_slope(y, p, w), "OE": oe_ratio(y, p, w), "ICI": ici(y, p, w),
            "mean_pred": float(np.average(p, weights=w)), "obs_rate": float(np.average(y, weights=w))}


def prevalence_shift_diagnostic(y, p, w=None):
    """Separates a pure prevalence (intercept) shift from other miscalibration: re-centre predictions with the fitted calibration
    intercept and report the ICI that remains. Descriptive diagnostic: it uses the evaluation labels."""
    a = cal_intercept(y, p, w)
    p_adj = 1 / (1 + np.exp(-(_logit(p) + a)))
    return {"cal_intercept": a, "ICI_raw": ici(y, p, w), "ICI_after_intercept_fix": ici(y, p_adj, w), "cal_slope": cal_slope(y, p, w)}


# ----------------------------------------------------------------------------------------------------------- Definition B
def definition_B(t0, visits: pd.DataFrame, window_months=(21, 27), endpoint="ad"):
    """Status at the dated, non-conflicting visit closest to t0 + 24 months inside [t0 + lo months, t0 + hi months].
    visits: date, dx (DEM/NONDEM/other), dem_type (AD/OTHER/UNKNOWN). Returns EVENT / NONEVENT / INDETERMINATE + reason.
    Same-date DEM+NONDEM records are a conflict (excluded as uncertain); non-AD (or unknown-type) dementia at the chosen visit is
    INDETERMINATE for the AD endpoint. Unlike Definition A this is a point-prevalence status, not cumulative incidence: a participant
    demented at month 18 who reverts by month 24 is a non-event, and events after 24 months inside the window count."""
    t0 = pd.Timestamp(t0); H = t0 + pd.DateOffset(months=24)
    lo, hi = t0 + pd.DateOffset(months=window_months[0]), t0 + pd.DateOffset(months=window_months[1])
    v = visits.copy(); v["date"] = pd.to_datetime(v["date"], errors="coerce")
    v = v[v.date.notna() & (v.date >= lo) & (v.date <= hi)]
    if "dem_type" not in v:
        v["dem_type"] = "AD"
    rows = []
    for d, g in v.groupby("date"):
        s = set(g.dx)
        if "DEM" in s and "NONDEM" in s:
            continue
        if "DEM" in s:
            t = set(g.loc[g.dx == "DEM", "dem_type"])
            rows.append((d, "DEM", t.pop() if len(t) == 1 else "UNKNOWN"))
        elif "NONDEM" in s:
            rows.append((d, "NONDEM", ""))
    if not rows:
        return "INDETERMINATE", "no_valid_visit_in_window"
    c = pd.DataFrame(rows, columns=["date", "dx", "typ"])
    c["dist"] = (c.date - H).abs()
    r = c.sort_values(["dist", "date"]).iloc[0]
    if r.dx == "NONDEM":
        return "NONEVENT", "nondementia_at_window_visit"
    if endpoint == "ad" and r.typ != "AD":
        return "INDETERMINATE", "nonAD_or_unknown_dementia_at_window_visit"
    return "EVENT", "dementia_at_window_visit"


# ----------------------------------------------------------------------------------------------------------- incidence estimators
def turnbull(L, R, tol=1e-8, maxit=5000):
    """NPMLE for interval-censored times (L, R]; R = inf for right-censored. Returns innermost intervals and masses."""
    L = np.asarray(L, float); R = np.asarray(R, float)
    lefts = np.unique(L); rights = np.unique(R[np.isfinite(R)])
    pts = sorted(set([(l, 0) for l in lefts] + [(r, 1) for r in rights]))
    inner = [(pts[i][0], pts[i + 1][0]) for i in range(len(pts) - 1) if pts[i][1] == 0 and pts[i + 1][1] == 1]
    inner.append((max(L.max(), rights.max() if len(rights) else 0), np.inf))
    a = np.array([x for x, _ in inner]); b = np.array([y for _, y in inner])
    q = ((a[None, :] >= L[:, None]) & (b[None, :] <= R[:, None])).astype(float)
    s = np.full(len(inner), 1.0 / len(inner))
    for _ in range(maxit):
        s_new = (q * s / (q @ s)[:, None]).mean(axis=0)
        if np.abs(s_new - s).max() < tol:
            s = s_new; break
        s = s_new
    return inner, s


def F_at(inner, s, t):
    lo = sum(m for (a, b), m in zip(inner, s) if b <= t)
    hi = sum(m for (a, b), m in zip(inner, s) if a < t)
    return lo, hi


def intervals_with_death(L, R, death_day, mode):
    """Death handling for the interval-censored incidence of recorded AD dementia.
    mode='censor'  : deaths without recorded dementia are censored at their last assessment (NET risk; assumes they would
                     otherwise behave like survivors - the estimator used on 2026-10-15).
    mode='absorb'  : deaths without recorded dementia are treated as never developing recorded dementia (cumulative incidence of
                     RECORDED dementia before death, i.e. competing-risk 'crude' risk); their interval becomes (inf, inf).
    Undiagnosed dementia between the last visit and death is ignored in both modes (Leffondre et al. 2013)."""
    L = np.asarray(L, float).copy(); R = np.asarray(R, float).copy(); dd = np.asarray(death_day, float)
    died_no_event = np.isfinite(dd) & ~np.isfinite(R)
    if mode == "absorb":
        L[died_no_event] = np.inf
    elif mode != "censor":
        raise ValueError(mode)
    return L, R


def turnbull_with_absorbing(L, R, t):
    """F(t) when some participants are known never to have the event (L = inf): estimate on the others and scale by their share."""
    L = np.asarray(L, float); R = np.asarray(R, float)
    never = np.isinf(L)
    inner, s = turnbull(L[~never], R[~never])
    lo, hi = F_at(inner, s, t)
    k = (~never).mean()
    return lo * k, hi * k


def censoring_bounds(L, R, t):
    """Crude bounds for F(t) without the non-informative-censoring assumption: censored-before-t treated as all non-events (lower)
    or all events (upper). Interval events straddling t are counted as non-events (lower) / events (upper)."""
    L = np.asarray(L, float); R = np.asarray(R, float)
    event_by_t = np.isfinite(R) & (R <= t)
    unknown = (~event_by_t) & (L < t)
    return float(event_by_t.mean()), float((event_by_t | unknown).mean())


def icph_fit(L, R, X, cuts):
    """Interval-censored piecewise-exponential proportional-hazards model (maximum likelihood). Hazard h(t|x) = h_k exp(x'beta)
    on [cuts_k, cuts_{k+1}). Likelihood contribution S(L|x) - S(R|x) (S(inf) = 0). Returns (log_h, beta)."""
    L = np.asarray(L, float); R = np.asarray(R, float); X = np.asarray(X, float)
    cuts = np.asarray(cuts, float); K = len(cuts)

    def cumbase(t, logh):
        h = np.exp(logh)
        t = np.minimum(t, 1e12)
        widths = np.clip(t[:, None] - cuts[None, :], 0, np.r_[np.diff(cuts), np.inf][None, :])
        return widths @ h

    def nll(theta):
        logh, beta = theta[:K], theta[K:]
        eta = X @ beta
        SL = np.exp(-cumbase(L, logh) * np.exp(eta))
        SR = np.where(np.isfinite(R), np.exp(-cumbase(np.where(np.isfinite(R), R, 0.0), logh) * np.exp(eta)), 0.0)
        return -np.sum(np.log(np.clip(SL - SR, 1e-300, None))) + 1e-4 * np.sum(beta ** 2)

    th0 = np.r_[np.full(K, np.log(1e-3)), np.zeros(X.shape[1])]
    res = minimize(nll, th0, method="L-BFGS-B")
    return res.x[:K], res.x[K:]


def icph_standardized_F(L, R, X, cuts, t):
    """Covariate-standardised F(t) = mean_i [1 - S(t | x_i)] from the interval-censored PH model. Valid when censoring is
    non-informative CONDITIONAL on X (weaker than the unconditional assumption of the Turnbull NPMLE)."""
    logh, beta = icph_fit(L, R, X, cuts)
    h = np.exp(logh); cuts = np.asarray(cuts, float)
    H0 = np.clip(t - cuts, 0, np.r_[np.diff(cuts), np.inf]) @ h
    return float(np.mean(1 - np.exp(-H0 * np.exp(np.asarray(X, float) @ beta))))


# ----------------------------------------------------------------------------------------------------------- eligibility
ADVISORY_PTID = r"^381_S_\d{5}$"  # ADNI March-2026 advisory: participants removed from the database, not to be used


def advisory_exclusion_mask(ptids) -> np.ndarray:
    """True where a PTID falls under the March-2026 ADNI advisory."""
    return pd.Series(ptids).astype(str).str.match(ADVISORY_PTID).to_numpy()
