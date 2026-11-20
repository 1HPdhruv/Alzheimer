import numpy as np
import pandas as pd
import pytest
from yoad.experiments.today_audit import audit_core as ac

def _sim(n=20000, a=0.0, b=1.0, seed=0):
    rng = np.random.default_rng(seed)
    lp = rng.normal(-1, 1.2, n)
    p = 1 / (1 + np.exp(-lp))
    y = rng.binomial(1, 1 / (1 + np.exp(-(a + b * lp))))
    return y, p

def test_perfect_calibration_recovered():
    y, p = _sim()
    m = ac.metric_set(y, p)
    assert abs(m["cal_intercept"]) < 0.05 and abs(m["cal_slope"] - 1) < 0.05 and abs(m["OE"] - 1) < 0.05 and m["ICI"] < 0.02

def test_known_intercept_shift_recovered_and_removed_by_recentring():
    y, p = _sim(a=0.5)
    assert abs(ac.cal_intercept(y, p) - 0.5) < 0.06
    d = ac.prevalence_shift_diagnostic(y, p)
    assert d["ICI_after_intercept_fix"] < d["ICI_raw"] / 3

def test_known_slope_recovered():
    y, p = _sim(b=0.7)
    assert abs(ac.cal_slope(y, p) - 0.7) < 0.05

def test_weighted_auc_matches_sklearn_and_weights_equal_duplication():
    from sklearn.metrics import roc_auc_score, average_precision_score
    y, p = _sim(n=3000, seed=1)
    assert ac.wauc(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-10)
    assert ac.pr_auc(y, p) == pytest.approx(average_precision_score(y, p), abs=1e-10)
    w = np.r_[np.full(1500, 2.0), np.ones(1500)]
    yd, pd_ = np.r_[y[:1500], y], np.r_[p[:1500], p]
    assert ac.wauc(y, p, w) == pytest.approx(roc_auc_score(yd, pd_), abs=1e-10)

def test_brier_and_curve():
    y = np.array([0, 1, 1, 0]); p = np.array([.1, .9, .6, .4])
    assert ac.brier(y, p) == pytest.approx(np.mean((p - y) ** 2))
    c = ac.calibration_curve(*_sim(n=5000), bins=10)
    assert c.n.sum() == 5000 and (abs(c.pred - c.obs) < 0.06).all()

def V(*rows):
    return pd.DataFrame(rows, columns=["date", "dx", "dem_type"])

def test_defB_event_nonevent_and_window():
    assert ac.definition_B(T0, V(("2012-01-05", "DEM", "AD")))[0] == "EVENT"
    assert ac.definition_B(T0, V(("2012-01-05", "NONDEM", "")))[0] == "NONEVENT"
    # outside 21-27 months -> indeterminate
    assert ac.definition_B(T0, V(("2011-06-01", "DEM", "AD"), ("2012-06-01", "NONDEM", "")))[0] == "INDETERMINATE"

def test_defB_is_point_status_not_cumulative():
    # demented at 18 months, reverted to non-dementia at 24 months -> NONEVENT under B (would be EVENT under A)
    assert ac.definition_B(T0, V(("2011-07-01", "DEM", "AD"), ("2012-01-01", "NONDEM", "")))[0] == "NONEVENT"

def test_defB_closest_visit_conflict_and_type():
    v = V(("2011-11-01", "NONDEM", ""), ("2012-01-10", "DEM", "AD"))
    assert ac.definition_B(T0, v)[0] == "EVENT"  # closest to H wins
    v = V(("2012-01-01", "DEM", "AD"), ("2012-01-01", "NONDEM", ""), ("2012-03-01", "NONDEM", ""))
    assert ac.definition_B(T0, v)[0] == "NONEVENT"  # conflicting date skipped
    assert ac.definition_B(T0, V(("2012-01-01", "DEM", "OTHER")))[0] == "INDETERMINATE"

def _ic_sim(n=3000, seed=0, informative=False, death_rate=0.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    lam = 0.0008 * np.exp(0.8 * x)  # per-day hazard
    t = rng.exponential(1 / lam)
    true_F = float(np.mean(1 - np.exp(-lam * 730.5)))
    # visits every ~182 days; dropout time depends on x if informative
    drop = rng.exponential(1 / (0.0006 * (np.exp(1.2 * x) if informative else np.ones(n))))
    L = np.zeros(n); R = np.full(n, np.inf)
    for i in range(n):
        v = np.arange(182.6, 3000, 182.6) + rng.normal(0, 15, 1)[0]
        v = v[v < drop[i]]
        hit = v[v >= t[i]]
        if len(hit):
            R[i] = hit[0]; prev = v[v < t[i]]; L[i] = prev[-1] if len(prev) else 0.0
        else:
            L[i] = v[-1] if len(v) else 0.0
    return x, L, R, true_F

def test_turnbull_recovers_F_under_noninformative_censoring():
    x, L, R, F = _ic_sim()
    lo, hi = ac.F_at(*ac.turnbull(L, R), 730.5)
    assert lo - 0.03 <= F <= hi + 0.03

def test_icph_standardised_handles_covariate_dependent_censoring():
    x, L, R, F = _ic_sim(informative=True, seed=2)
    lo, hi = ac.F_at(*ac.turnbull(L, R), 730.5)
    f_ph = ac.icph_standardized_F(L, R, x.reshape(-1, 1), [0, 365.25, 730.5], 730.5)
    assert abs(f_ph - F) < 0.03
    assert abs(f_ph - F) < abs(0.5 * (lo + hi) - F)  # better than the unconditional NPMLE

def test_censoring_bounds_bracket_truth():
    x, L, R, F = _ic_sim(informative=True, seed=3)
    lo, hi = ac.censoring_bounds(L, R, 730.5)
    assert lo <= F <= hi

def test_death_absorb_vs_censor():
    L = np.array([0, 100, 200, 300, 0.0]); R = np.array([150, np.inf, np.inf, 400, np.inf]); dd = np.array([np.inf, 250, np.inf, np.inf, 50])
    La, Ra = ac.intervals_with_death(L, R, dd, "absorb")
    assert np.isinf(La[1]) and np.isinf(La[4]) and La[2] == 200  # only deaths without recorded dementia
    Lc, Rc = ac.intervals_with_death(L, R, dd, "censor")
    assert (Lc == L).all()
    fa = ac.turnbull_with_absorbing(La, Ra, 730.5); fc = ac.F_at(*ac.turnbull(Lc, Rc), 730.5)
    assert fa[1] <= fc[1] + 1e-9  # recorded-dementia risk before death <= net risk
    with pytest.raises(ValueError):
        ac.intervals_with_death(L, R, dd, "drop")

T0 = pd.Timestamp("2010-01-01")
