import math, pytest
from yoad.evaluation.sample_size import *

def test_se_ln_oe_closed_form():
    assert se_ln_oe(1000, 0.1) == pytest.approx(math.sqrt(0.9/100))
    with pytest.raises(ValueError): se_ln_oe(0, 0.1)

def test_hanley_mcneil_known_properties():
    # SE shrinks with sample size and equals the textbook value for AUC=0.5 with 100/100 (approx 0.0408)
    assert hanley_mcneil_se(0.5, 100, 100) == pytest.approx(0.0408, abs=1e-3)
    assert hanley_mcneil_se(0.8, 400, 400) < hanley_mcneil_se(0.8, 100, 100)
    with pytest.raises(ValueError): hanley_mcneil_se(0.4, 10, 10)

def test_epv_and_max_r2():
    assert events_per_variable(60, 6) == 10
    assert max_r2_cs(0.5) == pytest.approx(0.75)      # 1-(0.5)^2
    with pytest.raises(ValueError): events_per_variable(10, 0)

def test_riley_shrinkage_monotone():
    n_small = riley_shrinkage_n(10, 0.2); n_big = riley_shrinkage_n(20, 0.2)
    assert n_big == pytest.approx(2*n_small) and riley_shrinkage_n(10, 0.4) < n_small

def test_simulation_matches_analytic_and_is_reproducible():
    a = simulate_validation_precision(800, 0.25, 1.5, n_sim=150, seed=3)
    b = simulate_validation_precision(800, 0.25, 1.5, n_sim=150, seed=3)
    assert a == b
    assert abs(a["mean_events"] - 200) < 15                       # prevalence honoured
    # empirical SD of slope close to model-based SE (well-calibrated by construction)
    assert a["slope_empirical_sd"]*1.96 == pytest.approx(a["slope_ci_halfwidth"], rel=0.25)
    # Hanley-McNeil half-width agrees with simulated AUC spread
    hm = 1.96*hanley_mcneil_se(a["mean_auc"], int(a["mean_events"]), 800-int(a["mean_events"]))
    assert a["auc_ci_halfwidth"] == pytest.approx(hm, rel=0.3)


def test_newcombe_matches_hanley_mcneil_and_simulation():
    n, phi, c = 800, 0.25, 0.80
    hm = hanley_mcneil_se(c, int(n * phi), int(n * (1 - phi)))
    nc = newcombe_se(c, n, phi)
    assert nc == pytest.approx(hm, rel=0.10)          # two published approximations agree closely
    sim = simulate_validation_precision(800, 0.25, 1.5, n_sim=150, seed=3)
    assert 1.96 * newcombe_se(sim["mean_auc"], 800, 0.25) == pytest.approx(sim["auc_ci_halfwidth"], rel=0.3)
    with pytest.raises(ValueError): newcombe_se(0.4, 100, 0.2)
