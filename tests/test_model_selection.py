"""Synthetic data only: tests of the mathematical behaviour of the prespecified selection rules."""
import numpy as np, pandas as pd, pytest
from yoad.models.model_selection import SelectionRule, paired_replacement_rule, one_se_rule, select_model
from yoad.models.frozen_pipeline import ModelSpec, TargetLeakageError

R = SelectionRule(repeats=10)

def losses(n, base_sd, diff, diff_sd, repeats=10, seed=0, rep_noise=0.0):
    rng = np.random.default_rng(seed)
    a = np.clip(rng.normal(0.2, base_sd, n), 0, 1)[None, :] + rng.normal(0, rep_noise, (repeats, n))
    d = rng.normal(diff, diff_sd, n)[None, :] + rng.normal(0, rep_noise, (repeats, n))
    return a, a - d      # loss_A, loss_B  (d>0 -> B better)

def test_challenger_selected_only_with_clear_stable_paired_improvement():
    la, lb = losses(400, 0.05, 0.01, 0.02)
    r = paired_replacement_rule(la, lb, R)
    assert r["selected"] == "B_xgboost" and r["status"] == "CHALLENGER_SELECTED" and r["z"] > 1

def test_equal_models_keep_default_and_report_no_stable_winner():
    la, lb = losses(400, 0.05, 0.0, 0.02, seed=1)
    r = paired_replacement_rule(la, lb, R)
    assert r["selected"] == "A_logistic" and r["status"] == "NO_STABLE_WINNER_DEFAULT_RETAINED" and r["uncertainty_reported"]
    assert r["ci95_diff"][0] < 0 < r["ci95_diff"][1]

def test_clearly_better_default_is_supported():
    la, lb = losses(400, 0.05, -0.01, 0.02, seed=2)
    assert paired_replacement_rule(la, lb, R)["status"] == "DEFAULT_SUPPORTED"

def test_instability_across_repeats_blocks_replacement():
    n = 300; rng = np.random.default_rng(3); la = np.full((10, n), 0.2) + rng.normal(0, 0.02, (10, n))
    shift = np.array([0.02] * 6 + [-0.012] * 4)[:, None]          # B better on average but wins only 6/10 repeats
    lb = la - shift
    r = paired_replacement_rule(la, lb, R)
    assert r["mean_diff_A_minus_B"] > 0 and r["stability_B_better"] < 0.8 and r["selected"] == "A_logistic"

def test_paired_replacement_and_conventional_one_se_are_different_rules():
    n = 300; rng = np.random.default_rng(4)
    base = np.clip(rng.normal(0.2, 0.3, n), 0, 1); la = np.tile(base, (10, 1)); lb = la - rng.normal(0.01, 0.01, n)[None, :]
    paired = paired_replacement_rule(la, lb, R)
    se = lambda x: float(x.mean(axis=0).std(ddof=1) / np.sqrt(n))
    one = one_se_rule({"A_logistic": float(la.mean()), "B_xgboost": float(lb.mean())}, {"A_logistic": se(la), "B_xgboost": se(lb)}, ["A_logistic", "B_xgboost"])
    assert paired["selected"] == "B_xgboost" and one["selected"] == "A_logistic"      # same data, different decision

def test_select_model_end_to_end_synthetic_and_source_only_guard():
    rng = np.random.default_rng(5); n = 220
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "x3": rng.normal(size=n)})
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.9 * X.x1 - 0.7 * X.x2 - 0.3)))); ids = pd.Series(np.arange(n))
    f = ("x1", "x2", "x3")
    A = ModelSpec("A_logistic", f, (("C", (0.1, 1.0)),), cv_folds=3); B = ModelSpec("B_xgboost", f, (("max_depth", (2,)),), cv_folds=3)
    rule = SelectionRule(repeats=3, outer_folds=3)
    res = select_model(A, B, X, y, ids, rule)
    assert res["selected"] in ("A_logistic", "B_xgboost") and res["n_participants"] == n and "one_se_comparison" in res
    assert res == select_model(A, B, X, y, ids, rule)                                   # reproducible
    with pytest.raises(TargetLeakageError): select_model(A, B, X, y, ids, rule, cohort_role="target")
    with pytest.raises(ValueError): select_model(A, ModelSpec("B_xgboost", ("x1", "x2"), (("max_depth", (2,)),)), X, y, ids, rule)
