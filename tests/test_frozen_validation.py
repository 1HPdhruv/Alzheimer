"""All data here are synthetic; assertions test pipeline behaviour, not any empirical finding."""
import numpy as np, pandas as pd, pytest
from yoad.evaluation.frozen_validation import (FrozenModelSpec, HarmonizationError, check_harmonization,
    apply_locked_preprocessing, evaluate_frozen, run_record)
from yoad.evaluation.safeguards import check_feature_names, check_feature_dates, assert_disjoint, LeakageError
from yoad.data.cohort import derive_outcomes, derive_outcomes_label

SPEC = FrozenModelSpec("synthetic_lr", "0.1", 24, ("mmse", "age"), {"standardize": {"mmse": (27.0, 2.0), "age": (60.0, 5.0)}})

def test_harmonization_fails_loudly_on_missing_uncertain_unmapped():
    with pytest.raises(HarmonizationError) as e:
        check_harmonization(SPEC, ["age"], {"age": "supported", "mmse": "uncertain"})
    d = e.value.args[0]
    assert d["missing_in_target"] == ["mmse"] and d["incompatible_or_uncertain"] == ["mmse"]
    assert check_harmonization(SPEC, ["age", "mmse"], {"age": "supported", "mmse": "transform"})["needs_transform"] == ["mmse"]
    with pytest.raises(HarmonizationError):
        check_harmonization(SPEC, ["age", "mmse"], {"age": "supported"})   # unmapped -> not silently accepted

def test_locked_preprocessing_uses_source_constants_not_target_statistics():
    X = pd.DataFrame({"mmse": [29.0, 25.0], "age": [55.0, 65.0]})
    out = apply_locked_preprocessing(SPEC, X)
    assert out["mmse"].tolist() == [1.0, -1.0] and out["age"].tolist() == [-1.0, 1.0]
    out2 = apply_locked_preprocessing(SPEC, X + 100)   # shifted target distribution must NOT be re-centred
    assert out2["mmse"].mean() > 10

def test_metrics_recover_known_calibration_on_synthetic_data():
    rng = np.random.default_rng(0); n = 6000
    lp = rng.normal(-0.5, 1.0, n); p = 1 / (1 + np.exp(-lp)); y = rng.binomial(1, p)
    r = evaluate_frozen(y, p, n_boot=200)
    assert r["status"] == "ok" and 0.60 < r["auroc"] < 0.80
    assert abs(r["cal_slope"] - 1) < 0.1 and abs(r["cal_in_large"]) < 0.1 and 0.9 < r["oe_ratio"] < 1.1
    assert r["auroc_ci"][0] < r["auroc"] < r["auroc_ci"][1]
    p_over = 1 / (1 + np.exp(-(2.0 * lp)))        # overconfident predictor -> slope < 1
    assert evaluate_frozen(y, p_over, n_boot=50)["cal_slope"] < 0.7

def test_insufficient_sample_is_reported_not_silently_scored():
    y = np.array([1] * 5 + [0] * 100); p = np.full(105, 0.1)
    r = evaluate_frozen(y, p)
    assert r["status"] == "insufficient" and "auroc" not in r
    with pytest.raises(ValueError):
        evaluate_frozen(np.array([1.0, np.nan, 0.0]), np.array([.5, .5, .5]))   # NaN label is never treated as 0

def test_fixed_horizon_label_missing_for_short_followup_and_leakage_guards():
    base = pd.DataFrame({"participant_id": [1, 2, 3], "baseline_date": pd.to_datetime(["2020-01-01"] * 3)})
    long = pd.DataFrame({"participant_id": [1, 1, 2, 3, 3],
        "visit_date": pd.to_datetime(["2021-01-01", "2021-11-01", "2020-07-01", "2021-06-01", "2022-07-01"]),
        "diagnosis": ["MCI", "AD", "MCI", "MCI", "MCI"]})
    out = derive_outcomes(long, base)
    lab = derive_outcomes_label(out, 2.0)
    by = dict(zip(out["participant_id"], lab))
    assert by[1] == 1 and np.isnan(by[2]) and by[3] == 0     # participant 2: 0.5 y follow-up -> unlabelled
    assert check_feature_names(["mmse", "converted_24"], ["converted_24"]) == ["converted_24"]
    fd = pd.DataFrame({"participant_id": [1], "visit_date": pd.to_datetime(["2021-03-01"])})
    assert len(check_feature_dates(fd, base)) == 1             # post-baseline predictor flagged
    with pytest.raises(LeakageError): assert_disjoint([1, 2], [2, 3])

def test_run_record_is_labelled_synthetic_and_fingerprint_changes_with_spec():
    r = run_record(SPEC, {"seed": 0}, {"status": "ok"}, synthetic=True)
    assert r["SYNTHETIC"] is True and r["spec_fingerprint"] == SPEC.fingerprint()
    assert FrozenModelSpec("synthetic_lr", "0.2", 24, SPEC.features, SPEC.preprocessing).fingerprint() != SPEC.fingerprint()
