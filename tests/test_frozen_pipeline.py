"""Synthetic ADNI-like source / NACC-like target. Nothing here is an empirical finding."""
import inspect
import numpy as np, pandas as pd, pytest
from yoad.models.frozen_pipeline import ModelSpec, fit_source, update_calibration_on_target, FrozenModel
from yoad.evaluation.frozen_validation import evaluate_frozen

FEATS = ("age", "mmse", "faq", "cdrsb")

def make(n_part, visits, shift=0.0, seed=0, miss=0.05):
    rng = np.random.default_rng(seed); rows = []
    for pid in range(n_part):
        base = rng.normal(0, 1, 4)
        for v in range(visits):
            x = base + rng.normal(0, 0.2, 4)
            lp = -0.6 + 0.9 * x[2] + 0.7 * x[3] - 0.6 * x[1] + 0.2 * x[0] + shift
            y = rng.binomial(1, 1 / (1 + np.exp(-lp)))
            rows.append(dict(pid=pid, age=60 + 8 * x[0], mmse=27 + 2 * x[1], faq=5 + 4 * x[2], cdrsb=2 + 1.5 * x[3], y=y))
    d = pd.DataFrame(rows)
    for c in FEATS: d.loc[rng.random(len(d)) < miss, c] = np.nan
    return d

SPEC_A = ModelSpec("A_logistic", FEATS, (("C", (0.01, 0.1, 1.0)),), cv_folds=4)
SPEC_B = ModelSpec("B_xgboost", FEATS, (("max_depth", (2, 3)), ("learning_rate", (0.05, 0.2))), cv_folds=3)

@pytest.fixture(scope="module")
def data():
    return make(500, 2, seed=1), make(300, 1, shift=-0.5, seed=2)

def test_fit_source_signature_has_no_target_argument():
    params = inspect.signature(fit_source).parameters
    assert list(params)[:4] == ["spec", "X_src", "y_src", "participant_ids"]
    assert not any("target" in k for k in params) and params["cohort_role"].default == "source"

@pytest.mark.parametrize("spec", [SPEC_A, SPEC_B])
def test_frozen_model_is_unchanged_by_scoring_target_and_is_deterministic(data, spec):
    src, tgt = data
    m1 = fit_source(spec, src, src.y.values, src.pid); fp = m1.fingerprint()
    p1 = m1.predict_proba(tgt)
    _ = m1.predict_proba(tgt)                       # repeated scoring
    assert m1.fingerprint() == fp and m1.status == "FROZEN_SOURCE_ONLY"
    assert not any(hasattr(m1, a) for a in ("fit", "refit", "partial_fit"))
    m2 = fit_source(spec, src, src.y.values, src.pid)
    assert m2.fingerprint() == fp and np.allclose(m2.predict_proba(tgt), p1)
    r = evaluate_frozen(tgt.y.values, p1, n_boot=100)
    assert r["status"] == "ok" and r["auroc"] > 0.6

def test_covariate_shift_shows_miscalibration_but_frozen_scores_do_not_move(data):
    src, tgt = data
    m = fit_source(SPEC_A, src, src.y.values, src.pid)
    r = evaluate_frozen(tgt.y.values, m.predict_proba(tgt), n_boot=100)
    assert r["oe_ratio"] < 0.95                      # target prevalence is lower (shift=-0.5): over-prediction visible
    before = m.predict_proba(tgt).copy()
    upd = update_calibration_on_target(m, tgt, tgt.y.values)
    assert upd.status == "MODEL_UPDATING_NOT_FROZEN_VALIDATION"
    assert np.allclose(m.predict_proba(tgt), before)  # frozen model untouched
    ru = evaluate_frozen(tgt.y.values, upd.predict_proba(m, tgt), n_boot=100)
    assert abs(ru["oe_ratio"] - 1) < abs(r["oe_ratio"] - 1)   # updating improves calibration (reported separately)

def test_missing_target_feature_fails_loudly_and_nan_labels_rejected(data):
    src, tgt = data
    m = fit_source(SPEC_A, src, src.y.values, src.pid)
    with pytest.raises(KeyError): m.predict_proba(tgt.drop(columns=["faq"]))
    with pytest.raises(ValueError): fit_source(SPEC_A, src, np.where(np.arange(len(src)) == 0, np.nan, src.y.values), src.pid)

def test_grouped_cv_never_splits_participants(data):
    from yoad.evaluation.safeguards import group_kfold
    src, _ = data
    for tr, te in group_kfold(src.pid.reset_index(drop=True), src.y, k=4, seed=0):
        assert not set(src.pid.iloc[tr]) & set(src.pid.iloc[te])
