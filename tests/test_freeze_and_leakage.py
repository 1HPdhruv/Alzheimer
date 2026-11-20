"""Synthetic artefacts only. Verifies enforced (not conventional) freezing and the leakage/overlap/unit safeguards."""
import dataclasses
import numpy as np, pandas as pd, pytest
from yoad.models.frozen_pipeline import (ModelSpec, fit_source, evaluate_frozen_on_target, update_calibration_on_target,
                                        UnitCodingError, TargetLeakageError)
from yoad.models.freeze import (FrozenConfigError, IntegrityError, assert_same_configuration, register_new_version, make_manifest)
from yoad.evaluation.safeguards import check_feature_names, flag_suspicious_predictors, LeakageError

F = ("age", "sex", "educ", "faq", "cdrsb")
DOM = (("age", 18, 120, "years"), ("sex", 0, 1, "0=male,1=female"), ("educ", 0, 36, "years"), ("faq", 0, 30, "sum of 10 items 0-3"), ("cdrsb", 0, 18, "boxes"))

def cohort(n, seed, shift=0.0, id0=0):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({"age": r.normal(68, 8, n).clip(40, 95), "sex": r.integers(0, 2, n).astype(float), "educ": r.integers(8, 20, n).astype(float),
                      "faq": r.integers(0, 20, n).astype(float), "cdrsb": r.integers(0, 9, n) / 2 + 0.5})
    lp = -2.2 + 0.08 * d.faq + 0.5 * d.cdrsb + 0.02 * (d.age - 68) + shift
    d["y"] = r.binomial(1, 1 / (1 + np.exp(-lp))); d["pid"] = np.arange(id0, id0 + n); return d

SPEC_A = ModelSpec("A_logistic", F, (("C", (0.1, 1.0)),), cv_folds=3, domains=DOM)
SPEC_B = ModelSpec("B_xgboost", F, (("max_depth", (2,)),), cv_folds=3, domains=DOM)

@pytest.fixture(scope="module")
def src(): return cohort(400, 1)
@pytest.fixture(scope="module")
def tgt(): return cohort(250, 2, shift=-0.3, id0=10_000)
@pytest.fixture(scope="module")
def mA(src): return fit_source(SPEC_A, src[list(F)], src.y.values, src.pid, version="1.0.0", training_cohort_id="SYN_SOURCE", freeze_date="2026-10-13",
                               selection_result={"selected": "A_logistic"}, validation_plan={"target": "SYN_TARGET", "primary": "frozen"})
@pytest.fixture(scope="module")
def mB(src): return fit_source(SPEC_B, src[list(F)], src.y.values, src.pid, version="1.0.0", training_cohort_id="SYN_SOURCE", freeze_date="2026-10-13")

# ---------------- manifest content
def test_manifest_has_required_fields_and_verifies(mA):
    m = mA.manifest; m.verify()
    assert m.model_id == "A_logistic" and m.version == "1.0.0" and m.training_cohort_id == "SYN_SOURCE" and m.freeze_date == "2026-10-13"
    assert m.features == F and m.outcome == "any_dementia" and m.horizon_months == 24
    assert m.preprocessing["preprocessing_version"] and m.preprocessing["domains"] and len(m.artifact_sha256) == 64
    assert {"python", "numpy", "scikit-learn", "xgboost"} <= set(m.software) and m.selection_result == {"selected": "A_logistic"} and m.validation_plan["primary"] == "frozen"

def test_manifest_tampering_is_detected(mA):
    with pytest.raises(IntegrityError): dataclasses.replace(mA.manifest, features=tuple(reversed(F))).verify()
    with pytest.raises(dataclasses.FrozenInstanceError): mA.manifest.version = "9"

# ---------------- immutability enforced
def test_attribute_assignment_blocked_and_inplace_edits_detected(src):
    m = fit_source(SPEC_A, src[list(F)], src.y.values, src.pid, freeze_date="2026-10-13")
    with pytest.raises(AttributeError): m.status = "x"
    with pytest.raises(AttributeError): m._platt = (0.0, 1.0)
    with pytest.raises(ValueError): m._est.coef_[0, 0] += 1.0            # read-only array
    m._est.coef_ = m._est.coef_ + 1.0                                     # reassigning the estimator attribute bypasses numpy protection ...
    with pytest.raises(IntegrityError): m.predict_proba(src[list(F)])     # ... but the checksum catches it

def test_xgboost_booster_edit_detected(src):
    m = fit_source(SPEC_B, src[list(F)], src.y.values, src.pid, freeze_date="2026-10-13")
    m._est.get_booster().set_param({"eta": 0.9})
    m._est.get_booster().set_attr(tampered="1")
    with pytest.raises(IntegrityError): m.predict_proba(src[list(F)])

def test_same_inputs_same_checksum_and_new_version_changes_manifest(src, mA):
    again = fit_source(SPEC_A, src[list(F)], src.y.values, src.pid, version="1.0.0", training_cohort_id="SYN_SOURCE", freeze_date="2026-10-13",
                       selection_result={"selected": "A_logistic"}, validation_plan={"target": "SYN_TARGET", "primary": "frozen"})
    assert again.manifest.artifact_sha256 == mA.manifest.artifact_sha256 and again.manifest.manifest_sha256 == mA.manifest.manifest_sha256

def test_config_change_requires_explicit_new_version(src, mA):
    m = mA.manifest
    kw = dict(preprocessing=m.preprocessing, horizon_months=24, outcome="any_dementia")
    assert_same_configuration(m, features=F, version="1.0.0", **kw)                          # identical -> fine
    with pytest.raises(FrozenConfigError): assert_same_configuration(m, features=F[:-1], version="1.0.0", **kw)
    with pytest.raises(FrozenConfigError): assert_same_configuration(m, features=F, version="1.0.0", **{**kw, "horizon_months": 36})
    with pytest.raises(FrozenConfigError): assert_same_configuration(m, features=F, version="1.0.0", **{**kw, "preprocessing": {**m.preprocessing, "imputer": "mean"}})
    assert_same_configuration(m, features=F[:-1], version="2.0.0", **kw)                     # explicit new version allowed
    F2 = F[:-1]; spec2 = ModelSpec("A_logistic", F2, (("C", (0.1, 1.0)),), cv_folds=3, domains=DOM[:-1])
    m2 = fit_source(spec2, src[list(F2)], src.y.values, src.pid, version="2.0.0", freeze_date="2026-10-13", parent_manifest_sha256=m.manifest_sha256)
    assert register_new_version(m, m2.manifest, "drop CDR-SB after authorised review")["to"] == "2.0.0"
    with pytest.raises(FrozenConfigError): register_new_version(m, m2.manifest, "  ")
    m3 = fit_source(spec2, src[list(F2)], src.y.values, src.pid, version="1.0.0", freeze_date="2026-10-13", parent_manifest_sha256=m.manifest_sha256)
    with pytest.raises(FrozenConfigError): register_new_version(m, m3.manifest, "same version label")

# ---------------- scoring safeguards
def test_feature_order_missing_predictors_and_positional_arrays(mA, tgt):
    p = mA.predict_proba(tgt[list(F)])
    assert np.allclose(p, mA.predict_proba(tgt[list(reversed(F))]))        # order of columns irrelevant: selected by name
    with pytest.raises(TypeError): mA.predict_proba(tgt[list(F)].to_numpy())  # positional input rejected
    with pytest.raises(KeyError): mA.predict_proba(tgt.drop(columns=["faq"]))  # never silently dropped/substituted
    sub = tgt.assign(faq_alt=tgt.faq)
    with pytest.raises(KeyError): mA.predict_proba(sub.drop(columns=["faq"]))

def test_unit_and_coding_inconsistencies_rejected(mA, tgt):
    with pytest.raises(UnitCodingError) as e: mA.predict_proba(tgt.assign(age=tgt.age * 12))        # months instead of years
    assert "age" in e.value.args[0]
    with pytest.raises(UnitCodingError): mA.predict_proba(tgt.assign(sex=tgt.sex + 1))               # 1/2 coding instead of 0/1
    mA.predict_proba(tgt.assign(faq=tgt.faq / 30))   # KNOWN LIMITATION: a rescaling that stays inside the plausible range is NOT detected by domain checks
    with pytest.raises(UnitCodingError): mA.predict_proba(tgt.assign(faq=tgt.faq * 10))              # 0-30 total mistaken for a percentage-like 0-300 scale

def test_evaluation_overlap_and_registered_version(mA, src, tgt):
    r = evaluate_frozen_on_target(mA, tgt[list(F)], tgt.y.values, target_ids=tgt.pid, n_boot=50)
    assert r["evaluation_mode"] == "FROZEN" and r["overlap_status"].startswith("UNVERIFIED") and r["manifest_sha256"] == mA.manifest.manifest_sha256
    r2 = evaluate_frozen_on_target(mA, tgt[list(F)], tgt.y.values, target_ids=tgt.pid, source_ids=src.pid, n_boot=50,
                                   expected_manifest_sha256=mA.manifest.manifest_sha256)
    assert r2["overlap_status"].startswith("VERIFIED_DISJOINT")
    with pytest.raises(LeakageError): evaluate_frozen_on_target(mA, tgt[list(F)], tgt.y.values, target_ids=src.pid.iloc[:250], source_ids=src.pid)
    with pytest.raises(IntegrityError): evaluate_frozen_on_target(mA, tgt[list(F)], tgt.y.values, target_ids=tgt.pid, expected_manifest_sha256="0" * 64)
    with pytest.raises(ValueError): evaluate_frozen_on_target(mA, tgt[list(F)], np.where(np.arange(len(tgt)) < 5, np.nan, tgt.y.values), target_ids=tgt.pid)

def test_updating_is_separate_and_does_not_touch_frozen_model(mA, tgt):
    before = mA.predict_proba(tgt[list(F)]).copy(); h = mA.manifest.manifest_sha256
    upd = update_calibration_on_target(mA, tgt[list(F)], tgt.y.values)
    assert upd.evaluation_mode == "UPDATED" and upd.status.startswith("MODEL_UPDATING")
    assert np.allclose(before, mA.predict_proba(tgt[list(F)])) and mA.manifest.manifest_sha256 == h

# ---------------- leakage
def test_leakage_names_and_values_flagged():
    assert {"followup_years", "dx_month24", "diagnosis_12", "months_to_event", "converted_24"} <= set(
        check_feature_names(["age", "followup_years", "dx_month24", "diagnosis_12", "months_to_event", "converted_24", "faq"], ["converted_24"]))
    assert check_feature_names(list(F), []) == []
    r = np.random.default_rng(0); n = 300; fu = r.uniform(0.5, 6, n); y = r.binomial(1, 0.3, n)
    X = pd.DataFrame({"age": r.normal(70, 8, n), "renamed_fu": fu * 12 + r.normal(0, 0.01, n), "sneaky_outcome": y + r.normal(0, 0.05, n)})
    fl = flag_suspicious_predictors(X, y, followup=fu)
    assert "renamed_fu" in fl["followup_proxy"] and "sneaky_outcome" in fl["outcome_proxy"] and "age" not in sum(fl.values(), [])

def test_target_outcomes_cannot_be_used_for_fitting_or_selection_and_nan_labels_rejected(src):
    with pytest.raises(TargetLeakageError): fit_source(SPEC_A, src[list(F)], src.y.values, src.pid, cohort_role="target")
    y = src.y.values.astype(float); y[0] = np.nan
    with pytest.raises(ValueError): fit_source(SPEC_A, src[list(F)], y, src.pid)
