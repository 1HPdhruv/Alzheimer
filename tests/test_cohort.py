import numpy as np, pandas as pd, pytest
from yoad.data.cohort import *

def test_normalize_unmapped_is_na_not_guessed():
    s = pd.Series([1,2,9]); out = normalize_diagnosis(s, {1:"CN",2:"MCI"}); assert pd.isna(out[2])

def test_normalize_rejects_invalid_label():
    with pytest.raises(ValueError): normalize_diagnosis(pd.Series([1]), {1:"WEIRD"})

def test_transitions(long_df):
    r = validate_transitions(long_df)
    assert r["missing_date_rows"]==1 and r["conflicting_same_day_dx"]==1
    assert r["reversion_participants"]==["P5"]
    assert r["transition_patterns"]["MCI>AD"]>=1

def test_baseline_and_outcomes(long_df):
    base = define_baseline(long_df)
    assert "P7" not in set(base.participant_id) and "P9" not in set(base.participant_id)
    o = derive_outcomes(long_df, base, horizon_years=2, min_followup_years=1).set_index("participant_id")
    assert o.status["P1"]=="AD_CONVERTER" and o.status["P4"]=="OTHER_DEM_CONVERTER"
    assert o.status["P2"]=="STABLE_OBSERVED" and o.status["P3"]=="CENSORED_INSUFFICIENT"
    assert o.status["P6"]=="NO_FOLLOWUP"
    lab = o["label_ad_2y"]
    assert lab["P1"]==1 and lab["P2"]==0
    # censored / competing / no follow-up must NOT be negatives
    assert pd.isna(lab["P3"]) and pd.isna(lab["P4"]) and pd.isna(lab["P6"])
    assert o.event_ad["P4"]==0 and o.censored_for_ad["P4"]==1

def test_short_followup_never_negative(long_df):
    base = define_baseline(long_df)
    o = derive_outcomes(long_df, base, horizon_years=5).set_index("participant_id")
    assert pd.isna(o["label_ad_5y"]["P2"])  # 3y follow-up < 5y horizon

def test_followup_window_comparison(long_df):
    o = derive_outcomes(long_df, define_baseline(long_df))
    t = compare_followup_windows(o, (1,2,5)).set_index("horizon_years")
    assert t.n_labelled[1] >= t.n_labelled[5]

def test_baseline_consistency(long_df):
    base = define_baseline(long_df)
    declared = base.copy(); declared.loc[declared.participant_id=="P1","baseline_date"]=pd.Timestamp("2019-01-01")
    declared = declared[declared.participant_id!="P2"]
    r = check_baseline_consistency(base, long_df, declared)
    assert r["baseline_date_mismatch"]==1 and r["missing_in_declared"]==1 and r["baseline_visit_not_mci"]==0

def test_exclusion_guard():
    assert exclusion_report(100,95,"x")["excluded"]==5
    with pytest.raises(ExclusionError): exclusion_report(100,50,"big")
    assert exclusion_report(100,50,"big",allow_large=True)["fraction"]==0.5

def test_age_eligibility_not_verified_from_enrollment():
    df = pd.DataFrame({"enr":[60,70,np.nan],"ons":[np.nan,np.nan,np.nan]})
    r = classify_age_eligibility(df,"ons","enr")
    assert r.age_class.tolist()==["APPROX_ENROLLMENT_AGE_LT_CUTOFF","ENROLLMENT_AGE_GE_CUTOFF","UNKNOWN_AGE"]
    assert not r.age_class.str.contains("VERIFIED").any()
    df2 = pd.DataFrame({"enr":[66,60],"ons":[63,70]}); r2 = classify_age_eligibility(df2,"ons","enr")
    assert r2.age_class.tolist()==["ONSET_VERIFIED_LT_CUTOFF","ONSET_GE_CUTOFF"]
    r3 = classify_age_eligibility(pd.DataFrame({"enr":[60],"ons":[61]}),"ons","enr"); assert r3.onset_after_enrollment_flag[0]


def test_resolve_onset_age_policies_and_instability():
    v = pd.DataFrame({"participant_id": ["a","a","a","b","c","c"],
                      "visit_date": pd.to_datetime(["2020-01-01","2021-01-01","2022-01-01","2020-01-01","2020-01-01","2021-01-01"]),
                      "DECAGE": [None, 60, 64, 70, None, None]})
    f = resolve_onset_age(v, "DECAGE", "first_visit").set_index("participant_id")
    l = resolve_onset_age(v, "DECAGE", "latest").set_index("participant_id")
    assert f.onset_age["a"] == 60 and l.onset_age["a"] == 64          # first *non-missing*, not first row
    assert f.onset_unstable["a"] and not f.onset_unstable["b"]        # range 4y > 2y tolerance
    assert f.onset_missing["c"] and len(f) == 3                       # missing kept, never dropped
    with pytest.raises(ValueError): resolve_onset_age(v, "DECAGE", "bogus")


def test_onset_vs_age_definition_asymmetry_and_flags():
    df = pd.DataFrame({"onset": [58, 63, 64, 66, 70, None, 67], "age": [60, 66, 64, 68, 72, 61, 66]})
    r = onset_vs_age_definition(df, "onset", "age")
    assert r["n_missing_onset"] == 1                       # missing kept, not replaced by age
    assert r["n_onset_after_baseline_flagged"] == 1        # onset 67 > age 66 impossible
    assert r["onset_lt_cutoff"] == 3 and r["age_lt_cutoff"] == 2
    assert r["onset_lt_but_age_ge (missed by proxy)"] == 1  # onset 63, age 66
    assert r["age_lt_but_onset_ge (must be 0 if consistent)"] == 0
    assert r["proxy_sensitivity_for_onset_lt"] == pytest.approx(2 / 3)
