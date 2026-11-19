import numpy as np, pandas as pd, pytest
from yoad.data.nacc_audit import AuditConfig, run_audit, clean_onset, assemble_visit_date, derive_outcomes, select_baseline

def mk():
    rows = []
    def p(pid, age, onset, seq):  # seq: list of (days_from_start, status, ad_etiology)
        for i, (d, s, ad) in enumerate(seq, 1):
            rows.append(dict(NACCID=pid, NACCVNUM=i, VISIT_DATE=pd.Timestamp("2018-01-01") + pd.Timedelta(days=d), NACCUDSD=s,
                             NACCALZD=ad, NACCAGEB=age, NACCCOGAGE=onset, MMSE=26))
    p("A", 60, 58, [(0,3,1),(365,3,1),(730,4,1)])            # converts to AD within 2y, onset<65
    p("B", 62, 61, [(0,3,0),(365,3,0),(730,3,0)])            # observed to 2y, no event -> non-event
    p("C", 63, 999, [(0,3,0),(365,3,0)])                      # onset unknown; censored (1y)
    p("D", 70, 66, [(0,3,1),(400,4,0),(800,4,0)])             # any-dementia event, not AD; onset>=65
    p("E", 61, 67, [(0,3,1),(365,3,1)])                       # onset 67 > age 61 impossible -> flagged
    p("F", 68, 64, [(0,3,1),(365,4,1),(730,3,1)])             # reversion flag; onset<65 though age>=65 (proxy miss)
    p("G", 66, 70, [(0,1,8),(365,3,1),(730,4,1)])             # normal at initial visit: excluded under initial_visit_mci
    rows.append(dict(NACCID="A", NACCVNUM=3, VISIT_DATE=pd.Timestamp("2018-01-01")+pd.Timedelta(days=730), NACCUDSD=4, NACCALZD=1, NACCAGEB=60, NACCCOGAGE=58, MMSE=26))  # duplicate row
    return pd.DataFrame(rows)

def test_audit_counts_and_rules():
    r = run_audit(mk(), AuditConfig(predictor_cols=("MMSE",)))
    assert r["duplicate_same_day_rows_dropped"] == 1
    assert r["n_mci_baseline"] == 6                               # G excluded (normal at initial visit)
    o = r["onset_baseline_visit"]
    assert o["n_missing"] == 1 and o["n_onset_lt_cutoff"] == 3 and o["n_onset_ge_cutoff"] == 2
    assert o["n_onset_after_baseline_age_flagged"] == 1           # E
    s = r["strata"]["by_baseline_visit_onset"]["onset_lt_cutoff"]
    assert s["n"] == 3 and s["any_events"] == 2 and s["ad_events"] == 2 and s["any_nonevents_observed_to_horizon"] == 1
    assert r["followup"]["n_reversion_flagged"] == 1
    assert r["eligible_for_frozen_model_validation"]["onset_lt_cutoff_any_evaluable_complete_predictors"] == 3

def test_short_followup_is_never_a_nonevent():
    v = mk(); base = select_baseline(v.drop_duplicates(["NACCID","VISIT_DATE"]), AuditConfig())
    oc = derive_outcomes(v.drop_duplicates(["NACCID","VISIT_DATE"]), base, AuditConfig()).set_index("NACCID")
    assert pd.isna(oc.loc["C","label_any"])                      # 1 year of follow-up only
    assert oc.loc["D","label_ad"] != 1 and oc.loc["D","label_any"] == 1   # any-dementia vs AD endpoints differ

def test_missing_codes_and_masking_and_schema():
    assert clean_onset(pd.Series([58, 999, 888, -4, None])).isna().sum() == 4
    r = run_audit(mk(), AuditConfig(min_cell=5))
    assert r["n_mci_baseline"] == 6 and r["strata"]["by_baseline_visit_onset"]["onset_lt_cutoff"]["n"] == "<5"
    with pytest.raises(KeyError): run_audit(mk().drop(columns=["NACCUDSD"]), AuditConfig())

def test_alt_baseline_rule_and_date_assembly():
    r = run_audit(mk(), AuditConfig(baseline_rule="first_mci_visit"))
    assert r["n_mci_baseline"] == 7                               # G enters at first MCI visit
    d = assemble_visit_date(pd.DataFrame({"VISITYR":[2020],"VISITMO":[3],"VISITDAY":[15]}))
    assert d.iloc[0] == pd.Timestamp("2020-03-15")
