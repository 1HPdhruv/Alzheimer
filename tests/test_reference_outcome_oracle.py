"""Independent known-answer tests for the reference outcome. Each expected state was derived by hand from the written rules in
research/reference_outcome_specification.md (v1.2; the oracle), not from running the code. Synthetic timelines only.

v1.1 -> v1.2 changes to expected values (recorded, not silently overwritten):
  * reason "nondementia_observed_near_or_after_H" renamed "nondementia_observed_in_window";
  * reason "only_uncertain_diagnoses_near_H" renamed "uncertain_diagnosis_in_window";
  * "stable observed at 36 mo, missing 24-mo visit": NONEVENT (v1.1, no upper bound) -> INDETERMINATE nondementia_only_beyond_window
    (v1.2, u = 90 d); the v1.1 result is retained as the u_days=None sensitivity case;
  * "same-date conflict before H, nothing else": insufficient_followup -> conflicting_diagnoses_on_or_before_H (new precedence P3).

v1.2 -> v1.3 (Prompt 20) changes, recorded here: new reason codes `dementia_type_uncertain` and `died_before_ascertainment`; same-date
dementia records with different or missing etiology are UNKNOWN type (v1.2: 'AD' won by alphabetical order, a missing type was AD);
conflicts are checked against the event cut-off H+e; earliest dated cause among death/withdrawal/cut-off; uncertain diagnoses after a
first dementia no longer count as "uncertain in window". The 51 v1.2 expectations are unchanged; the divergence test for same-date
conflicts now documents AGREEMENT after endpoint.py was aligned (old behaviour: event).
"""
import numpy as np, pandas as pd, pytest
from yoad.data.reference_outcome import reference_state, EVENT, NONEVENT, INDET, REASONS, REASON_PRECEDENCE
from yoad.data.endpoint import label_endpoint

T0 = "2020-01-01"          # H = 2022-01-01 ; H-90 d = 2021-10-03 ; H+90 d = 2022-04-01 ; H-180 d = 2021-07-05 ; H+180 d = 2022-06-30
def V(*rows):
    out = []
    for r in rows:
        d = {"date": r[0], "dx": r[1]}
        if len(r) > 2: d["dem_type"] = r[2]
        out.append(d)
    df = pd.DataFrame(out, columns=["date", "dx"] + (["dem_type"] if any(len(r) > 2 for r in rows) else []))
    if "dem_type" in df: df["dem_type"] = df["dem_type"].fillna("AD")
    return df

ORACLE = [  # (name, visits, kwargs, expected_state, expected_reason)
 # --- carried from v1.1 (expected values updated only where listed in the module docstring)
 ("dementia at 18 mo", V(("2021-01-01", "NONDEM"), ("2021-07-01", "DEM")), {}, EVENT, "dementia_on_or_before_H"),
 ("dementia exactly at H", V(("2022-01-01", "DEM")), {}, EVENT, "dementia_on_or_before_H"),
 ("late converter observed MCI at 24 mo", V(("2022-01-05", "NONDEM"), ("2022-07-01", "DEM")), {}, NONEVENT, "nondementia_observed_in_window"),
 ("late converter, last MCI at 21.5 mo within tolerance", V(("2021-10-15", "NONDEM"), ("2022-03-01", "DEM")), {}, NONEVENT, "nondementia_observed_in_window"),
 ("straddle: last MCI at 12 mo, dementia at 30 mo", V(("2021-01-01", "NONDEM"), ("2022-07-01", "DEM")), {}, INDET, "straddles_H"),
 ("straddle under strict tolerance", V(("2021-10-15", "NONDEM"), ("2022-03-01", "DEM")), {"w_days": 0}, INDET, "straddles_H"),
 ("stable but follow-up ends at 12 mo", V(("2021-01-01", "NONDEM")), {}, INDET, "insufficient_followup"),
 ("stable observed at 36 mo, missing 24-mo visit", V(("2021-01-01", "NONDEM"), ("2023-01-01", "NONDEM")), {}, INDET, "nondementia_only_beyond_window"),
 ("reversion after dementia (primary)", V(("2021-01-01", "DEM"), ("2022-01-01", "NONDEM")), {}, EVENT, "dementia_on_or_before_H"),
 ("reversion after dementia (confirm=True)", V(("2021-01-01", "DEM"), ("2022-01-01", "NONDEM")), {"confirm": True}, INDET, "unconfirmed_dementia_reverted"),
 ("non-AD dementia for AD endpoint", V(("2021-06-01", "DEM", "OTHER")), {}, INDET, "competing_non_ad_dementia"),
 ("non-AD dementia for any-dementia endpoint", V(("2021-06-01", "DEM", "OTHER")), {"endpoint": "any"}, EVENT, "dementia_on_or_before_H"),
 ("death before H without dementia", V(("2021-01-01", "NONDEM")), {"death_date": "2021-06-01"}, INDET, "competing_death"),
 ("undated dementia diagnosis only", V((None, "DEM"), ("2021-01-01", "NONDEM")), {}, INDET, "insufficient_followup"),
 ("visit before index ignored", V(("2019-06-01", "DEM"), ("2022-02-01", "NONDEM")), {}, NONEVENT, "nondementia_observed_in_window"),
 ("same-date conflict at 24 mo is not evidence", V(("2021-01-01", "NONDEM"), ("2022-01-02", "DEM"), ("2022-01-02", "NONDEM")), {}, INDET, "uncertain_diagnosis_in_window"),
 ("unknown diagnosis near H", V(("2021-01-01", "NONDEM"), ("2022-01-02", "unknown")), {}, INDET, "uncertain_diagnosis_in_window"),
 ("same-date conflict before H, nothing else", V(("2021-06-01", "DEM"), ("2021-06-01", "NONDEM")), {}, INDET, "conflicting_diagnoses_on_or_before_H"),
 ("exact duplicate rows collapse", V(("2021-07-01", "DEM"), ("2021-07-01", "DEM")), {}, EVENT, "dementia_on_or_before_H"),
 ("no post-index visits", V(), {}, INDET, "insufficient_followup"),
 # --- v1.2: window bounds (primary w = u = 90 d)
 ("non-dementia at H+90 d (upper edge, inclusive)", V(("2022-04-01", "NONDEM")), {}, NONEVENT, "nondementia_observed_in_window"),
 ("non-dementia at H+91 d (just beyond)", V(("2022-04-02", "NONDEM")), {}, INDET, "nondementia_only_beyond_window"),
 ("non-dementia at H-90 d (lower edge, inclusive)", V(("2021-10-03", "NONDEM")), {}, NONEVENT, "nondementia_observed_in_window"),
 ("non-dementia at H-91 d (just before)", V(("2021-10-02", "NONDEM")), {}, INDET, "insufficient_followup"),
 ("u=180: non-dementia at H+150 d qualifies", V(("2022-05-31", "NONDEM")), {"u_days": 180}, NONEVENT, "nondementia_observed_in_window"),
 ("u=None (v1.1 behaviour): 36-mo non-dementia qualifies", V(("2021-01-01", "NONDEM"), ("2023-01-01", "NONDEM")), {"u_days": None}, NONEVENT, "nondementia_observed_in_window"),
 ("w=0: non-dementia 1 day before H does not qualify", V(("2021-12-31", "NONDEM")), {"w_days": 0}, INDET, "insufficient_followup"),
 ("w=0: non-dementia exactly at H qualifies", V(("2022-01-01", "NONDEM")), {"w_days": 0}, NONEVENT, "nondementia_observed_in_window"),
 ("w=180: non-dementia at H-170 d qualifies", V(("2021-07-15", "NONDEM")), {"w_days": 180}, NONEVENT, "nondementia_observed_in_window"),
 ("w=90: same visit at H-170 d does not", V(("2021-07-15", "NONDEM")), {}, INDET, "insufficient_followup"),
 ("beyond-window non-dementia then later dementia", V(("2022-06-01", "NONDEM"), ("2022-12-01", "DEM")), {}, INDET, "nondementia_only_beyond_window"),
 ("non-dementia in window but AFTER a post-H dementia does not qualify", V(("2021-01-01", "NONDEM"), ("2022-01-10", "DEM"), ("2022-02-01", "NONDEM")), {}, INDET, "straddles_H"),
 # --- v1.2: competing events and unknown status are distinct
 ("death after qualifying visit but before H -> competing death", V(("2021-12-01", "NONDEM")), {"death_date": "2021-12-20"}, INDET, "competing_death"),
 ("death after H with qualifying visit -> non-event", V(("2021-12-01", "NONDEM")), {"death_date": "2022-03-01"}, NONEVENT, "nondementia_observed_in_window"),
 ("AD dementia then death before H -> event", V(("2021-06-01", "DEM")), {"death_date": "2021-09-01"}, EVENT, "dementia_on_or_before_H"),
 ("visit after death is discarded", V(("2021-12-01", "NONDEM"), ("2021-09-01", "DEM")), {"death_date": "2021-06-01"}, INDET, "competing_death"),
 ("withdrawal before window, no qualifying visit", V(("2021-01-01", "NONDEM")), {"withdrawal_date": "2021-07-01"}, INDET, "withdrawn_before_ascertainment"),
 ("withdrawal after qualifying visit -> non-event", V(("2021-12-01", "NONDEM")), {"withdrawal_date": "2021-12-15"}, NONEVENT, "nondementia_observed_in_window"),
 ("withdrawal after window end is not the reason", V(("2021-01-01", "NONDEM")), {"withdrawal_date": "2022-06-01"}, INDET, "insufficient_followup"),
 ("data cut-off before window end", V(("2021-01-01", "NONDEM")), {"data_cutoff": "2021-11-01"}, INDET, "administratively_censored"),
 # --- v1.2: precedence between applicable rules
 ("non-AD dementia outranks later death", V(("2021-03-01", "DEM", "OTHER")), {"death_date": "2021-09-01"}, INDET, "competing_non_ad_dementia"),
 ("death outranks conflict on/before H", V(("2021-03-01", "DEM"), ("2021-03-01", "NONDEM")), {"death_date": "2021-09-01"}, INDET, "competing_death"),
 ("conflict on/before H outranks a qualifying visit", V(("2021-03-01", "DEM"), ("2021-03-01", "NONDEM"), ("2021-12-15", "NONDEM")), {}, INDET, "conflicting_diagnoses_on_or_before_H"),
 ("clean dementia outranks an earlier conflict", V(("2021-03-01", "DEM"), ("2021-03-01", "NONDEM"), ("2021-06-01", "DEM")), {}, EVENT, "dementia_on_or_before_H"),
 ("uncertain in window outranks beyond-window non-dementia", V(("2022-02-01", "unknown"), ("2022-08-01", "NONDEM")), {}, INDET, "uncertain_diagnosis_in_window"),
 ("withdrawal outranks administrative censoring", V(("2021-01-01", "NONDEM")), {"withdrawal_date": "2021-07-01", "data_cutoff": "2021-11-01"}, INDET, "withdrawn_before_ascertainment"),
 ("straddle outranks withdrawal", V(("2021-01-01", "NONDEM"), ("2022-07-01", "DEM")), {"withdrawal_date": "2022-08-01"}, INDET, "straddles_H"),
 # --- v1.2: visit-window event sensitivity (e = 28 d)
 ("dementia at H+20 d: not an event in the primary analysis", V(("2021-01-01", "NONDEM"), ("2022-01-21", "DEM")), {}, INDET, "straddles_H"),
 ("dementia at H+20 d: event with event_grace_days=28", V(("2021-01-01", "NONDEM"), ("2022-01-21", "DEM")), {"event_grace_days": 28}, EVENT, "dementia_on_or_before_H"),
 # --- v1.2: calendar months
 ("leap-day index: H = 28 Feb, dementia on H is an event", V(("2022-02-28", "DEM")), {"_t0": "2020-02-29"}, EVENT, "dementia_on_or_before_H"),
 ("leap-day index: dementia on 1 Mar is after H", V(("2022-03-01", "DEM")), {"_t0": "2020-02-29"}, INDET, "straddles_H"),
 # --- v1.3 (Prompt 20): chronology of first dementia, etiology and same-date records
 ("first dementia AD, later non-AD <= H -> event (later diagnoses do not change the primary endpoint)", V(("2021-03-01", "DEM"), ("2021-06-01", "DEM", "OTHER")), {}, EVENT, "dementia_on_or_before_H"),
 ("first dementia non-AD, later AD <= H -> competing (first dementia decides)", V(("2021-03-01", "DEM", "OTHER"), ("2021-06-01", "DEM")), {}, INDET, "competing_non_ad_dementia"),
 ("type_rule any_ad: non-AD first, AD later <= H -> event (sensitivity only)", V(("2021-03-01", "DEM", "OTHER"), ("2021-06-01", "DEM")), {"type_rule": "any_ad"}, EVENT, "dementia_on_or_before_H"),
 ("type_rule any_ad: AD only after H does not rescue", V(("2021-03-01", "DEM", "OTHER"), ("2022-03-01", "DEM")), {"type_rule": "any_ad"}, INDET, "competing_non_ad_dementia"),
 ("same-date AD + non-AD dementia -> type unknown", V(("2021-03-01", "DEM", "AD"), ("2021-03-01", "DEM", "OTHER")), {}, INDET, "dementia_type_uncertain"),
 ("same-date AD + non-AD dementia, any-dementia endpoint -> event", V(("2021-03-01", "DEM", "AD"), ("2021-03-01", "DEM", "OTHER")), {"endpoint": "any"}, EVENT, "dementia_on_or_before_H"),
 ("same-date AD + non-AD, then later clean AD, type_rule any_ad -> event", V(("2021-03-01", "DEM", "AD"), ("2021-03-01", "DEM", "OTHER"), ("2021-05-01", "DEM")), {"type_rule": "any_ad"}, EVENT, "dementia_on_or_before_H"),
 ("explicit UNKNOWN etiology on first dementia -> type uncertain", V(("2021-03-01", "DEM", "UNKNOWN")), {}, INDET, "dementia_type_uncertain"),
 ("dementia + missing-diagnosis record on one date is dementia, AD", V(("2021-03-01", "DEM"), ("2021-03-01", "unknown")), {}, EVENT, "dementia_on_or_before_H"),
 ("same-date DEM(non-AD)+NONDEM is a conflict, not a competing event", V(("2021-03-01", "DEM", "OTHER"), ("2021-03-01", "NONDEM")), {}, INDET, "conflicting_diagnoses_on_or_before_H"),
 ("conflict, then clean AD dementia later <= H -> event", V(("2021-03-01", "DEM"), ("2021-03-01", "NONDEM"), ("2021-09-01", "DEM")), {}, EVENT, "dementia_on_or_before_H"),
 ("conflict, then clean non-AD dementia later <= H -> competing", V(("2021-03-01", "DEM"), ("2021-03-01", "NONDEM"), ("2021-09-01", "DEM", "OTHER")), {}, INDET, "competing_non_ad_dementia"),
 ("reversion then non-AD dementia <= H keeps the first-dementia event", V(("2021-01-01", "DEM"), ("2021-03-01", "NONDEM"), ("2021-06-01", "DEM", "OTHER")), {}, EVENT, "dementia_on_or_before_H"),
 # conflicts relative to the event cut-off (uncertain outcomes never become non-events)
 ("conflict at H+10 d, e=0: later than the cut-off, earlier qualifying non-dementia -> non-event", V(("2021-12-20", "NONDEM"), ("2022-01-11", "DEM"), ("2022-01-11", "NONDEM")), {}, NONEVENT, "nondementia_observed_in_window"),
 ("conflict at H+10 d, e=28: potential event inside cut-off -> indeterminate", V(("2021-12-20", "NONDEM"), ("2022-01-11", "DEM"), ("2022-01-11", "NONDEM")), {"event_grace_days": 28}, INDET, "conflicting_diagnoses_on_or_before_H"),
 ("conflict at H+40 d, e=28: beyond cut-off, qualifying visit kept", V(("2021-12-20", "NONDEM"), ("2022-02-10", "DEM"), ("2022-02-10", "NONDEM")), {"event_grace_days": 28}, NONEVENT, "nondementia_observed_in_window"),
 ("early conflict (month 6) blocks a later qualifying non-dementia", V(("2020-07-01", "DEM"), ("2020-07-01", "NONDEM"), ("2021-12-20", "NONDEM")), {}, INDET, "conflicting_diagnoses_on_or_before_H"),
 ("missing-diagnosis record before the window does not block a qualifying non-dementia", V(("2020-07-01", "unknown"), ("2021-12-20", "NONDEM")), {}, NONEVENT, "nondementia_observed_in_window"),
 # uncertain diagnoses vs first dementia
 ("uncertain diagnosis in window AFTER first dementia is a straddle, not 'uncertain in window'", V(("2021-01-01", "NONDEM"), ("2022-01-30", "DEM"), ("2022-02-10", "unknown")), {}, INDET, "straddles_H"),
 ("uncertain diagnosis in window before first dementia", V(("2021-01-01", "NONDEM"), ("2022-01-30", "unknown"), ("2022-09-01", "DEM")), {}, INDET, "uncertain_diagnosis_in_window"),
 # death / withdrawal / cut-off: distinct codes, earliest dated cause
 ("death in (H, H+u], no qualifying visit -> died_before_ascertainment", V(("2021-01-01", "NONDEM")), {"death_date": "2022-02-01"}, INDET, "died_before_ascertainment"),
 ("death after H+u, no qualifying visit -> insufficient follow-up (death is not the cause)", V(("2021-01-01", "NONDEM")), {"death_date": "2022-06-01"}, INDET, "insufficient_followup"),
 ("death at H exactly is competing", V(("2021-01-01", "NONDEM")), {"death_date": "2022-01-01"}, INDET, "competing_death"),
 ("death after H, qualifying visit before it -> non-event", V(("2021-12-20", "NONDEM")), {"death_date": "2022-02-01"}, NONEVENT, "nondementia_observed_in_window"),
 ("death after H, dementia after H before death -> straddle", V(("2021-01-01", "NONDEM"), ("2022-01-20", "DEM")), {"death_date": "2022-02-01"}, INDET, "straddles_H"),
 ("withdrawal earlier than death -> withdrawn", V(("2021-01-01", "NONDEM")), {"withdrawal_date": "2022-01-20", "death_date": "2022-02-01"}, INDET, "withdrawn_before_ascertainment"),
 ("death earlier than withdrawal -> died", V(("2021-01-01", "NONDEM")), {"withdrawal_date": "2022-02-20", "death_date": "2022-02-01"}, INDET, "died_before_ascertainment"),
 ("death and withdrawal same day -> death", V(("2021-01-01", "NONDEM")), {"withdrawal_date": "2022-02-01", "death_date": "2022-02-01"}, INDET, "died_before_ascertainment"),
 ("cut-off earlier than withdrawal -> administratively censored", V(("2021-01-01", "NONDEM")), {"data_cutoff": "2021-11-01", "withdrawal_date": "2022-02-01"}, INDET, "administratively_censored"),
 ("competing death outranks a conflict even with early withdrawal", V(("2021-03-01", "DEM"), ("2021-03-01", "NONDEM")), {"death_date": "2021-09-01", "withdrawal_date": "2021-04-01"}, INDET, "competing_death"),
]

@pytest.mark.parametrize("name,visits,kw,state,reason", ORACLE, ids=[o[0] for o in ORACLE])
def test_reference_matches_hand_derived_oracle(name, visits, kw, state, reason):
    if visits.empty: visits = pd.DataFrame(columns=["date", "dx"])
    kw = dict(kw); t0 = kw.pop("_t0", T0)
    s, r, _ = reference_state(t0, visits, **kw)
    assert (s, r) == (state, reason)

def test_every_reason_is_declared_and_reachable():
    seen = {o[4] for o in ORACLE}
    assert seen <= set(REASONS)
    assert set(REASONS) == seen, f"untested reasons: {set(REASONS) - seen}"
    assert len(REASONS) == len(set(REASONS)) and REASON_PRECEDENCE[-1][1] == "insufficient_followup"

def test_row_order_does_not_change_the_result():
    for name, visits, kw, state, reason in ORACLE:
        if len(visits) < 2: continue
        kw = dict(kw); t0 = kw.pop("_t0", T0)
        assert reference_state(t0, visits.iloc[::-1].reset_index(drop=True), **kw)[:2] == (state, reason), name

def test_nonevent_requires_assessment_inside_window_for_all_sensitivity_windows():
    """Property: a NONEVENT always has a non-dementia diagnosis inside [H-w, H+u] for w in {0, 90, 180}."""
    rng = np.random.default_rng(0); H = pd.Timestamp("2022-01-01")
    for _ in range(300):
        n = rng.integers(0, 5)
        days = rng.integers(1, 1500, n)
        dx = rng.choice(["DEM", "NONDEM", "unk"], n, p=[.2, .7, .1])
        vis = pd.DataFrame({"date": pd.Timestamp(T0) + pd.to_timedelta(days, "D"), "dx": dx})
        for w in (0, 90, 180):
            s, r, _ = reference_state(T0, vis, w_days=w, u_days=90)
            if s == NONEVENT:
                ok = vis[(vis["dx"] == "NONDEM") & (vis["date"] >= H - pd.Timedelta(days=w)) & (vis["date"] <= H + pd.Timedelta(days=90))]
                assert len(ok) and not ((vis["dx"] == "DEM") & (vis["date"] <= H)).any()
            if s == EVENT:
                assert ((vis["dx"] == "DEM") & (vis["date"] <= H)).any()

def test_flags_reversion_conflict_undated_and_post_death():
    _, _, f = reference_state(T0, V(("2021-01-01", "DEM"), ("2022-01-01", "NONDEM"), (None, "NONDEM")))
    assert f["reverted"] and f["n_undated"] == 1
    _, _, f = reference_state(T0, V(("2022-01-02", "DEM"), ("2022-01-02", "NONDEM")))
    assert f["same_date_conflict"]
    _, _, f = reference_state(T0, V(("2021-12-01", "NONDEM")), death_date="2021-06-01")
    assert f["n_after_death"] == 1

def test_bad_options_rejected():
    with pytest.raises(ValueError): reference_state(T0, V(("2021-01-01", "NONDEM")), endpoint="mci")
    with pytest.raises(ValueError): reference_state(T0, V(("2021-01-01", "NONDEM")), w_days=-1)

V12_DIVERGENT = ("stable observed at 36 mo, missing 24-mo visit",)

def test_cross_check_with_existing_endpoint_code_on_shared_cases():
    """The older endpoint implementation (written earlier, different code path) must agree on cases both specify identically."""
    v11_cases = ORACLE[:20]
    shared = [o for o in v11_cases if not o[2] and o[0] not in ("same-date conflict at 24 mo is not evidence", "unknown diagnosis near H",
              "undated dementia diagnosis only", "no post-index visits", "visit before index ignored", "exact duplicate rows collapse",
              "same-date conflict before H, nothing else") + V12_DIVERGENT]
    assert len(shared) >= 8
    for name, visits, kw, state, reason in shared:
        base = pd.DataFrame({"participant_id": [1], "baseline_date": [pd.Timestamp(T0)]})
        vv = visits.assign(participant_id=1).rename(columns={"date": "visit_date"})
        lab = label_endpoint(base, vv, rule="A", endpoint="ad_dementia").iloc[0]["label"]
        expected = {EVENT: 1.0, NONEVENT: 0.0}.get(state, np.nan)
        assert (np.isnan(lab) and np.isnan(expected)) or lab == expected, name


def test_same_date_conflict_now_agrees_between_labellers():
    """v1.0-1.2: endpoint.py counted a same-date DEM/NONDEM conflict before H as an event while the reference gave INDETERMINATE. Prompt 20
    aligned endpoint.py: both now treat the date as uncertain (unlabelled / INDETERMINATE)."""
    base = pd.DataFrame({"participant_id": [1], "baseline_date": [pd.Timestamp(T0)]})
    vv = V(("2021-06-01", "DEM"), ("2021-06-01", "NONDEM")).assign(participant_id=1).rename(columns={"date": "visit_date"})
    r = label_endpoint(base, vv, rule="A", endpoint="ad_dementia").iloc[0]
    assert np.isnan(r["label"]) and r["reason"] == "conflicting_diagnoses_on_or_before_H"
    assert reference_state(T0, V(("2021-06-01", "DEM"), ("2021-06-01", "NONDEM")))[:2] == (INDET, "conflicting_diagnoses_on_or_before_H")


def test_documented_divergence_upper_bound():
    """v1.2 known difference: endpoint.py Definition A has no upper bound on the confirming non-dementia visit, so a 36-month
    non-dementia diagnosis is a non-event there; the v1.2 reference (u = 90 d) is INDETERMINATE and agrees only with u_days=None."""
    vis = V(("2021-01-01", "NONDEM"), ("2023-01-01", "NONDEM"))
    base = pd.DataFrame({"participant_id": [1], "baseline_date": [pd.Timestamp(T0)]})
    vv = vis.assign(participant_id=1).rename(columns={"date": "visit_date"})
    assert label_endpoint(base, vv, rule="A", endpoint="ad_dementia").iloc[0]["label"] == 0.0
    assert reference_state(T0, vis)[0] == INDET
    assert reference_state(T0, vis, u_days=None)[0] == NONEVENT


# ---------------------------------------------------------------------------------------------------------------------------
# Prompt 20: independent chronological re-statement of the written decision procedure, fuzzed against the implementation.
# It is a second formulation of the SAME specification (not independent evidence about ADNI): it walks the dated records in time order
# with plain Python, never groups with pandas, and never sorts labels.
def slow_oracle(t0, recs, death=None, wd=None, cut=None, w=90, u=90, e=0, endpoint="ad", confirm=False, type_rule="first", has_type=True):
    t0 = pd.Timestamp(t0); H = t0 + pd.DateOffset(months=24); D = pd.Timedelta
    lo = H - D(days=w); hi = None if u is None else H + D(days=u); He = H + D(days=e)
    death, wd, cut = [None if x is None else pd.Timestamp(x) for x in (death, wd, cut)]
    kept = [(pd.Timestamp(d), dx, t) for d, dx, t in recs if d is not None and pd.Timestamp(d) > t0 and (death is None or pd.Timestamp(d) <= death)]
    dates = sorted({d for d, _, _ in kept}); timeline = []
    for d in dates:
        todays = [(dx, t) for dd, dx, t in kept if dd == d]
        has_dem = any(dx == "DEM" for dx, _ in todays); has_non = any(dx == "NONDEM" for dx, _ in todays)
        if has_dem and has_non: timeline.append((d, "CONFLICT", None))
        elif has_dem:
            types = {(t if (has_type and t in ("AD", "OTHER")) else ("AD" if not has_type else "UNKNOWN")) for dx, t in todays if dx == "DEM"}
            timeline.append((d, "DEM", types.pop() if len(types) == 1 else "UNKNOWN"))
        elif has_non: timeline.append((d, "NONDEM", None))
        else: timeline.append((d, "UNK", None))
    firstdem = next(((d, t) for d, c, t in timeline if c == "DEM"), None)
    if firstdem and firstdem[0] <= He:
        any_ad = any(c == "DEM" and t == "AD" and d <= He for d, c, t in timeline)
        if endpoint == "ad" and firstdem[1] != "AD" and not (type_rule == "any_ad" and any_ad):
            return INDET, "competing_non_ad_dementia" if firstdem[1] == "OTHER" else "dementia_type_uncertain"
        if confirm and any(c == "NONDEM" and d > firstdem[0] for d, c, _ in timeline): return INDET, "unconfirmed_dementia_reverted"
        return EVENT, "dementia_on_or_before_H"
    if death is not None and death <= H: return INDET, "competing_death"
    if any(c == "CONFLICT" and d <= He for d, c, _ in timeline): return INDET, "conflicting_diagnoses_on_or_before_H"
    before = [(d, c) for d, c, _ in timeline if firstdem is None or d < firstdem[0]]
    inside = lambda d: d >= lo and (hi is None or d <= hi)
    if any(c == "NONDEM" and inside(d) for d, c in before): return NONEVENT, "nondementia_observed_in_window"
    if any(c in ("UNK", "CONFLICT") and inside(d) for d, c in before): return INDET, "uncertain_diagnosis_in_window"
    if hi is not None and any(c == "NONDEM" and d > hi for d, c in before): return INDET, "nondementia_only_beyond_window"
    if firstdem: return INDET, "straddles_H"
    cands = []
    if death is not None and death > H and (hi is None or death <= hi): cands.append((death, 0, "died_before_ascertainment"))
    if wd is not None and (hi is None or wd <= hi): cands.append((wd, 1, "withdrawn_before_ascertainment"))
    if cut is not None and (hi is None or cut < hi): cands.append((cut, 2, "administratively_censored"))
    return INDET, (sorted(cands)[0][2] if cands else "insufficient_followup")


def _random_history(rng):
    pool = list(pd.Timestamp(T0) + pd.to_timedelta(rng.choice([40, 200, 400, 640, 700, 725, 730, 731, 745, 760, 800, 820, 825, 830, 850, 1000], 16, replace=False), "D"))
    n = int(rng.integers(0, 7)); recs = []
    for _ in range(n):
        d = pool[int(rng.integers(0, len(pool)))] if rng.random() < .9 else None
        dx = rng.choice(["DEM", "NONDEM", "unk"], p=[.3, .55, .15]); t = rng.choice(["AD", "OTHER", None], p=[.6, .25, .15])
        recs.append((d, str(dx), t))
        if rng.random() < .3: recs.append((d, str(rng.choice(["DEM", "NONDEM"])), rng.choice(["AD", "OTHER", None])))   # same-date second record
    return recs


def _frame(recs, has_type=True):
    df = pd.DataFrame(recs, columns=["date", "dx", "dem_type"]) if recs else pd.DataFrame(columns=["date", "dx", "dem_type"])
    return df if has_type else df.drop(columns="dem_type")


def test_fuzz_implementation_matches_independent_chronological_oracle():
    rng = np.random.default_rng(20)
    opt = lambda c: None if rng.random() < .6 else str(pd.Timestamp(T0) + pd.Timedelta(days=int(rng.choice(c))))
    n_checked = 0
    for _ in range(600):
        recs = _random_history(rng); has_type = bool(rng.random() < .85)
        kw = dict(w=int(rng.choice([0, 90, 180])), u=rng.choice([90, 180, None]), e=int(rng.choice([0, 0, 28])), endpoint=str(rng.choice(["ad", "any"])),
                  confirm=bool(rng.random() < .2), type_rule=str(rng.choice(["first", "any_ad"])))
        kw["u"] = None if kw["u"] is None else int(kw["u"])
        death, wd, cut = opt([300, 650, 740, 790, 900]), opt([300, 700, 735, 800]), opt([500, 760, 900])
        got = reference_state(T0, _frame(recs, has_type), w_days=kw["w"], u_days=kw["u"], event_grace_days=kw["e"], endpoint=kw["endpoint"],
                              confirm=kw["confirm"], type_rule=kw["type_rule"], death_date=death, withdrawal_date=wd, data_cutoff=cut)[:2]
        want = slow_oracle(T0, [(None if d is None else d, dx, t) for d, dx, t in recs], death, wd, cut, kw["w"], kw["u"], kw["e"], kw["endpoint"],
                           kw["confirm"], kw["type_rule"], has_type)
        assert got == want, (recs, kw, death, wd, cut, has_type)
        n_checked += 1
    assert n_checked == 600


def test_result_is_invariant_to_record_order_including_same_date_records():
    """No outcome may depend on the order of records, in particular among same-date records of different diagnosis or etiology."""
    rng = np.random.default_rng(21)
    for _ in range(250):
        recs = _random_history(rng)
        if len(recs) < 2: continue
        df = _frame(recs); base = reference_state(T0, df, death_date=None)[:2]
        for _k in range(4):
            perm = df.sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True)
            assert reference_state(T0, perm)[:2] == base, recs
            assert reference_state(T0, perm, endpoint="any", type_rule="any_ad", event_grace_days=28)[:2] == \
                   reference_state(T0, df, endpoint="any", type_rule="any_ad", event_grace_days=28)[:2]


def test_uncertain_outcomes_are_never_nonevents():
    """Property: a NONEVENT never has (i) any same-date conflict on/before the event cut-off, (ii) any dementia diagnosis of any type
    on/before the cut-off, nor (iii) a record the rules treat as potentially an event (type-uncertain dementia counts as dementia)."""
    rng = np.random.default_rng(22); H = pd.Timestamp(T0) + pd.DateOffset(months=24)
    n_nonevent = 0
    for _ in range(600):
        recs = _random_history(rng); e = int(rng.choice([0, 28])); w = int(rng.choice([0, 90, 180]))
        s, r, _ = reference_state(T0, _frame(recs), w_days=w, event_grace_days=e)
        if s != NONEVENT: continue
        n_nonevent += 1
        cut = H + pd.Timedelta(days=e)
        dated = [(pd.Timestamp(d), dx) for d, dx, _ in recs if d is not None and pd.Timestamp(d) > pd.Timestamp(T0)]
        assert not any(d <= cut and dx == "DEM" for d, dx in dated), recs                 # any dementia record on/before the cut-off (incl. conflicts)
        assert r == "nondementia_observed_in_window"
    assert n_nonevent > 20


def test_missing_etiology_is_not_ad_when_column_supplied():
    df = pd.DataFrame({"date": ["2021-03-01"], "dx": ["DEM"], "dem_type": [None]})
    assert reference_state(T0, df)[:2] == (INDET, "dementia_type_uncertain")
    assert reference_state(T0, df, endpoint="any")[:2] == (EVENT, "dementia_on_or_before_H")
    df2 = df.drop(columns="dem_type")                                                      # no column: documented AD assumption, flagged
    s, r, f = reference_state(T0, df2)
    assert (s, r) == (EVENT, "dementia_on_or_before_H") and f["dem_type_assumed_ad"]


def test_later_diagnoses_change_primary_endpoint_only_before_the_first_dementia():
    """After the first dementia date, subsequent diagnoses (reversion, further dementia of any type) cannot change the primary state."""
    base = V(("2021-01-01", "NONDEM", "AD"), ("2021-04-01", "DEM", "AD"))
    want = reference_state(T0, base)[:2]
    for tail in [V(("2021-05-01", "NONDEM")), V(("2021-05-01", "DEM", "OTHER")), V(("2021-05-01", "unknown")), V(("2021-05-01", "DEM"), ("2021-05-01", "NONDEM"))]:
        assert reference_state(T0, pd.concat([base, tail], ignore_index=True))[:2] == want == (EVENT, "dementia_on_or_before_H")


def test_endpoint_py_is_order_invariant_and_treats_conflicts_and_unknown_type_as_unlabelled():
    import itertools
    base = pd.DataFrame({"participant_id": [1], "baseline_date": [pd.Timestamp(T0)]})
    cases = {"AD+OTHER same date": ([("2021-03-01", "DEM", "AD"), ("2021-03-01", "DEM", "OTHER")], "dementia_type_uncertain"),
             "DEM(OTHER)+NONDEM same date": ([("2021-03-01", "DEM", "OTHER"), ("2021-03-01", "NONDEM", "AD")], "conflicting_diagnoses_on_or_before_H"),
             "early conflict then qualifying visit": ([("2020-07-01", "DEM", "AD"), ("2020-07-01", "NONDEM", "AD"), ("2021-12-20", "NONDEM", "AD")], "conflicting_diagnoses_on_or_before_H"),
             "missing etiology": ([("2021-03-01", "DEM", "UNKNOWN")], "dementia_type_uncertain")}
    for name, (rows, reason) in cases.items():
        seen = set()
        for p in itertools.permutations(rows):
            v = pd.DataFrame(list(p), columns=["visit_date", "dx", "dem_type"]).assign(participant_id=1)
            r = label_endpoint(base, v, rule="A", endpoint="ad_dementia").iloc[0]
            seen.add((None if np.isnan(r["label"]) else r["label"], r["reason"]))
        assert seen == {(None, reason)}, (name, seen)


def test_summary_date_labeller_does_not_read_missing_type_as_ad():
    from yoad.data.endpoint import label_from_summary_dates
    df = pd.DataFrame({"participant_id": [1, 2, 3], "baseline_date": [T0] * 3, "first_dementia_date": ["2021-03-01"] * 3,
                       "last_nondementia_date": [None] * 3, "first_dementia_type": ["AD", "OTHER", None]})
    out = label_from_summary_dates(df, 24, 90, "ad_dementia", "unlabelled").set_index("participant_id")
    assert out.loc[1, "label"] == 1 and np.isnan(out.loc[2, "label"]) and np.isnan(out.loc[3, "label"])
    assert out.loc[2, "reason"] == "competing_non_AD_dementia" and out.loc[3, "reason"] == "dementia_type_uncertain"
