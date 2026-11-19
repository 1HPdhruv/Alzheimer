"""Hand-derived synthetic cases for src/cohort/visit_anchor_labels.py (no real participants; IDs are invented)."""
import pandas as pd
import pytest

from yoad.data.visit_anchor_labels import (EXCLUDED, anchors, label_calendar_reference, label_visitcode_sim2026,
                                            summarise)
from yoad.data.reference_outcome import EVENT, INDET, NONEVENT

T0 = pd.Timestamp("2010-01-01")


def hist(pid, seq):
    """seq: list of (month_code, dx); dates = T0 + month (on schedule)."""
    return pd.DataFrame([{"participant_id": pid, "visit_month": m, "date": T0 + pd.DateOffset(months=m), "dx": d} for m, d in seq])


CASES = {
    "conv_full":      [(0, "MCI"), (6, "MCI"), (12, "DEM"), (18, "DEM"), (24, "DEM")],
    "stable_full":    [(0, "MCI"), (6, "MCI"), (12, "MCI"), (18, "MCI"), (24, "MCI")],
    "conv_lost":      [(0, "MCI"), (12, "DEM")],
    "stable_lost":    [(0, "MCI"), (12, "MCI")],
    "rev_cn":         [(0, "MCI"), (6, "CN"), (24, "MCI")],
    "dem6_mci24":     [(0, "MCI"), (6, "DEM"), (24, "MCI")],
    "dem18_mci24":    [(0, "MCI"), (18, "DEM"), (24, "MCI")],
    "annual_gap":     [(0, "MCI"), (12, "MCI"), (36, "MCI")],
}
V = pd.concat([hist(k, s) for k, s in CASES.items()], ignore_index=True)


def _first(df, pid, col="state"):
    return df[(df.participant_id == pid) & (df.anchor_month == 0)][col].iloc[0]


def test_reference_paper_rule_on_hand_derived_cases():
    a = anchors(V, "first_mci_visit")
    L = label_visitcode_sim2026(V, a)
    expect = {"conv_full": (EVENT, "dementia_by_m24_visitcode"), "stable_full": (NONEVENT, "mci_at_m24"),
              "conv_lost": (EVENT, "dementia_by_m24_visitcode"), "stable_lost": (EXCLUDED, "no_m24_status"),
              "rev_cn": (EXCLUDED, "reverted_to_cn"), "dem6_mci24": (EXCLUDED, "changed_then_mci"),
              "dem18_mci24": (NONEVENT, "mci_at_m24"), "annual_gap": (EXCLUDED, "no_m24_status")}
    for pid, (state, reason) in expect.items():
        assert (_first(L, pid), _first(L, pid, "reason")) == (state, reason), pid


def test_asymmetry_converters_need_no_m24_visit_but_stable_participants_do():
    a = anchors(V, "first_mci_visit")
    L = label_visitcode_sim2026(V, a).set_index("participant_id")
    assert L.loc["conv_lost", "state"] == EVENT and L.loc["stable_lost", "state"] == EXCLUDED


def test_calendar_reference_disagrees_where_expected():
    a = anchors(V, "first_mci_visit")
    C = label_calendar_reference(V, a).set_index("participant_id")
    assert C.loc["conv_full", "state"] == EVENT and C.loc["stable_full", "state"] == NONEVENT
    assert C.loc["conv_lost", "state"] == EVENT                      # first dementia within 24 months
    assert C.loc["stable_lost", "state"] == INDET                    # never a non-event without a window assessment
    assert C.loc["rev_cn", "state"] == NONEVENT                      # reversion to CN is not an exclusion here
    assert C.loc["dem6_mci24", "state"] == EVENT                     # first-dementia rule
    assert C.loc["dem18_mci24", "state"] == EVENT                    # the visit-code rule calls this NONEVENT
    assert C.loc["annual_gap", "state"] == INDET


def test_visit_level_anchors_give_several_samples_per_participant():
    a = anchors(V, "all_mci_visits")
    assert (a.participant_id == "stable_full").sum() == 5
    L = label_visitcode_sim2026(V, a)
    s = L[L.participant_id == "stable_full"].set_index("anchor_month")["state"]
    assert s.loc[0] == NONEVENT and (s.loc[[6, 12, 18, 24]] == EXCLUDED).all()   # later anchors lack an m+24 visit
    c = L[L.participant_id == "conv_full"].set_index("anchor_month")["state"]
    assert c.loc[0] == EVENT and c.loc[6] == EVENT                   # the same converter contributes two positive samples


def test_conflicting_duplicate_visits_are_rejected_and_missing_columns_fail():
    bad = pd.concat([V, hist("conv_full", [(12, "MCI")])])
    with pytest.raises(ValueError, match="conflicting"):
        anchors(bad)
    with pytest.raises(ValueError, match="missing columns"):
        anchors(V.drop(columns=["visit_month"]))
    with pytest.raises(ValueError):
        anchors(V, "every_visit")


def test_summary_is_aggregate_and_suppresses_small_cells():
    a = anchors(V, "all_mci_visits")
    s = summarise(label_visitcode_sim2026(V, a))
    assert set(s.columns) == {"state", "reason", "n_anchors", "n_participants"}
    assert (s["n_participants"] == "<11").all()                      # every synthetic cell here is small


# ------------------------------------------------------------------ death as a competing event (calendar rule only)
from yoad.data.visit_anchor_labels import state_summary  # noqa: E402


def _death_case(death, extra=()):
    """One MCI anchor on 2010-01-01 (H = 2012-01-01) with an MCI visit at 2011-12-01 (inside [H - 90 d, H + 90 d])."""
    v = pd.DataFrame([("D", 0, "2010-01-01", "MCI"), ("D", 23, "2011-12-01", "MCI"), *extra],
                     columns=["participant_id", "visit_month", "date", "dx"])
    v["date"] = pd.to_datetime(v["date"])
    a = anchors(v, "first_mci_visit")
    deaths = None if death is None else {"D": pd.Timestamp(death)}
    return label_calendar_reference(v, a, death_dates=deaths, data_cutoff="2020-01-01").iloc[0][["state", "reason"]].tolist()


def test_death_before_horizon_without_dementia_is_competing():
    assert _death_case("2011-12-15") == [INDET, "competing_death"]          # would be NONEVENT if death were ignored
    assert _death_case(None) == [NONEVENT, "nondementia_observed_in_window"]  # missing death information: treated as alive


def test_dementia_before_death_is_an_event():
    assert _death_case("2011-12-20", extra=[("D", 22, "2011-11-01", "DEM")]) == [EVENT, "dementia_on_or_before_H"]


def test_death_after_horizon_and_boundary_dates():
    assert _death_case("2012-01-01") == [INDET, "competing_death"]          # death exactly on H counts as on/before H
    assert _death_case("2012-01-02") == [NONEVENT, "nondementia_observed_in_window"]   # one day after H: qualifying visit stands
    v = pd.DataFrame([("E", 0, "2010-01-01", "MCI")], columns=["participant_id", "visit_month", "date", "dx"])
    v["date"] = pd.to_datetime(v["date"])
    lab = label_calendar_reference(v, anchors(v, "first_mci_visit"), death_dates={"E": pd.Timestamp("2012-02-01")},
                                   data_cutoff="2020-01-01")
    assert lab.iloc[0][["state", "reason"]].tolist() == [INDET, "died_before_ascertainment"]   # after H, nothing observed


def test_state_summary_reports_competing_separately():
    lab = pd.DataFrame({"state": [EVENT, NONEVENT, INDET, INDET, EXCLUDED],
                        "reason": ["x", "y", "competing_death", "insufficient_followup", "no_m24_status"]})
    assert state_summary(lab, "p_") == {"p_n_event": 1, "p_n_nonevent": 1, "p_n_competing_death": 1,
                                        "p_n_indeterminate_other": 1, "p_n_excluded_visitcode_rule": 1}


def test_adsl_death_reader_documents_precision_and_missing_dates():
    from yoad.experiments.visit_anchor_audit.audit_lib import adsl_death_dates
    adsl = pd.DataFrame({"SUBJID": ["1", "2", "3", "4"], "DTHFL": ["Yes", "Yes", None, "Yes"],
                         "DTHDT": ["2012-03-04", "2013-06-15", None, None], "DTHDTF": [None, "M", None, None]})
    d, flow = adsl_death_dates(adsl)
    assert d == {1: pd.Timestamp("2012-03-04"), 2: pd.Timestamp("2013-06-15")}
    assert flow == {"adsl_death_imputation_handling": "as_recorded", "adsl_deaths_flagged": 3, "adsl_deaths_with_date": 2,
                    "adsl_death_dates_month_or_day_imputed": 1, "adsl_deaths_flagged_without_date": 1}


def test_reference_visitcode_rule_ignores_death_and_calendar_builder_applies_it():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from tests.synthetic_baseline import raw_tables
    from yoad.data.simkhada_dataset import build_dataset
    s0, f0, i0 = build_dataset(raw_tables())
    s1, f1, i1 = build_dataset(raw_tables(), death_dates={2: pd.Timestamp("2012-06-01")})   # SIM: ignored
    pd.testing.assert_frame_equal(s0, s1)
    assert "visit-code reference rule ignores death" in i1["death_rule"] and f1["labels_n_competing_death"] == 0
    r0, g0, _ = build_dataset(raw_tables(), label_rule="REF", data_cutoff="2020-01-01")
    r1, g1, j1 = build_dataset(raw_tables(), label_rule="REF", data_cutoff="2020-01-01", death_dates={2: pd.Timestamp("2012-06-01")})
    assert g1["labels_n_competing_death"] >= 1 and g0["labels_n_competing_death"] == 0 and j1["death_rule"].startswith("competing")
    assert len(r1) < len(r0)                                                    # the competing samples leave the binary endpoint


def test_same_day_dementia_and_death_is_an_event_and_post_death_records_are_ignored():
    assert _death_case("2011-11-01", extra=[("D", 22, "2011-11-01", "DEM")]) == [EVENT, "dementia_on_or_before_H"]
    # a non-dementia record dated after an earlier death cannot make the anchor a non-event
    assert _death_case("2011-11-15") == [INDET, "competing_death"]          # the 2011-12-01 visit is after death


def test_imputed_death_dates_bounds_and_sensitivity_counts():
    from yoad.experiments.visit_anchor_audit.audit_lib import adsl_death_dates, death_imputation_sensitivity
    adsl = pd.DataFrame({"SUBJID": ["1", "2"], "DTHFL": ["Yes", "Yes"], "DTHDT": ["2011-07-01", "2011-07-01"],
                         "DTHDTF": ["M", None]})
    rec, f = adsl_death_dates(adsl)
    early, _ = adsl_death_dates(adsl, "earliest"); late, _ = adsl_death_dates(adsl, "latest")
    assert rec[1] == pd.Timestamp("2011-07-01") and early[1] == pd.Timestamp("2011-01-01") and late[1] == pd.Timestamp("2011-12-31")
    assert early[2] == late[2] == rec[2] == pd.Timestamp("2011-07-01")       # a non-imputed date never moves
    assert f["adsl_death_imputation_handling"] == "as_recorded"
    with pytest.raises(ValueError):
        adsl_death_dates(adsl, "middle")
    # participant 1: anchor 2009-06-01 (H = 2011-06-01), MCI visit 2011-05-01 inside the window. Earliest bound (2011-01-01)
    # puts death before H -> competing; latest bound (2011-12-31) after H -> non-event. One label differs.
    v = pd.DataFrame({"participant_id": [1, 1], "visit_month": [0, 23], "date": pd.to_datetime(["2009-06-01", "2011-05-01"]),
                      "dx": ["MCI", "MCI"]})
    sens = death_imputation_sensitivity(v, adsl, "2020-01-01")
    assert sens["death_imputation_label_changes_first_mci_visit"] == 1


def test_two_by_two_and_audit_share_the_death_reader():
    from yoad.experiments.visit_anchor_audit.audit_lib import adsl_death_dates
    from yoad.experiments.visit_anchor_audit.design_lib import adni_static
    adsl = pd.DataFrame({"SUBJID": ["1", "2"], "AGE": [70, 72], "SEX": ["Male", "Female"], "EDUC": [16, 12],
                         "DTHFL": ["Yes", None], "DTHDT": ["2012-01-01", None], "DTHDTF": ["M", None]})
    apoe = pd.DataFrame({"RID": [1, 2], "GENOTYPE": ["3/4", "3/3"]})
    for mode in ("as_recorded", "earliest", "latest"):
        assert adni_static(adsl, apoe, mode)[1] == adsl_death_dates(adsl, mode)[0]
