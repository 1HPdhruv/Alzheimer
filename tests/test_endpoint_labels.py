"""Synthetic participants only. Each case documents one rule of src/cohort/endpoint.py."""
import numpy as np, pandas as pd, pytest
from yoad.data.endpoint import label_endpoint, audit_labels, DEM, NONDEM

def mk(case):
    base = pd.DataFrame([{"participant_id": k, "baseline_date": pd.Timestamp("2020-01-01"), "death_date": v.get("death")} for k, v in case.items()])
    rows = []
    for k, v in case.items():
        for x in v["visits"]:
            r = {"participant_id": k, "visit_date": pd.Timestamp("2020-01-01") + pd.DateOffset(months=x[0]), "dx": x[1]}
            if len(x) > 2: r["dem_type"] = x[2]
            rows.append(r)
    df = pd.DataFrame(rows)
    if "dem_type" in df: df["dem_type"] = df["dem_type"].fillna("AD")   # untyped dementia rows are AD in these fixtures
    return base, df

CASES = {
    "event_at_18": {"visits": [(12, NONDEM), (18, DEM)]},
    "event_at_26": {"visits": [(12, NONDEM), (23, NONDEM), (26, DEM)]},       # after H, but non-dementia seen at 23 mo
    "event_at_25": {"visits": [(12, NONDEM), (22, NONDEM), (25, DEM)]},       # dementia visit nearest to H, 1 month after it
    "straddle": {"visits": [(12, NONDEM), (30, DEM)]},                       # interval-censored across H
    "stable_to_25": {"visits": [(12, NONDEM), (25, NONDEM)]},
    "short_followup": {"visits": [(6, NONDEM), (12, NONDEM)]},
    "reverted": {"visits": [(12, DEM), (24, NONDEM)]},
    "non_ad": {"visits": [(12, NONDEM), (20, DEM, "OTHER")]},
    "died": {"visits": [(6, NONDEM)], "death": pd.Timestamp("2021-06-01")},
    "uncertain_only": {"visits": [(24, "unknown")]},
}

def test_rule_A_labels_and_reasons():
    base, v = mk(CASES); L = label_endpoint(base, v, rule="A").set_index("participant_id")
    assert L.loc["event_at_18", "label"] == 1
    assert L.loc["event_at_26", "label"] == 0 and L.loc["event_at_26", "reason"].startswith("nondementia_observed")
    assert np.isnan(L.loc["straddle", "label"]) and L.loc["straddle", "interval_censored"]
    assert L.loc["stable_to_25", "label"] == 0
    assert np.isnan(L.loc["short_followup", "label"]) and L.loc["short_followup", "reason"] == "insufficient_followup"
    assert L.loc["reverted", "label"] == 1 and L.loc["reverted", "reverted"]          # first dementia dx defines the event
    assert np.isnan(L.loc["died", "label"]) and L.loc["died", "death_before_H"]
    assert np.isnan(L.loc["uncertain_only", "label"]) and L.loc["uncertain_only", "uncertain_dx_seen"]

def test_competing_non_AD_dementia_and_death_sensitivity():
    base, v = mk(CASES)
    assert label_endpoint(base, v, endpoint="any_dementia").set_index("participant_id").loc["non_ad", "label"] == 1
    La = label_endpoint(base, v, endpoint="ad_dementia").set_index("participant_id")
    assert np.isnan(La.loc["non_ad", "label"]) and La.loc["non_ad", "competing_nonAD"]
    Ld = label_endpoint(base, v, death_handling="nonevent").set_index("participant_id")
    assert Ld.loc["died", "label"] == 0 and "SENSITIVITY" in Ld.loc["died", "reason"]

def test_rules_A_and_B_are_not_interchangeable():
    base, v = mk(CASES); A = label_endpoint(base, v, rule="A").set_index("participant_id"); B = label_endpoint(base, v, rule="B").set_index("participant_id")
    assert A.loc["event_at_25", "label"] == 0 and B.loc["event_at_25", "label"] == 1        # B counts an event after 24 months
    assert B.loc["event_at_25", "event_after_H"]
    assert A.loc["event_at_18", "label"] == 1 and np.isnan(B.loc["event_at_18", "label"])   # B ignores it: no in-window visit
    assert A.loc["reverted", "label"] == 1 and B.loc["reverted", "label"] == 0               # reversion handled differently
    assert (A["label"].fillna(-1) != B["label"].fillna(-1)).sum() >= 4

def test_grace_zero_is_strict_and_audit_flags_forced_negative_labels():
    base, v = mk(CASES)
    strict = label_endpoint(base, v, rule="A", grace_days=0).set_index("participant_id")
    assert np.isnan(strict.loc["event_at_26", "label"])                                      # 23-month visit < H when g = 0
    bad = label_endpoint(base, v, rule="A").set_index("participant_id").reset_index()
    assert audit_labels(bad, base, v) == []
    bad.loc[bad.participant_id == "short_followup", "label"] = 0.0                            # wrongly coding insufficient follow-up as 0
    probs = audit_labels(bad, base, v)
    assert probs and "short_followup" in probs[0]
