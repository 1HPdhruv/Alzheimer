"""Synthetic tests: frozen-population reproduction, instrument timing, landmark history, follow-up data, folds and leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yoad.experiments.visit_anchor_audit import design_lib as frozen
from yoad.multimodal import cohort as C, engine as E, registry as R, sources as SRC
from tests.multimodal import synthetic_cohort as syn
from yoad.features.anchor_predictors import attach_predictors, eligible_anchors
from yoad.evaluation.safeguards import LeakageError

D = pd.Timestamp


@pytest.fixture(scope="module")
def data():
    d = syn.make(320, seed=11)
    mri, _ = SRC.mri_table(d["mri_raw"], syn.FS_EXTRA, SRC.verify_mri_codes(syn.datadic()))
    wide = {"mri": (mri, [c for c in mri.columns if c not in ("participant_id", "date")])}
    fr = C.build_frames(d["visits"], d["measurements"], d["static"], d["cutoff"], d["death_dates"], d["extra_long"], wide)
    return d, fr


def test_frames_reproduce_the_frozen_population_labels_and_split(data):
    d, fr = data
    anchors, _ = eligible_anchors(d["visits"], d["cutoff"], u_days=90, death_dates=d["death_dates"])
    rows, _ = attach_predictors(anchors, d["measurements"], d["static"], windows={k: (-90, 14) for k in ("mmse", "adas13", "faq")},
                                post_days=14, data_cutoff=d["cutoff"])
    rows = frozen.label_anchors(d["visits"], rows, d["cutoff"], d["death_dates"], "any")
    train, test = frozen.split_participants(rows, 20261017, 0.6)
    assert (fr["train"], fr["test"]) == (train, test)
    mine = fr["rows"]["primary"]
    cols = ["participant_id", "anchor_month", "state_SIM", "state_REF", *frozen.PREDICTOR_SETS["primary"]]
    pd.testing.assert_frame_equal(mine[cols].reset_index(drop=True), rows[cols].reset_index(drop=True))
    strict = fr["rows"]["strict"]
    assert (strict["state_REF"].to_numpy() == mine["state_REF"].to_numpy()).all()
    assert strict["mmse_offset_days"].dropna().max() <= 0 and mine["mmse_offset_days"].dropna().max() <= 14
    assert set(mine["partition"]) == {"train", "held_out"} and "phase" in mine.columns


def test_wide_instrument_window_ties_conflicts_and_cutoff():
    anchors = pd.DataFrame({"participant_id": [1, 2, 3], "anchor_month": [0, 0, 0],
                            "anchor_date": [D("2020-01-10")] * 3})
    t = pd.DataFrame({"participant_id": [1, 1, 1, 2, 2, 3, 3], "x": [1.0, 2.0, 99.0, 5.0, 6.0, 7.0, 7.0],
                      "date": ["2020-01-03", "2020-01-17", "2020-01-25", "2019-12-01", "2019-12-01", "2027-01-01", "2020-01-01"]})
    out, flow = C.attach_instrument(anchors, t, ["x"], "primary", D("2026-08-31"), "inst")
    o = out.set_index("participant_id")
    assert o.loc[1, "x"] == 1.0                 # -7 and +7 days: tie -> the earlier; +15 days is after the index time
    assert np.isnan(o.loc[2, "x"])              # same-date conflicting values -> that date removed
    assert o.loc[3, "x"] == 7.0 and flow["inst_after_data_cutoff"] == 1
    strict, _ = C.attach_instrument(anchors, t, ["x"], "strict", D("2026-08-31"), "inst")
    assert strict.set_index("participant_id").loc[1, "x"] == 1.0
    assert flow["inst_candidates_after_index_time"] >= 1


def test_landmark_ignores_labels_and_history_never_uses_post_index_values(data):
    d, fr = data
    rows = fr["rows"]["primary"]
    lm, flow = C.landmark_frame(rows, d["measurements"], d["cutoff"])
    assert (lm["anchor_month"] >= R.LANDMARK_MIN_MONTH).all() and not lm["participant_id"].duplicated().any()
    flipped = rows.copy(); flipped["state_REF"] = flipped["state_REF"].map({"EVENT": "NONEVENT", "NONEVENT": "EVENT"}).fillna("x")
    lm2, _ = C.landmark_frame(flipped, d["measurements"], d["cutoff"])
    assert lm2["anchor_month"].tolist() == lm["anchor_month"].tolist()          # the label is never consulted
    r0 = lm.iloc[0]
    poison = pd.DataFrame({"participant_id": [r0["participant_id"]] * 3, "variable": ["mmse"] * 3,
                           "date": [r0["anchor_date"] + pd.Timedelta(days=k) for k in (15, 40, 400)], "value": [-500.0] * 3})
    lm3, _ = C.landmark_frame(rows, pd.concat([d["measurements"], poison], ignore_index=True), d["cutoff"])
    pd.testing.assert_series_equal(lm3["mmse_slope_per_year"], lm["mmse_slope_per_year"])


def test_follow_up_data_rules():
    v = pd.DataFrame({"participant_id": [1, 1, 1, 2, 2, 3, 3, 4, 4, 4], "visit_month": [0, 6, 12, 0, 6, 0, 6, 0, 6, 48],
                      "date": [D("2020-01-01"), D("2020-07-01"), D("2021-01-01"), D("2020-01-01"), D("2020-07-01"), D("2020-01-01"),
                               D("2020-07-01"), D("2020-01-01"), D("2020-07-01"), D("2024-01-01")],
                      "dx": ["MCI", "MCI", "DEM", "MCI", "MCI", "MCI", "MCI", "MCI", "MCI", "MCI"]})
    base = pd.DataFrame({"participant_id": [1, 2, 3, 4], "anchor_date": [D("2020-01-01")] * 4})
    s = C.survival_data(v, base, D("2026-08-31"), {3: D("2020-03-01")}).set_index("participant_id")
    assert s.loc[1, "event"] == 1 and s.loc[1, "time_days"] == 366
    assert s.loc[2, "event"] == 0 and s.loc[2, "time_days"] == 182
    assert 3 not in s.index                                   # the only later visit is after death: no follow-up
    assert s.loc[4, "time_days"] == pytest.approx(C.ADMIN_CENSOR_DAYS) and s.loc[4, "event"] == 0


def test_outer_folds_cover_every_development_participant_once_per_repeat(data):
    _, fr = data
    rows = fr["rows"]["primary"]
    dev = rows[rows["is_baseline"] & (rows["partition"] == "train")]
    f = C.outer_folds(dev)
    assert f.groupby("repeat")["participant_id"].nunique().eq(dev["participant_id"].nunique()).all()
    assert not f.duplicated(["participant_id", "repeat"]).any()
    assert not set(f["participant_id"]) & set(rows.loc[rows["partition"] == "held_out", "participant_id"])


def test_workers_never_see_outer_test_labels_and_checks_catch_wrong_fits(data):
    d, fr = data
    rows = fr["rows"]["primary"]
    dev = rows[rows["is_baseline"] & (rows["partition"] == "train")]
    sv = C.survival_data(d["visits"], dev, d["cutoff"], d["death_dates"])
    pops = E.build_populations(fr["rows"], None, sv)
    fm = E.fold_maps(C.outer_folds(dev))
    spec = {"id": "t", "model": "lr_l2_ref", "features": R.resolve("FS08_CORE7"), "groups": {}, "imputation": "median",
            "timing": "primary", "ids": pops["P_ALL"].frame.index.to_numpy()}
    tasks = E.build_tasks(spec, pops["P_ALL"], fm)
    t = tasks[0]
    assert "y_te" not in t and not set(t["X_tr"].index) & set(t["X_te"].index)
    r1 = E.run_task(t)
    t2 = dict(t); t2["X_te"] = t["X_te"] * 0 + 1e6            # arbitrary change of outer-test rows
    r2 = E.run_task(t2)
    assert r1["threshold"] == r2["threshold"] and r1["params"] == r2["params"] and "error" not in r1
    cox = E.build_tasks({**spec, "model": "cox_l2"}, pops["P_ALL"], fm)
    for ct in cox:
        assert not set(ct["S_tr"].index) & set(ct["X_te"].index)
        assert len(ct["S_tr"]) > len(ct["X_tr"])              # indeterminate participants are training data only
    est = frozen.make_pipeline_factory("logistic_l2", {"C": 1.0}, spec["features"])("x", 1).fit(t["X_te"].fillna(0), np.r_[0, 1][
        np.arange(len(t["X_te"])) % 2])
    with pytest.raises(LeakageError):
        E._check(est, t["X_tr"].index, t["X_te"].index)
