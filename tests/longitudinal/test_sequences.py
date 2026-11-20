"""Synthetic tests (regressions for the 2026-11-02 review): prediction time at the anchor date, dated modality streams with
their own elapsed times, the merged view and its anchor token, equal-date and conflicting-date handling, infinities,
training-only feature activity / branch enabling, and the fold-safe MRI plausibility rule."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.multimodal import synthetic_cohort as syn
from yoad.longitudinal import config as CF
from yoad.longitudinal import mri_adapter as MA
from yoad.longitudinal import sequences as SQ
from tests.longitudinal.conftest import build

D = pd.Timestamp
COG = CF.TV_BLOCKS["COGFUNC"]


# ------------------------------------------------------------------------------------------------ Priority 1: prediction time
def test_prediction_time_is_the_anchor_date():
    assert CF.POST_DAYS == 0 and CF.WINDOW == (-90, 0) and CF.TIMING == "strict"


def test_no_stream_entry_after_the_anchor(cohort):
    s = cohort["s"]
    for b in SQ.TV:
        real = np.arange(s.SD[b].shape[1])[None, :] < s.SL[b][:, None]
        assert (s.SD[b][real] <= 0).all()


def _first_anchor_rows(s):
    return s.meta.reset_index().sort_values("anchor_month").groupby("participant_id").head(1)


def test_post_anchor_measurements_change_neither_inputs_nor_predictions():
    import torch
    from yoad.longitudinal import train as TR
    d = syn.make(160, seed=3)
    s0 = build(d)["s"]
    first = _first_anchor_rows(s0)
    inject, mri = [], []
    for pid, ad in zip(first["participant_id"], first["anchor_date"]):
        for off in (1, 7, 14, 200):                    # +1..+14 d were allowed by the earlier window; now all excluded
            inject += [(pid, "mmse", ad + pd.Timedelta(days=off), 0.0), (pid, "adas13", ad + pd.Timedelta(days=off), 85.0)]
        mri.append({**d["mri_raw"].iloc[0].to_dict(), "RID": pid, "EXAMDATE": ad + pd.Timedelta(days=10), "IMAGEUID": 9_999_999,
                    "OVERALLQC": "Pass"})
    d2 = dict(d, measurements=pd.concat([d["measurements"], pd.DataFrame(inject, columns=d["measurements"].columns)],
                                        ignore_index=True), mri_raw=pd.concat([d["mri_raw"], pd.DataFrame(mri)], ignore_index=True))
    s1 = build(d2)["s"]
    key = ["participant_id", "anchor_month"]
    i0 = s0.meta.reset_index().merge(first[key], on=key)["index"].to_numpy()
    i1 = s1.meta.reset_index().merge(first[key], on=key)["index"].to_numpy()
    for b in SQ.TV:
        assert np.array_equal(np.nan_to_num(s0.SX[b][i0], nan=-9), np.nan_to_num(s1.SX[b][i1], nan=-9)), b
        assert np.array_equal(np.nan_to_num(s0.SD[b][i0], nan=-9), np.nan_to_num(s1.SD[b][i1], nan=-9)), b
    pre = SQ.Preprocessor().fit(s0, np.arange(s0.n))
    a0, a1 = pre.transform(s0.subset(i0)), pre.transform(s1.subset(i1))
    torch.manual_seed(0)
    for mid in ("B_LSTM", "E_COMBINED", "D_FUSION"):
        from yoad.longitudinal import models as MD
        m = MD.build(mid, 16).eval()
        with torch.no_grad():
            p0 = m(TR.tensors(a0, np.arange(len(i0)))).numpy(); p1 = m(TR.tensors(a1, np.arange(len(i1)))).numpy()
        assert np.array_equal(p0, p1), mid


def test_anchor_window_snapshot_matches_the_frozen_strict_row(cohort):
    s = cohort["s"]
    val, stale = SQ.last_observation(s)
    for k, f in enumerate(COG):
        if f not in s.anchor_frame.columns:
            continue
        ref = s.anchor_frame[f].to_numpy(float)
        recent = stale["COGFUNC"][:, k] * 365.25 <= -CF.WINDOW[0] + 1e-6
        # the frozen strict row = nearest assessment in [-90, 0] = the stream's latest one when it is recent enough
        ok = np.isfinite(ref)
        assert np.allclose(val["COGFUNC"][ok, k], ref[ok]) and recent[ok].all(), f
        assert not (recent & ~ok & np.isfinite(val["COGFUNC"][:, k])).any(), f


# ------------------------------------------------------------------------------------------------ Priority 5: sequence integrity
def _toy(extra_rows=()):
    pid = 900001
    rows = pd.DataFrame({"participant_id": pid, "anchor_month": [0, 12, 13], "anchor_date": [D("2015-01-10"), D("2016-01-15"),
                                                                                         D("2016-01-15")],
                         "is_baseline": [True, False, False], "partition": "train", "state_REF": ["NONEVENT", "EVENT", "EVENT"],
                         "age": 70.0, "female": 1.0, "education_years": 16.0, "apoe4": 1.0})
    meas = pd.DataFrame([(pid, "mmse", D("2015-01-10"), 28.0), (pid, "mmse", D("2015-07-01"), 27.0),
                         (pid, "mmse", D("2016-01-10"), 25.0), (pid, "faq", D("2016-01-12"), 4.0),
                         *extra_rows], columns=["participant_id", "variable", "date", "value"])
    mri = pd.DataFrame({"participant_id": [pid], "date": [D("2015-12-06")], **{f: [1.0] for f in CF.TV_BLOCKS["MRI"]},
                        **{f"raw_{k}": [v] for k, v in zip(MA.RAW_KEYS, (1.5e6, 7000, 3500, 3.3, 20000, 20000, 30000, 3000))}})
    streams, flow = SQ.dated_streams(meas, None, {"mri": (mri, list(mri.columns[2:]))}, D("2026-08-31"))
    return rows, streams, flow


def test_equal_date_anchors_and_flagged_visits():
    rows, streams, _ = _toy()
    s = SQ.build_samples(rows, streams)
    assert list(s.meta["anchor_month"]) == [0, 12] and s.flow["anchors_excluded_equal_date_duplicate"] == 1
    s2 = SQ.build_samples(rows, streams, flagged_visits=frozenset({(900001, 12)}))
    assert list(s2.meta["anchor_month"]) == [0, 13] and s2.flow["anchors_excluded_conflicting_dxsum_dates"] == 1


def test_measurement_specific_elapsed_times_and_merged_anchor_token(monkeypatch):
    monkeypatch.setattr(CF, "MIN_BRANCH_TRAIN_PARTICIPANTS", 1)     # one-participant toy: evidence rules relaxed here only
    monkeypatch.setattr(CF, "MIN_FEATURE_TRAIN_OBS", 1)
    rows, streams, _ = _toy()
    s = SQ.build_samples(rows, streams)
    n = 1                                                       # anchor 2016-01-15
    assert list(s.SD["COGFUNC"][n, :s.SL["COGFUNC"][n]]) == [-370, -198, -5, -3]
    assert list(s.SD["MRI"][n, :s.SL["MRI"][n]]) == [-40]       # its own date, not snapped to a visit
    mv = SQ.merged_view(s.SX, s.SD, s.SL)
    L = mv["lengths"][n]
    assert L == 6 and mv["dt"][n, L - 1] == 0                   # 5 observation dates + the anchor token, last
    assert np.allclose(mv["dt"][n, :L] * 365.25, [-370, -198, -40, -5, -3, 0])
    assert np.allclose(mv["gap"][n, L - 1] * 365.25, 3)         # gap = time from the latest observation to the anchor
    pre = SQ.Preprocessor().fit(s, [0, 1])
    a = pre.transform(s)
    assert np.allclose(a["st_MRI"][n, 0] * 365.25, -40) and np.allclose(a["sg_COGFUNC"][n, 1] * 365.25, 172)
    assert a["m_COGFUNC"][n, L - 1].sum() == 0 and a["m_MRI"][n, L - 1].sum() == 0   # anchor token carries no measurement


def test_truncation_keeps_most_recent_and_anchor_token_last(cohort):
    s = cohort["s"]
    mv = SQ.merged_view(s.SX, s.SD, s.SL, max_dates=3)
    assert mv["lengths"].max() <= 3
    i = np.arange(s.n)
    assert (mv["dt"][i, mv["lengths"] - 1] == 0).all()
    full = SQ.merged_view(s.SX, s.SD, s.SL)
    n = int(np.argmax(full["lengths"]))
    Lf, L3 = full["lengths"][n], mv["lengths"][n]
    assert np.allclose(mv["dt"][n, :L3 - 1], full["dt"][n, Lf - L3:Lf - 1])


def test_conflicting_same_date_and_duplicates_are_resolved_deterministically():
    pid = 1
    meas = pd.DataFrame([(pid, "mmse", D("2015-01-01"), 28.0), (pid, "mmse", D("2015-01-01"), 28.0),   # exact duplicate
                         (pid, "faq", D("2015-02-01"), 2.0), (pid, "faq", D("2015-02-01"), 5.0)],      # conflict -> removed
                        columns=["participant_id", "variable", "date", "value"])
    streams, flow = SQ.dated_streams(meas, None, {}, D("2026-08-31"))
    c = streams["COGFUNC"]
    assert len(c) == 1 and c.iloc[0]["mmse"] == 28.0 and flow["COGFUNC_measurements_conflicting_same_date"] == 2


def test_check_sequences_rejects_disorder_and_future(cohort):
    s = cohort["s"].subset(np.arange(20))
    n = int(np.argmax(s.SL["COGFUNC"]))
    bad = s.subset(np.arange(s.n)); bad.SD["COGFUNC"][n, 0] = 5.0
    with pytest.raises(SQ.TemporalLeakageError):
        SQ.check_sequences(bad)
    bad2 = s.subset(np.arange(s.n)); bad2.SD["COGFUNC"][n, 1] = bad2.SD["COGFUNC"][n, 0]
    with pytest.raises(SQ.TemporalLeakageError):
        SQ.check_sequences(bad2)


# ------------------------------------------------------------------------------------------------ infinities
def test_infinite_values_are_missing_and_counted():
    pid = 1
    meas = pd.DataFrame([(pid, "mmse", D("2015-01-01"), np.inf), (pid, "adas13", D("2015-01-01"), -np.inf),
                         (pid, "faq", D("2015-01-01"), 3.0)], columns=["participant_id", "variable", "date", "value"])
    csf = pd.DataFrame({"participant_id": [pid, pid], "date": [D("2015-01-01"), D("2015-06-01")],
                        "csf_log_ptau_abeta42": [-np.inf, 0.1], "csf_abeta42": [np.inf, 900.0], "csf_ptau": [0.0, 20.0]})
    streams, flow = SQ.dated_streams(meas, None, {"csf": (csf, list(csf.columns[2:]))}, D("2026-08-31"))
    assert flow["COGFUNC_infinite_values_set_missing"] == 2 and flow["CSF_infinite_values_set_missing"] == 2
    assert np.isfinite(streams["CSF"][list(CF.TV_BLOCKS["CSF"])].to_numpy(float)[~np.isnan(
        streams["CSF"][list(CF.TV_BLOCKS["CSF"])].to_numpy(float))]).all()


def test_preprocessor_ignores_infinity_and_outputs_are_finite(cohort):
    s = cohort["dev"].subset(np.arange(cohort["dev"].n))
    n = int(np.argmax(s.SL["COGFUNC"]))
    s.SX["COGFUNC"][n, 0, 0] = np.inf; s.SX["COGFUNC"][n, 0, 1] = -np.inf; s.static[n, 0] = np.inf
    ref = SQ.Preprocessor().fit(cohort["dev"], np.arange(s.n))
    pre = SQ.Preprocessor().fit(s, np.arange(s.n))
    a = pre.transform(s)
    assert all(np.isfinite(v).all() for v in a.values() if v.dtype.kind == "f")
    assert a["sm_COGFUNC"][n, 0, 0] == 0 and a["sm_COGFUNC"][n, 0, 1] == 0
    assert np.isfinite(pre.stats_["COGFUNC"][0]).all() and np.isfinite(pre.stats_["STATIC"][0]).all()
    assert abs(pre.stats_["COGFUNC"][0][0] - ref.stats_["COGFUNC"][0][0]) < 0.2   # one value removed, statistics finite


# ------------------------------------------------------------------------------------------------ Priority 2: training-only activity
def _split(s):
    pid = s.meta["participant_id"].to_numpy()
    ids = np.unique(pid)
    tr = np.flatnonzero(np.isin(pid, ids[: len(ids) // 2])); te = np.flatnonzero(~np.isin(pid, ids[: len(ids) // 2]))
    return tr, te


def test_feature_absent_in_training_stays_inactive_in_evaluation(cohort):
    s = cohort["dev"].subset(np.arange(cohort["dev"].n))
    tr, te = _split(s)
    k = COG.index("trabscor")
    s.SX["COGFUNC"][tr, :, k] = np.nan                         # never observed in training ...
    assert np.isfinite(s.SX["COGFUNC"][te, :, k]).any()        # ... but observed in evaluation
    pre = SQ.Preprocessor().fit(s, tr)
    a = pre.transform(s)
    assert not pre.active_["COGFUNC"][k]
    for key in ("m_COGFUNC", "sm_COGFUNC"):
        assert a[key][te][..., k].sum() == 0 and a[key.replace("m_", "x_")][te][..., k].sum() == 0
    assert a["lm_COGFUNC"][te, k].sum() == 0


def test_modality_absent_in_training_disables_the_branch(cohort):
    import torch
    from yoad.longitudinal import models as MD, train as TR
    s = cohort["dev"].subset(np.arange(cohort["dev"].n))
    tr, te = _split(s)
    s.SX["AMY"][tr] = np.nan; s.SD["AMY"][tr] = np.nan; s.SL["AMY"][tr] = 0
    assert (s.SL["AMY"][te] > 0).any()
    pre = SQ.Preprocessor().fit(s, tr)
    assert not pre.enabled_["AMY"]
    a = pre.transform(s)
    assert a["m_AMY"].sum() == 0 and a["sm_AMY"].sum() == 0 and a["lm_AMY"].sum() == 0 and not a["sv_AMY"].any()
    s2 = s.subset(np.arange(s.n)); s2.SX["AMY"][te] = np.where(np.isfinite(s2.SX["AMY"][te]), 1e6, np.nan)
    a2 = pre.transform(s2)
    torch.manual_seed(0)
    for mid in ("B_LSTM", "C_TRANSFORMER", "D_FUSION", "E_COMBINED"):
        m = MD.build(mid, 16).eval()
        with torch.no_grad():
            assert np.array_equal(m(TR.tensors(a, te)).numpy(), m(TR.tensors(a2, te)).numpy()), mid


def test_corrupted_evaluation_data_do_not_change_training_transformations(cohort):
    s = cohort["dev"]
    tr, te = _split(s)
    p1 = SQ.Preprocessor().fit(s, tr)
    c = s.subset(np.arange(s.n))
    for b in SQ.TV:
        c.SX[b][te] = np.where(np.isfinite(c.SX[b][te]), 1e7, np.nan)
    c.raw_mri[te] = np.where(np.isfinite(c.raw_mri[te]), 1.0, np.nan)
    c.static[te] = -1e7
    p2 = SQ.Preprocessor().fit(c, tr)
    for b in p1.stats_:
        for u, v in zip(p1.stats_[b], p2.stats_[b]):
            assert np.allclose(u, v)
    assert p1.summary() == p2.summary() and p1.mri_medians_ == p2.mri_medians_


# ------------------------------------------------------------------------------------------------ Priority 3: MRI leakage
def test_mri_adapter_applies_no_source_wide_median_rule():
    d = syn.make(80, seed=4)
    from yoad.multimodal import sources as SRC
    codes = SRC.verify_mri_codes(syn.datadic())
    t1, _ = MA.mri_stream_table(d["mri_raw"], codes)
    raw = d["mri_raw"].copy()
    raw.loc[raw.index[:5], "ST29SV"] = 1e9                     # a few absurd scans would shift a source-wide decision ...
    t2, info = MA.mri_stream_table(raw, codes)
    assert set(t1.columns) == set(t2.columns)                   # ... but the adapter keeps every feature (fixed rules only)
    assert all(v.startswith("kept") for v in info["feature_status"].values())


def test_mri_retention_is_decided_inside_the_training_fold(cohort):
    s = cohort["dev"]
    tr, te = _split(s)
    base = SQ.Preprocessor().fit(s, tr)
    assert base.active_["MRI"].all()
    c = s.subset(np.arange(s.n))
    hip = SQ.RAW_MRI.index("raw_mri_hippocampus")
    c.raw_mri[te, :, hip] = np.where(np.isfinite(c.raw_mri[te, :, hip]), 1e9, np.nan)   # implausible EVALUATION values
    assert np.array_equal(SQ.Preprocessor().fit(c, tr).active_["MRI"], base.active_["MRI"])
    c.raw_mri[tr, :, hip] = np.where(np.isfinite(c.raw_mri[tr, :, hip]), 1e9, np.nan)   # implausible TRAINING values
    p = SQ.Preprocessor().fit(c, tr)
    assert not p.active_["MRI"][CF.TV_BLOCKS["MRI"].index("mri_hippocampus")] and p.active_["MRI"].sum() == 6
    icv = SQ.RAW_MRI.index("raw_icv")
    c2 = s.subset(np.arange(s.n)); c2.raw_mri[tr, :, icv] = np.where(np.isfinite(c2.raw_mri[tr, :, icv]), 10.0, np.nan)
    p2 = SQ.Preprocessor().fit(c2, tr)
    act = dict(zip(CF.TV_BLOCKS["MRI"], p2.active_["MRI"]))
    assert not any(act[f] for f in MA.VOLUME_FEATURES) and act["mri_entorhinal_thickness"]


def test_latest_only_and_weights(cohort):
    s = cohort["s"]
    a = SQ.latest_only(s)
    i = np.arange(s.n)
    for b in SQ.TV:
        has = s.SL[b] > 0
        assert (a.SL[b] == has).all()
        assert np.array_equal(a.SD[b][has, 0], s.SD[b][i[has], s.SL[b][has] - 1])
    w = SQ.weights(s.meta["participant_id"].to_numpy())
    assert np.allclose(pd.Series(w).groupby(s.meta["participant_id"].to_numpy()).sum(), 1.0)
