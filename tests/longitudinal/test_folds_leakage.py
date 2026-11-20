"""Synthetic tests: participant-level folds and leakage — outer folds broadcast to anchors, identical folds across candidates,
grouped inner splits, and outer-test anchors that cannot influence training, tuning, thresholds or preprocessing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yoad.evaluation.safeguards import LeakageError
from yoad.longitudinal import baselines as BL
from yoad.longitudinal import config as CF
from yoad.longitudinal import evaluate as EV


def test_anchor_folds_follow_participants(cohort):
    s, folds = cohort["dev"], cohort["folds"]
    af = EV.anchor_folds(s.meta, folds)
    pid = s.meta["participant_id"].to_numpy()
    for r in range(CF.REPEATS):
        assert pd.Series(af[r]).groupby(pid).nunique().max() == 1
        f = folds[folds["repeat"] == r].set_index("participant_id")["fold"]
        assert (af[r] == pd.Series(pid).map(f).to_numpy()).all()       # the frozen benchmark folds, unchanged


def test_outer_tasks_are_participant_disjoint_and_shared_by_all_candidates(cohort):
    s = cohort["dev"]
    tasks = EV.outer_tasks(s.meta, EV.anchor_folds(s.meta, cohort["folds"]))
    assert len(tasks) == CF.REPEATS * CF.SPLITS
    pid = s.meta["participant_id"].to_numpy()
    for r, k, tr, te in tasks:
        assert not set(pid[tr]) & set(pid[te]) and len(tr) + len(te) == s.n
    specs = EV.build_specs(list(CF.CANDIDATES), s, tasks)
    by = {}
    for sp in specs:
        by.setdefault((sp["r"], sp["k"]), []).append(sp["te"])
    assert all(all(np.array_equal(v[0], x) for x in v) for v in by.values())


def test_overlapping_folds_are_refused(cohort):
    s = cohort["dev"]
    af = EV.anchor_folds(s.meta, cohort["folds"])
    bad = af.copy()
    pid = s.meta["participant_id"].to_numpy()
    one = np.flatnonzero(pid == pd.Series(pid).value_counts().index[0])      # the participant with most anchors
    if len(one) > 1:
        bad[0, one[0]] = (bad[0, one[1]] + 1) % CF.SPLITS          # split one participant's anchors across folds
        with pytest.raises(LeakageError):
            EV.outer_tasks(s.meta, bad)
    missing = s.meta.copy(); missing.loc[0, "participant_id"] = -1
    with pytest.raises(LeakageError):
        EV.anchor_folds(missing, cohort["folds"])


def test_inner_splits_are_grouped(cohort):
    s = cohort["dev"]
    y, pid = s.meta["y"].to_numpy(int), s.meta["participant_id"].to_numpy()
    for tr, va in BL.inner_splits(y, pid, CF.INNER_SPLITS, 7):
        assert not set(pid[tr]) & set(pid[va])


def _corrupt(s, te):
    c = s.subset(np.arange(s.n))
    for b in c.SX:
        c.SX[b][te] = np.where(np.isnan(c.SX[b][te]), np.nan, 1e4)
    c.raw_mri[te] = np.where(np.isnan(c.raw_mri[te]), np.nan, 1.0)
    c.static[te] = 99.0
    c.anchor_frame.loc[te, [x for x in c.anchor_frame.columns if x in ("mmse", "adas13", "faq", "age", "cdrsb")
                            or x.startswith("fs_")]] = 1e4
    return c


@pytest.mark.parametrize("cid", ("D_FUSION", "E_COMBINED", "A1_XGB_MATCHED", "LR_REF", "A_SIMKHADA"))
def test_outer_test_anchors_cannot_change_training(cohort, monkeypatch, cid):
    monkeypatch.setattr(CF, "NEURAL_FIXED", dict(CF.NEURAL_FIXED, max_epochs=12, patience=4))
    s = cohort["dev"]
    r, k, tr, te = EV.outer_tasks(s.meta, EV.anchor_folds(s.meta, cohort["folds"]))[0]
    monkeypatch.setattr(CF, "SIMKHADA_SEARCH", {"n_iter": 3, "inner_splits": 3})
    run = lambda ss: EV.run_task(EV.build_specs([cid], ss, [(r, k, tr, te)])[0])
    a, b = run(s), run(_corrupt(s, te))
    assert "error" not in a and "error" not in b
    assert a["threshold"] == b["threshold"] and str(a["params"]) == str(b["params"])
    assert a["threshold_recal"] == b["threshold_recal"]                  # recalibration fitted on training anchors only
    if cid in CF.NEURAL:
        assert a["preprocessing"] == b["preprocessing"]                 # feature activity / branch enabling unchanged
    if cid in CF.NEURAL:
        assert a["inner_scores"] == b["inner_scores"]
    assert not np.allclose(a["p_test"], b["p_test"])           # the corruption is visible only in test predictions


def test_preprocessing_leak_tripwire(cohort):
    s = cohort["dev"]
    tr = np.arange(s.n)
    with pytest.raises(LeakageError):                          # outer-test anchors inside the training indices
        EV.neural_task("D_FUSION", s, tr, tr[:3], 0, 0)
