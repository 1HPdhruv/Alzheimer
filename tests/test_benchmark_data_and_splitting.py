"""Stage 1 benchmark: synthetic data integrity (Test 1) and group separation (Test 2). Synthetic data only."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from yoad.benchmark.data import ADAPTERS, DatasetAdapter, GovernanceError, generate_synthetic, load_dataset
from yoad.benchmark.splitting import SplitError, check_disjoint, group_kfold_splits, group_partition
from yoad.evaluation.safeguards import LeakageError


# ---------------------------------------------------------------- Test 1: synthetic data integrity
def test_generator_is_deterministic_and_seed_sensitive():
    a, b = generate_synthetic(seed=3), generate_synthetic(seed=3)
    pd.testing.assert_frame_equal(a.X, b.X)
    pd.testing.assert_series_equal(a.y, b.y)
    pd.testing.assert_series_equal(a.groups, b.groups)
    pd.testing.assert_series_equal(a.visit_dates, b.visit_dates)
    assert not generate_synthetic(seed=4).X.equals(a.X)


def test_generator_marks_data_synthetic_and_has_valid_structure():
    d = generate_synthetic(n_participants=200, positive_fraction=0.25, missing_rate=0.15, seed=1)
    assert d.synthetic and "synthetic" in d.provenance.lower()
    assert set(d.y.unique()) <= {0, 1}
    assert d.groups.notna().all() and d.groups.str.startswith("SYN-").all()
    assert d.y.groupby(d.groups).nunique().max() == 1                     # label constant within participant
    assert d.groups.value_counts().max() > 1                              # repeated observations exist
    assert set(d.feature_groups()) == {"cognitive", "demographic", "imaging"}
    assert {f.kind for f in d.features} == {"numeric", "categorical"}
    part_y = d.y.groupby(d.groups).max()
    assert 0.15 < part_y.mean() < 0.35                                    # controlled imbalance (requested 0.25)
    assert 0.10 < d.X.isna().to_numpy().mean() < 0.20                     # controlled missingness (requested 0.15)
    assert d.visit_dates.notna().all()


def test_generator_contains_the_documented_signal():
    d = generate_synthetic(n_participants=800, missing_rate=0.0, seed=2)
    assert roc_auc_score(d.y, d.X["syn_cog_1"]) > 0.65
    assert roc_auc_score(d.y, -d.X["syn_img_1"]) > 0.6
    assert 0.4 < roc_auc_score(d.y, d.X["syn_img_6"]) < 0.6               # noise feature


def test_generator_rejects_invalid_parameters():
    for kw in [dict(n_participants=5), dict(positive_fraction=1.2), dict(missing_rate=-0.1), dict(visits_per_participant=(3, 1))]:
        with pytest.raises(ValueError):
            generate_synthetic(**kw)


def test_non_synthetic_adapter_is_refused():
    real_like = generate_synthetic(seed=0)
    real_like.synthetic = False
    ADAPTERS["_test_nonsynthetic"] = DatasetAdapter("_test_nonsynthetic", False, frozenset(), (), lambda: real_like)
    try:
        with pytest.raises(GovernanceError):
            load_dataset("_test_nonsynthetic", {})
    finally:
        del ADAPTERS["_test_nonsynthetic"]


# ---------------------------------------------------------------- Test 2: group separation
def _yg(n=120, seed=0):
    d = generate_synthetic(n_participants=n, seed=seed)
    return d.y.to_numpy(), d.groups.to_numpy()


def test_three_way_partition_has_no_participant_overlap_and_both_classes():
    y, g = _yg()
    p = group_partition(y, g, {"train": .6, "validation": .2, "test": .2}, seed=5,
                        require_both_classes=("train", "validation", "test"))
    sets = {k: set(g[v]) for k, v in p.items()}
    assert not (sets["train"] & sets["validation"]) and not (sets["train"] & sets["test"]) and not (sets["validation"] & sets["test"])
    assert sum(len(v) for v in p.values()) == len(y)                      # every observation assigned exactly once
    for v in p.values():
        assert len(np.unique(y[v])) == 2


def test_kfold_every_participant_in_exactly_one_test_fold():
    y, g = _yg()
    folds = group_kfold_splits(y, g, 5, seed=1)
    seen = np.concatenate([te for _, te in folds])
    assert sorted(seen) == list(range(len(y)))
    for tr, te in folds:
        assert not set(g[tr]) & set(g[te])


def test_split_is_deterministic():
    y, g = _yg()
    a = group_partition(y, g, {"train": .75, "test": .25}, seed=9)
    b = group_partition(y, g, {"train": .75, "test": .25}, seed=9)
    assert all(np.array_equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("y,g,msg", [
    ([0, 1, 0], ["a", "b"], "different lengths"),
    ([0, 1, 0, 1], ["a", None, "c", "d"], "missing group"),
    ([0, 2, 0, 1], ["a", "b", "c", "d"], "binary"),
])
def test_invalid_inputs_raise(y, g, msg):
    with pytest.raises(SplitError, match=msg):
        group_partition(y, g, {"train": .5, "test": .5}, seed=0)


def test_impossible_configurations_raise_instead_of_falling_back():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0]); g = np.array(["p1", "p1", "p2", "p3", "p4", "p5", "p6", "p7"])
    with pytest.raises(SplitError, match="class 1 has 1"):
        group_partition(y, g, {"train": .5, "test": .5}, seed=0)      # one positive participant cannot be in both sides
    with pytest.raises(SplitError):
        group_kfold_splits(y, g, 3, seed=0)
    with pytest.raises(SplitError, match="n_splits"):
        group_kfold_splits(*_yg(), 1, seed=0)
    with pytest.raises(SplitError, match="fractions"):
        group_partition(*_yg(), {"train": .5, "test": .6}, seed=0)


def test_single_class_partition_is_detected():
    y = np.array([0, 0, 0, 0, 0, 1]); g = np.array(["a", "b", "c", "d", "e", "f"])   # one positive: cannot be on both sides
    with pytest.raises(SplitError, match="single class"):
        group_partition(y, g, {"train": .5, "test": .5}, seed=0, stratify=False, require_both_classes=("test", "train"))


def test_check_disjoint_detects_overlap():
    g = np.array(["a", "a", "b"])
    with pytest.raises(LeakageError):
        check_disjoint({"train": np.array([0]), "test": np.array([1, 2])}, g)
