"""Stage 1 benchmark (re-audit additions): runtime fit guard, leakage-safe late fusion, identical partitions across models.
Synthetic data only."""
import numpy as np
import pytest

from yoad.benchmark import runner as R
from yoad.benchmark.config import ConfigError, load_config
from yoad.benchmark.data import generate_synthetic
from yoad.benchmark.pipeline import assert_fit_rows, build_pipeline
from yoad.evaluation.safeguards import LeakageError


def _cfg(**over):
    b = {"experiment_id": "guard", "seed": 4, "dataset": {"params": {"n_participants": 180, "seed": 21}},
         "metrics": ["roc_auc", "brier"]}
    b.update(over)
    return b


# ---------------------------------------------------------------- runtime fit guard
def test_fit_guard_rejects_a_pipeline_fitted_on_held_out_rows():
    d = generate_synthetic(n_participants=80, seed=2)
    pipe = build_pipeline(load_config(_cfg()), d.features, {"C": 1.0}, 0).fit(d.X, d.y)   # global fit (the bug)
    with pytest.raises(LeakageError, match="held-out"):
        assert_fit_rows(pipe, d.X.index[:100], d.X.index[100:])
    assert_fit_rows(pipe, d.X.index)                                                     # correct claim passes


def test_runner_fails_loudly_if_a_split_leaks_participants(tmp_path, monkeypatch):
    """Inject a broken outer split that puts evaluation rows into training; the run must fail, not report metrics."""
    def leaky(cfg, y, g, seed):
        idx = np.arange(len(y))
        return [(idx, idx[: len(y) // 4])]
    monkeypatch.setattr(R, "_outer_splits", leaky)
    r = R.run(_cfg(), results_base=tmp_path, run_id="leak")
    assert r["status"] == "failed" and r["error"]["type"] == "LeakageError" and not r["summary"]


def test_successful_runs_record_that_the_fit_guard_passed(tmp_path):
    r = R.run(_cfg(), results_base=tmp_path, run_id="ok")
    assert r["status"] == "success" and all(f["fit_guard"] == "passed" for f in r["folds"])


# ---------------------------------------------------------------- late fusion without out-of-fold leakage
def test_late_fusion_runs_and_records_meta_learner(tmp_path):
    cfg = _cfg(fusion={"type": "late", "inner_cv_splits": 3}, model={"params": {"C": 1.0}},
               validation={"validation_fraction": 0.2}, threshold={"method": "youden"})
    r = R.run(cfg, results_base=tmp_path, run_id="late")
    assert r["status"] == "success", r["error"]
    f = r["folds"][0]
    assert f["fusion"]["base_groups"] == ["cognitive", "demographic", "imaging"]
    assert f["fusion"]["oof_inner_splits"] == 3
    assert f["fusion"]["meta_coefficients"]["cognitive"] > 0          # the synthetic signal lives in cognitive/imaging
    assert r["fusion_type"] == "late"


def test_late_fusion_out_of_fold_leak_is_caught(tmp_path, monkeypatch):
    """If the inner folds used for the meta-learner overlapped (in-sample base predictions), the guard must stop the run."""
    real = R.group_kfold_splits

    def overlapping(y, g, n, seed, stratify=True):
        return [(np.arange(len(y)), te) for _, te in real(y, g, n, seed, stratify)]
    monkeypatch.setattr(R, "group_kfold_splits", overlapping)
    cfg = _cfg(fusion={"type": "late", "inner_cv_splits": 3}, model={"params": {"C": 1.0}})
    r = R.run(cfg, results_base=tmp_path, run_id="oofleak")
    assert r["status"] == "failed" and r["error"]["type"] == "LeakageError"


@pytest.mark.parametrize("over,msg", [
    ({"fusion": {"type": "learned"}}, "planned"),
    ({"fusion": {"type": "late"}, "features": {"groups": ["cognitive"]}}, "at least two"),
    ({"fusion": {"type": "late"}, "model": {"search_space": {"C": [0.1, 1.0]}}}, "not implemented in Stage 1"),
    ({"fusion": {"type": "stacked"}}, "unknown fusion.type"),
])
def test_invalid_fusion_configurations_are_rejected(over, msg):
    with pytest.raises(ConfigError, match=msg):
        load_config(_cfg(**over))


# ---------------------------------------------------------------- identical partitions for comparable models
def test_comparable_models_see_identical_participant_partitions(tmp_path):
    common = dict(validation={"outer": {"type": "group_kfold", "n_splits": 3}, "validation_fraction": 0.2},
                  threshold={"method": "youden"})
    a = R.run(_cfg(model={"id": "logistic_l2"}, **common), results_base=tmp_path, run_id="lr")
    b = R.run(_cfg(model={"id": "dummy_prior"}, **common), results_base=tmp_path, run_id="dummy")
    c = R.run(_cfg(model={"id": "logistic_l2"}, sampling={"method": "random_oversample"}, **common),
              results_base=tmp_path, run_id="lr_os")
    assert a["status"] == b["status"] == c["status"] == "success"
    fp = lambda r: [{k: v["participants_sha256"] for k, v in f["partitions"].items()} for f in r["folds"]]
    assert fp(a) == fp(b) == fp(c)
    assert b["summary"]["roc_auc"]["mean"] == pytest.approx(0.5)      # prior-only model has no discrimination
