"""Stage 1 benchmark: configuration validation (Test 5), end-to-end smoke (Test 6), logging and governance safety (Test 7)."""
import csv
import json
import re
from pathlib import Path

import numpy as np
import pytest

from yoad.benchmark.config import ConfigError, load_config
from yoad.benchmark.metrics import compute_metrics, fit_calibrator, select_threshold
from yoad.benchmark.models import REGISTRY, ModelSpec, register_model
from yoad.benchmark.runner import RunExistsError, run

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "benchmark" / "synthetic_smoke_test.yaml"


def _base(**over):
    b = {"experiment_id": "cfg_test", "dataset": {"params": {"n_participants": 120, "seed": 1}}}
    b.update(over)
    return b


# ---------------------------------------------------------------- Test 5: configuration validation
def test_smoke_config_is_valid():
    cfg = load_config(SMOKE)
    assert cfg["model"]["id"] == "logistic_l2" and cfg["dataset"]["adapter"] == "synthetic"


@pytest.mark.parametrize("over,msg", [
    ({"model": {"id": "no_such_model"}}, "unknown model id"),
    ({"model": {"id": "catboost"}}, "requires missing package"),
    ({"model": {"id": "lightgbm"}}, "requires missing package"),
    ({"model": {"id": "tabnet"}}, "planned but not implemented"),
    ({"model": {"id": "svm_rbf"}}, "outputs scores"),
    ({"metrics": ["roc_auc", "accuracy_at_magic"]}, "unknown metric"),
    ({"input_path": "../01_RAW_DATA/x.csv"}, "unknown configuration key 'input_path'"),
    ({"dataset": {"adapter": "adni"}}, "unknown dataset adapter"),
    ({"dataset": {"params": {"path": "data.csv"}}}, "not accepted by adapter"),
    ({"sampling": {"method": "smotenc"}}, "planned but not implemented"),
    ({"model": {"id": "dummy_prior"}, "sampling": {"method": "class_weight"}}, "does not support class_weight"),
    ({"threshold": {"method": "youden"}}, "validation_fraction > 0"),
    ({"calibration": {"method": "isotonic"}}, "validation_fraction > 0"),
    ({"validation": {"outer": {"test_fraction": 1.5}}}, "test_fraction"),
    ({"validation": {"outer": {"type": "group_kfold", "n_splits": 1}}}, "n_splits"),
    ({"validation": {"outer": {"type": "row_kfold"}}}, "group_holdout or group_kfold"),
    ({"validation": {"unit": "observation"}}, "participant"),
    ({"output": {"root": "/tmp/elsewhere"}}, "inside 'results/'"),
    ({"output": {"root": "results/../../outside"}}, "inside 'results/'"),
    ({"task": "survival"}, "planned but not implemented"),
    ({"features": {"groups": ["genetic"]}}, "unknown for adapter"),
    ({"model": {"search_space": {"max_depth": [2, 3]}}}, "not allowed"),
    ({"model": {"params": {"C": 1.0}, "search_space": {"C": [1.0]}}}, "both fixed"),
    ({"feature_selection": {"method": "anova"}}, "positive integer"),
    ({"feature_selection": {"method": "stability"}}, "planned"),
    ({"feature_selection": {"method": "l1"}}, "l1_C"),
    ({"feature_selection": {"method": "pca", "n_components": 1.5}}, "n_components"),
    ({"preprocessing": {"numeric_imputation": "complete_case"}}, "planned"),
    ({"preprocessing": {"extra_steps": ["not_registered"]}}, "step hook"),
    ({"experiment_id": "bad id/../x"}, "experiment_id"),
])
def test_invalid_configurations_are_rejected_with_clear_errors(over, msg):
    with pytest.raises(ConfigError, match=re.escape(msg) if "'" in msg else msg):
        load_config(_base(**over))


# ---------------------------------------------------------------- metrics, calibration, thresholds
def test_metric_failures_are_recorded_not_hidden():
    vals, fails = compute_metrics(["roc_auc", "brier", "sensitivity"], [0, 0, 0], [0.1, 0.2, 0.3], 0.5)
    assert "roc_auc" in fails and "brier" in vals and "sensitivity" in fails


def test_threshold_and_calibration_use_validation_inputs_only():
    rng = np.random.default_rng(0)
    yv = rng.integers(0, 2, 200); pv = np.clip(0.3 + 0.4 * yv + rng.normal(0, .15, 200), 0.01, 0.99)
    t = select_threshold("youden", None, yv, pv)
    assert 0.3 < t < 0.75
    assert select_threshold("fixed", 0.4) == 0.4
    cal = fit_calibrator("sigmoid", yv, pv, 0)
    out = cal(np.array([0.1, 0.5, 0.9]))
    assert np.all(np.diff(out) > 0) and np.all((out > 0) & (out < 1))
    with pytest.raises(ValueError):
        fit_calibrator("isotonic", np.zeros(10), np.linspace(0, 1, 10), 0)


# ---------------------------------------------------------------- Test 6: end-to-end smoke and reproducibility
def test_smoke_run_end_to_end_and_reproducible(tmp_path):
    a = run(SMOKE, results_base=tmp_path, run_id="a")
    b = run(SMOKE, results_base=tmp_path, run_id="b")
    assert a["status"] == "success", a["error"]
    assert a["synthetic_data"] is True and "synthetic" in a["data_provenance"].lower()
    for m in ["roc_auc", "average_precision", "brier", "log_loss", "balanced_accuracy", "sensitivity", "specificity", "f1_macro"]:
        assert a["summary"][m]["mean"] is not None
    assert a["folds"][0]["prediction_sha256"] == b["folds"][0]["prediction_sha256"]
    assert a["summary"] == b["summary"] and a["folds"][0]["selected_params"] == b["folds"][0]["selected_params"]
    assert a["config_hash"] == b["config_hash"]
    d = tmp_path / "results/benchmark/synthetic_smoke_test/a"
    res = json.loads((d / "result.json").read_text())
    for key in ["run_id", "experiment_id", "timestamp_utc", "synthetic_data", "config_hash", "seed", "model_id",
                "feature_set_id", "preprocessing", "sampling", "split_strategy", "summary", "failed_metrics",
                "runtime_seconds", "status", "error", "environment"]:
        assert key in res
    assert (d / "config.resolved.yaml").exists()
    rows = list(csv.DictReader((d / "metrics.csv").read_text().splitlines()))
    assert {r["metric"] for r in rows} >= {"roc_auc", "brier"}
    index = list(csv.DictReader((tmp_path / "results/benchmark/synthetic_smoke_test/runs_index.csv").read_text().splitlines()))
    assert [r["run_id"] for r in index] == ["a", "b"]


def test_group_kfold_outer_runs_and_reports_per_fold(tmp_path):
    r = run(_base(validation={"outer": {"type": "group_kfold", "n_splits": 3}}, metrics=["roc_auc", "brier"]),
            results_base=tmp_path, run_id="k")
    assert r["status"] == "success" and len(r["folds"]) == 3 and r["summary"]["roc_auc"]["n_folds"] == 3
    assert r["summary"]["roc_auc"]["sd"] is not None


def test_existing_run_is_never_overwritten(tmp_path):
    run(_base(metrics=["roc_auc"]), results_base=tmp_path, run_id="same")
    before = (tmp_path / "results/benchmark/cfg_test/same/result.json").read_text()
    with pytest.raises(RunExistsError):
        run(_base(metrics=["roc_auc"]), results_base=tmp_path, run_id="same")
    assert (tmp_path / "results/benchmark/cfg_test/same/result.json").read_text() == before


def test_failed_runs_are_logged(tmp_path):
    def boom(p, s, cw):
        raise RuntimeError("synthetic failure")
    register_model(ModelSpec("_always_fails", "implemented", builder=boom), replace=True)
    try:
        r = run(_base(model={"id": "_always_fails"}, metrics=["roc_auc"]), results_base=tmp_path, run_id="f")
    finally:
        del REGISTRY["_always_fails"]
    assert r["status"] == "failed" and r["error"]["type"] == "RuntimeError"
    saved = json.loads((tmp_path / "results/benchmark/cfg_test/f/result.json").read_text())
    assert saved["status"] == "failed"
    index = list(csv.DictReader((tmp_path / "results/benchmark/cfg_test/runs_index.csv").read_text().splitlines()))
    assert index[-1]["status"] == "failed"


def test_logs_contain_no_observation_level_values(tmp_path):
    run(SMOKE, results_base=tmp_path, run_id="p")
    text = (tmp_path / "results/benchmark/synthetic_smoke_test/p/result.json").read_text()
    assert "SYN-" not in text                                             # no participant identifiers
    assert "syn_cog_1" not in text                                        # no feature columns or values


# ---------------------------------------------------------------- Test 7: governance and workspace safety
def test_benchmark_code_reads_no_data_files_and_names_no_restricted_location():
    forbidden_calls = re.compile(r"read_csv|read_excel|read_parquet|read_table|pyreadr|read_sas|read_stata|np\.load\(|pickle")
    forbidden_paths = re.compile(r"01_RAW_DATA|02_LONGITUDINAL|04_DERIVED|99_ADNIMERGE|investigator_nacc|local_outputs|Urop_RECOVERY")
    for f in (ROOT / "src" / "yoad" / "benchmark").glob("*.py"):
        src = f.read_text()
        assert not forbidden_calls.search(src), f
        assert not forbidden_paths.search(src), f
        assert src.count("open(") <= {"config.py": 1, "runner.py": 4, "campaign.py": 6}.get(f.name, 0), f   # config read + result writes only


def test_outputs_stay_inside_results_base(tmp_path):
    run(_base(metrics=["roc_auc"]), results_base=tmp_path, run_id="x")
    created = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert created and all(p.relative_to(tmp_path).parts[0] == "results" for p in created)
