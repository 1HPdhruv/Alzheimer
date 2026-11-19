"""Stage 2: participant-level bootstrap, two-pass campaign, and model-registry smoke tests. Synthetic data only."""
import csv
import json

import numpy as np
import pytest

from yoad.benchmark.bootstrap import cluster_bootstrap
from yoad.benchmark.campaign import load_campaign, run_campaign, run_campaign_on
from yoad.benchmark.config import ConfigError, load_config
from yoad.benchmark.data import generate_synthetic
from yoad.benchmark.models import REGISTRY, ModelSpec, register_model
from yoad.benchmark.runner import run


# ---------------------------------------------------------------- participant-level bootstrap
def test_identical_predictions_give_zero_paired_difference():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200); g = np.repeat(np.arange(50), 4); p = rng.random(200)
    r = cluster_bootstrap(y, {"a": p, "b": p.copy()}, g, ["roc_auc", "brier"], B=100, seed=1, reference="a")
    for d in r["paired_differences"]:
        assert d["difference"] == 0 and d["ci_low"] == 0 and d["ci_high"] == 0


def test_participant_bootstrap_is_wider_than_visit_bootstrap_for_correlated_visits():
    rng = np.random.default_rng(3)
    n_part, reps = 60, 8                                     # each participant contributes 8 near-identical visits
    yp = rng.integers(0, 2, n_part); sp = yp * 0.6 + rng.normal(0, 0.5, n_part)
    y = np.repeat(yp, reps); s = np.repeat(sp, reps) + rng.normal(0, 0.01, n_part * reps)
    p = 1 / (1 + np.exp(-s))
    grouped = cluster_bootstrap(y, {"m": p}, np.repeat(np.arange(n_part), reps), ["roc_auc"], B=400, seed=2)
    naive = cluster_bootstrap(y, {"m": p}, np.arange(n_part * reps), ["roc_auc"], B=400, seed=2)   # visits as units
    w = lambda r: r["estimates"][0]["ci_high"] - r["estimates"][0]["ci_low"]
    assert w(grouped) > 1.8 * w(naive)


def test_bootstrap_rejects_misalignment_threshold_metrics_and_unknown_reference():
    y = np.array([0, 1, 0, 1]); g = np.arange(4)
    with pytest.raises(ValueError, match="aligned"):
        cluster_bootstrap(y, {"a": np.array([0.1, 0.2])}, g, ["roc_auc"])
    with pytest.raises(ValueError, match="probability metrics"):
        cluster_bootstrap(y, {"a": np.full(4, .5)}, g, ["sensitivity"])
    with pytest.raises(ValueError, match="reference"):
        cluster_bootstrap(y, {"a": np.full(4, .5)}, g, ["brier"], reference="zzz")


def test_bootstrap_counts_skipped_single_class_resamples():
    y = np.array([1, 0, 0, 0]); g = np.arange(4)
    r = cluster_bootstrap(y, {"a": np.array([.9, .1, .2, .3])}, g, ["roc_auc"], B=200, seed=0)
    assert r["skipped_single_class"] > 0 and r["B_used"] + r["skipped_single_class"] == 200


# ---------------------------------------------------------------- campaign
def _camp(**over):
    c = {"campaign_id": "camp_test", "seed": 5,
         "dataset": {"adapter": "synthetic", "params": {"n_participants": 200, "seed": 3}},
         "screening": {"n_splits": 3}, "selection": {"metric": "log_loss", "top_k": 1, "reference": "lr"},
         "bootstrap": {"B": 50}, "metrics": ["roc_auc", "brier", "log_loss", "sensitivity"],
         "base": {"model": {"id": "logistic_l2", "params": {"C": 1.0}}},
         "candidates": [{"name": "lr"}, {"name": "dummy", "overrides": {"model": {"id": "dummy_prior", "params": {}}}},
                        {"name": "lr_cog", "overrides": {"features": {"groups": ["cognitive"]}}}]}
    c.update(over)
    return c


def test_campaign_runs_selects_from_development_only_and_writes_aggregates(tmp_path):
    r = run_campaign(_camp(), results_base=tmp_path, run_id="c1")
    assert r["status"] == "success", r["error"]
    assert all(s["status"] == "success" for s in r["screening"])
    assert "lr" in r["selection"]["finalists"]                        # the declared reference is always a finalist
    assert r["partitions"]["development"]["participants_sha256"] != r["partitions"]["locked_test"]["participants_sha256"]
    d = tmp_path / "results/benchmark/camp_test/c1"
    for f in ["campaign_result.json", "screening.csv", "confirmation.csv", "paired_differences.csv", "campaign.resolved.yaml"]:
        assert (d / f).exists(), f
    text = "".join((d / f).read_text() for f in ["campaign_result.json", "screening.csv", "confirmation.csv"])
    assert "SYN-" not in text and "syn_cog_1" not in text            # no identifiers or feature names/values
    assert json.loads((d / "campaign_result.json").read_text())["synthetic_data"] is True


def test_selection_is_invariant_to_corrupting_the_locked_test_set():
    c = load_campaign(_camp())
    d1 = generate_synthetic(n_participants=200, seed=3)
    d2 = generate_synthetic(n_participants=200, seed=3)
    r1 = run_campaign_on(c, d1)
    # corrupt every feature of the locked-test rows; partitions depend only on labels and groups, so they stay identical
    from yoad.benchmark.splitting import group_partition
    part = group_partition(d2.y.to_numpy(), d2.groups.to_numpy(), {"development": .75, "locked_test": .25}, 5, True,
                           ("development", "locked_test"))
    num = [f.name for f in d2.features if f.kind == "numeric"]
    d2.X.loc[part["locked_test"], num] = 999.0
    r2 = run_campaign_on(c, d2)
    assert r1["partitions"] == r2["partitions"]
    assert r1["selection"] == r2["selection"]
    assert [s["screen_score"] for s in r1["screening"]] == [s["screen_score"] for s in r2["screening"]]
    assert r1["confirmation"]["bootstrap"]["estimates"] != r2["confirmation"]["bootstrap"]["estimates"]   # test set did change


def test_failed_candidate_is_recorded_and_campaign_continues(tmp_path):
    def boom(p, s, cw):
        raise RuntimeError("synthetic failure")
    register_model(ModelSpec("_camp_fail", "implemented", builder=boom), replace=True)
    try:
        c = _camp(candidates=[{"name": "lr"}, {"name": "bad", "overrides": {"model": {"id": "_camp_fail", "params": {}}}}])
        r = run_campaign(c, results_base=tmp_path, run_id="f")
    finally:
        del REGISTRY["_camp_fail"]
    st = {s["candidate"]: s["status"] for s in r["screening"]}
    assert r["status"] == "success" and st == {"lr": "success", "bad": "failed"}
    assert "bad" in [row["candidate"] for row in csv.DictReader((tmp_path / "results/benchmark/camp_test/f/screening.csv").read_text().splitlines())]


@pytest.mark.parametrize("over,msg", [
    ({"selection": {"metric": "log_loss", "top_k": 1, "reference": "nobody"}}, "reference"),
    ({"candidates": [{"name": "lr", "overrides": {"dataset": {"adapter": "synthetic"}}}]}, "may not override"),
    ({"candidates": [{"name": "lr"}, {"name": "lr"}]}, "unique"),
    ({"surprise": 1}, "unknown campaign keys"),
    ({"metrics": ["roc_auc"]}, "also be listed"),
    ({"locked_test_fraction": 0.9}, "locked_test_fraction"),
    ({"candidates": [{"name": "lr", "overrides": {"model": {"id": "catboost"}}}]}, "requires missing package"),
])
def test_invalid_campaigns_are_rejected(over, msg):
    with pytest.raises(ConfigError, match=msg):
        load_campaign(_camp(**over))


# ---------------------------------------------------------------- every available registry model runs end to end
AVAILABLE = [k for k, v in REGISTRY.items() if v.status != "planned" and v.available() and not k.startswith("_")]


@pytest.mark.parametrize("model_id", AVAILABLE)
def test_each_available_model_runs_on_synthetic_data(model_id, tmp_path):
    spec = REGISTRY[model_id]
    cfg = {"experiment_id": f"m_{model_id}", "seed": 1, "dataset": {"params": {"n_participants": 120, "seed": 2}},
           "model": {"id": model_id, "search_metric": "log_loss" if spec.probabilistic else "roc_auc"},
           "metrics": ["roc_auc", "brier"]}
    if not spec.probabilistic:
        cfg.update({"validation": {"validation_fraction": 0.2}, "calibration": {"method": "sigmoid"}})
    r = run(cfg, results_base=tmp_path, run_id="r")
    assert r["status"] == "success", (model_id, r["error"])
    assert 0 <= r["summary"]["brier"]["mean"] <= 1


def test_score_only_models_require_calibration():
    with pytest.raises(ConfigError, match="outputs scores"):
        load_config({"experiment_id": "s", "model": {"id": "linear_svm", "search_metric": "roc_auc"}})


def test_log_oe_metric_matches_definition_and_fails_without_events():
    from yoad.benchmark.metrics import compute_metrics
    vals, _ = compute_metrics(["log_oe"], [1, 0, 0, 1], [0.25, 0.25, 0.25, 0.25], 0.5)
    assert vals["log_oe"] == pytest.approx(np.log(2 / 1.0))
    vals, fails = compute_metrics(["log_oe"], [0, 0, 0], [0.2, 0.2, 0.2], 0.5)
    assert "log_oe" in fails and "undefined" in fails["log_oe"]
