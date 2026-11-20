"""Experiment runner and structured logging.

Flow per outer fold:  outer split (participant-grouped)  ->  optional validation partition carved from the outer-training
groups  ->  optional grid search by inner grouped CV on the fitting partition  ->  refit on the fitting partition  ->
calibration and threshold fitted on the validation partition  ->  metrics on the outer evaluation partition.
The evaluation partition is used once, for metrics only.

Logs contain configuration, counts, selected settings and metrics. They never contain feature values, labels or
identifiers of observations. Each run gets its own directory; existing directories are never overwritten.

CLI:  .venv-data/bin/python -m src.benchmark.runner configs/benchmark/synthetic_smoke_test.yaml
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
import yaml

from yoad.benchmark.config import config_hash, load_config
from yoad.benchmark.data import GovernanceError, load_dataset
from yoad.benchmark.metrics import METRICS, compute_metrics, fit_calibrator, select_threshold
from sklearn.linear_model import LogisticRegression

from yoad.benchmark.pipeline import assert_fit_rows, build_pipeline
from yoad.benchmark.splitting import check_disjoint, group_kfold_splits, group_partition

REPO_ROOT = Path.cwd()  # public package: invocation-relative outputs
INDEX_FIELDS = ["run_id", "experiment_id", "timestamp_utc", "status", "model_id", "feature_set_id", "config_hash",
                "primary_summary"]


class RunExistsError(FileExistsError):
    pass


def _counts(y, groups, idx) -> dict:
    yy = np.asarray(y)[idx]
    return {"n_observations": int(idx.size), "n_participants": int(len(set(np.asarray(groups)[idx]))),
            "n_positive_observations": int(yy.sum())}


def _grid(space: dict) -> list[dict]:
    if not space:
        return [{}]
    keys = sorted(space)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(space[k] for k in keys))]


def _score(name, y, p) -> float:
    """Higher-is-better score for model selection."""
    v = METRICS[name].fn(np.asarray(y).astype(int), p)
    return -v if name in {"log_loss", "brier"} else v


def _search(cfg, data, idx_fit, seed) -> tuple[dict, list]:
    """Grid search by inner participant-grouped CV restricted to the fitting partition."""
    m, v = cfg["model"], cfg["validation"]
    candidates = _grid(m["search_space"])
    if len(candidates) == 1:
        return {**m["params"], **candidates[0]}, []
    X, y, g = data.X.iloc[idx_fit], data.y.iloc[idx_fit].to_numpy(), data.groups.iloc[idx_fit].to_numpy()
    folds = group_kfold_splits(y, g, v["inner_cv_splits"], seed + 1, v["stratify"])
    trace = []
    for cand in candidates:
        params = {**m["params"], **cand}
        scores = []
        for tr, te in folds:
            pipe = _fit_guarded(cfg, data.features, params, seed, X, y, tr, te)
            scores.append(_score(m["search_metric"], y[te], pipe.predict_proba(X.iloc[te])[:, 1]))
        trace.append({"params": cand, "mean_inner_score": float(np.mean(scores))})
    best = max(range(len(trace)), key=lambda i: trace[i]["mean_inner_score"])     # ties -> first in sorted grid
    return {**m["params"], **candidates[best]}, trace


def _fit_guarded(cfg, features, params, seed, X, y, fit_pos, heldout_pos):
    """Fit a fresh pipeline on positional rows `fit_pos` of X and verify (FitGuard) it saw exactly those rows and none of
    `heldout_pos`. Raises LeakageError otherwise."""
    pipe = build_pipeline(cfg, features, params, seed).fit(X.iloc[fit_pos], np.asarray(y)[fit_pos])
    assert_fit_rows(pipe, X.index[fit_pos], X.index[heldout_pos])
    return pipe


def _partition_fingerprint(groups, idx) -> str:
    """Hash of the set of participants in a partition (no identifier is logged; equal hashes = identical partitions)."""
    return hashlib.sha256("\n".join(sorted(set(np.asarray(groups)[idx].astype(str)))).encode()).hexdigest()[:16]


def _late_fusion(cfg, data, idx_fit, idx_heldout, params, seed):
    """Late fusion: one pipeline per feature group; a logistic meta-learner on logit base predictions. The meta-learner is
    trained only on participant-grouped OUT-OF-FOLD predictions inside the fitting partition (each base prediction comes
    from a model that did not see that row, verified by FitGuard). Base models are then refitted on the whole fitting
    partition. Returns (predict function over positional rows, info)."""
    y, g = data.y.to_numpy(), data.groups.to_numpy()
    names = data.feature_groups()
    folds = group_kfold_splits(y[idx_fit], g[idx_fit], cfg["fusion"]["inner_cv_splits"], seed + 7, cfg["validation"]["stratify"])
    Z = np.full((idx_fit.size, len(names)), np.nan)
    for j, name in enumerate(names):
        sub = data.select_groups([name])
        for tr, te in folds:
            pipe = _fit_guarded(cfg, sub.features, params, seed, sub.X, y, idx_fit[tr], np.r_[idx_fit[te], idx_heldout])
            Z[te, j] = pipe.predict_proba(sub.X.iloc[idx_fit[te]])[:, 1]
    if np.isnan(Z).any():
        raise RuntimeError("out-of-fold matrix incomplete")
    meta = LogisticRegression(C=1.0, max_iter=5000, random_state=seed).fit(_logit(Z), y[idx_fit])
    bases = [(data.select_groups([n]), _fit_guarded(cfg, data.select_groups([n]).features, params, seed,
                                                    data.select_groups([n]).X, y, idx_fit, idx_heldout)) for n in names]

    def predict(pos):
        return meta.predict_proba(_logit(np.column_stack([p.predict_proba(b.X.iloc[pos])[:, 1] for b, p in bases])))[:, 1]
    info = {"base_groups": names, "meta_coefficients": dict(zip(names, meta.coef_[0].round(6).tolist())),
            "oof_inner_splits": len(folds)}
    return predict, bases[0][1].named_steps["model"], info


def _logit(p):
    q = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def _outer_splits(cfg, y, g, seed):
    v, o = cfg["validation"], cfg["validation"]["outer"]
    if o["type"] == "group_holdout":
        p = group_partition(y, g, {"train": 1 - o["test_fraction"], "test": o["test_fraction"]}, seed, v["stratify"],
                            require_both_classes=("test",))
        return [(p["train"], p["test"])]
    return group_kfold_splits(y, g, o["n_splits"], seed, v["stratify"])


def run_fold(cfg, data, tr, te, seed, fold) -> dict:
    return fit_predict_fold(cfg, data, tr, te, seed, fold)[0]


def fit_predict_fold(cfg, data, tr, te, seed, fold, val_rel: float | None = None):
    """As run_fold, but also returns the evaluation-partition predictions (IN MEMORY ONLY; never written to disk).
    val_rel: validation share of the outer-training groups (default derived from the configuration)."""
    v = cfg["validation"]
    y, g = data.y.to_numpy(), data.groups.to_numpy()
    if v["validation_fraction"] > 0:
        vf = val_rel if val_rel is not None else v["validation_fraction"] / (
            1 - (v["outer"]["test_fraction"] if v["outer"]["type"] == "group_holdout" else 1 / v["outer"]["n_splits"]))
        inner = group_partition(y[tr], g[tr], {"fit": 1 - vf, "validation": vf}, seed + 100 + fold, v["stratify"],
                                require_both_classes=("validation",))
        idx_fit, idx_val = tr[inner["fit"]], tr[inner["validation"]]
    else:
        idx_fit, idx_val = tr, np.array([], dtype=int)
    parts = {"fit": idx_fit, "evaluation": te, **({"validation": idx_val} if idx_val.size else {})}
    check_disjoint(parts, g)
    heldout = np.r_[te, idx_val]
    fusion_info = None
    if cfg["fusion"]["type"] == "late":
        params, trace = {**cfg["model"]["params"]}, []
        predict, est, fusion_info = _late_fusion(cfg, data, idx_fit, heldout, params, seed)
    else:
        params, trace = _search(cfg, data, idx_fit, seed + fold)
        pipe = _fit_guarded(cfg, data.features, params, seed, data.X, y, idx_fit, heldout)
        est = pipe.named_steps["model"]
        predict = lambda pos: pipe.predict_proba(data.X.iloc[pos])[:, 1]
    cal = lambda p: p
    thr = float(cfg["threshold"]["value"]) if cfg["threshold"]["method"] == "fixed" else None
    if idx_val.size:
        p_val = predict(idx_val)
        cal = fit_calibrator(cfg["calibration"]["method"], y[idx_val], p_val, seed)
        thr = select_threshold(cfg["threshold"]["method"], cfg["threshold"]["value"], y[idx_val], cal(p_val))
    p_te = cal(predict(te))
    vals, fails = compute_metrics(cfg["metrics"], y[te], p_te, thr)
    return {"fold": fold, "partitions": {k: {**_counts(y, g, idx), "participants_sha256": _partition_fingerprint(g, idx)}
                                         for k, idx in parts.items()},
            "fit_guard": "passed", "fusion": fusion_info,
            "selected_params": params, "search_trace": trace, "threshold": thr,
            "resampling": {"rows_into_sampler": int(est.n_input_rows_), "rows_after_sampler": int(est.n_resampled_rows_)},
            "metrics": vals, "failed_metrics": fails,
            "prediction_sha256": hashlib.sha256(np.round(p_te, 10).tobytes()).hexdigest()[:16]}, p_te


def run(config, results_base: Path | None = None, run_id: str | None = None) -> dict:
    """Run one configured experiment. `results_base` (code-level, default: repo root) anchors output.root."""
    cfg = load_config(config)                                    # ConfigError before anything is created
    chash = config_hash(cfg)
    ts = datetime.now(timezone.utc)
    run_id = run_id or f"{ts.strftime('%Y%m%dT%H%M%SZ')}_{chash[:10]}"
    out = Path(results_base or REPO_ROOT) / cfg["output"]["root"] / cfg["experiment_id"] / run_id
    if out.exists():
        raise RunExistsError(f"run directory already exists; refusing to overwrite: {out.relative_to(Path(results_base or REPO_ROOT))}")
    out.mkdir(parents=True)
    groups_used = cfg["features"]["groups"]
    record = {
        "run_id": run_id, "experiment_id": cfg["experiment_id"], "timestamp_utc": ts.isoformat(),
        "synthetic_data": None, "data_provenance": None, "config_hash": chash, "seed": cfg["seed"],
        "task": cfg["task"], "model_id": cfg["model"]["id"],
        "feature_set_id": "+".join(sorted(groups_used)) if groups_used else "all",
        "fusion_type": cfg["fusion"]["type"],
        "preprocessing": cfg["preprocessing"], "feature_selection": cfg["feature_selection"], "sampling": cfg["sampling"],
        "split_strategy": cfg["validation"], "calibration": cfg["calibration"], "threshold_rule": cfg["threshold"],
        "folds": [], "summary": {}, "failed_metrics": {}, "status": "running", "error": None,
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "scikit_learn": sklearn.__version__,
                        "platform": platform.platform()},
    }
    with open(out / "config.resolved.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)
    t0 = time.perf_counter()
    try:
        data = load_dataset(cfg["dataset"]["adapter"], cfg["dataset"]["params"])
        if not data.synthetic:
            raise GovernanceError("non-synthetic data reached the runner")
        record["synthetic_data"], record["data_provenance"] = True, data.provenance
        if groups_used:
            data = data.select_groups(groups_used)
        y, g = data.y.to_numpy(), data.groups.to_numpy()
        for k, (tr, te) in enumerate(_outer_splits(cfg, y, g, cfg["seed"])):
            record["folds"].append(run_fold(cfg, data, tr, te, cfg["seed"], k))
        for name in cfg["metrics"]:
            vals = [f["metrics"][name] for f in record["folds"] if name in f["metrics"]]
            failed = [f["fold"] for f in record["folds"] if name in f["failed_metrics"]]
            record["summary"][name] = {"mean": float(np.mean(vals)) if vals else None,
                                       "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                                       "n_folds": len(vals)}
            if failed:
                record["failed_metrics"][name] = {"folds": failed,
                                                  "reasons": sorted({record["folds"][i]["failed_metrics"][name] for i in failed})}
        record["status"] = "success" if not record["failed_metrics"] else "success_with_failed_metrics"
    except Exception as e:
        record["status"] = "failed"
        record["error"] = {"type": type(e).__name__, "message": str(e)[:500],
                           "where": traceback.extract_tb(e.__traceback__)[-1].name}
    record["runtime_seconds"] = round(time.perf_counter() - t0, 3)
    _write(out, record)
    return record


def _write(out: Path, record: dict) -> None:
    with open(out / "result.json", "w") as f:
        json.dump(record, f, indent=2, sort_keys=True, default=str)
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "metric", "value", "status", "reason"])
        for fold in record["folds"]:
            for k, v in fold["metrics"].items():
                w.writerow([fold["fold"], k, v, "ok", ""])
            for k, v in fold["failed_metrics"].items():
                w.writerow([fold["fold"], k, "", "failed", v])
    index = out.parent / "runs_index.csv"                        # per-experiment append-only index, incl. failures
    new = not index.exists()
    first = next(iter(record["summary"]), None)
    with open(index, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        if new:
            w.writeheader()
        w.writerow({"run_id": record["run_id"], "experiment_id": record["experiment_id"],
                    "timestamp_utc": record["timestamp_utc"], "status": record["status"], "model_id": record["model_id"],
                    "feature_set_id": record["feature_set_id"], "config_hash": record["config_hash"][:16],
                    "primary_summary": f"{first}={record['summary'][first]['mean']}" if first else ""})


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m src.benchmark.runner <config.yaml>")
    r = run(sys.argv[1])
    print(json.dumps({"run_id": r["run_id"], "status": r["status"], "summary": r["summary"],
                      "failed_metrics": r["failed_metrics"], "error": r["error"]}, indent=2))
    sys.exit(0 if r["status"].startswith("success") else 1)
