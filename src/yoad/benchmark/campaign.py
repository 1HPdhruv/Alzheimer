"""Two-pass screening campaign over many candidate configurations, on ONE dataset with ONE set of participant partitions.

  0. A locked test partition of participants is set aside once (stratified, both classes required). It is not touched until
     Pass B, and the Pass B selection cannot depend on it (tested by permuting locked-test labels).
  Pass A (screening): every candidate runs the same participant-grouped k-fold CV inside the DEVELOPMENT participants; pooled
     out-of-fold predictions give the screening metric. Candidates that fail are recorded, not dropped.
  Selection (pre-declared in the config): rank successful candidates by the screening metric on development OOF predictions and
     keep `top_k`, plus the declared reference candidate.
  Pass B (confirmation): each finalist is refitted on all development participants and evaluated ONCE on the locked test set;
     participant-level cluster bootstrap gives CIs and paired differences versus the reference candidate.
Only aggregate results are written (no predictions, identifiers or feature values). Candidate configs differ only in modelling
choices; dataset, seed and partitions are shared, so comparisons (including modality ablations) are paired.

CLI:  .venv-data/bin/python -m src.benchmark.campaign configs/benchmark/synthetic_screening_campaign.yaml
"""
from __future__ import annotations

import copy
import csv
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from yoad.benchmark.bootstrap import cluster_bootstrap
from yoad.benchmark.config import ConfigError, config_hash, load_config
from yoad.benchmark.data import GovernanceError, load_dataset
from yoad.benchmark.metrics import METRICS, compute_metrics
from yoad.benchmark.runner import REPO_ROOT, RunExistsError, _partition_fingerprint, fit_predict_fold
from yoad.benchmark.splitting import check_disjoint, group_kfold_splits, group_partition

CAMPAIGN_KEYS = {"campaign_id", "description", "seed", "dataset", "locked_test_fraction", "screening", "selection",
                 "bootstrap", "metrics", "base", "candidates", "output"}
OVERRIDABLE = {"features", "fusion", "preprocessing", "feature_selection", "sampling", "model", "calibration", "threshold",
               "validation"}


def _deep_update(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in ("params", "search_space") else v
    return out


def load_campaign(path_or_dict) -> dict:
    if isinstance(path_or_dict, dict):
        raw = path_or_dict
    else:
        with open(path_or_dict) as f:                              # the campaign YAML is the only file read
            raw = yaml.safe_load(f)
    unknown = set(raw) - CAMPAIGN_KEYS
    if unknown:
        raise ConfigError(f"unknown campaign keys {sorted(unknown)}")
    for k in ("campaign_id", "dataset", "screening", "selection", "candidates", "metrics"):
        if k not in raw:
            raise ConfigError(f"campaign needs '{k}'")
    c = {"seed": 0, "locked_test_fraction": 0.25, "bootstrap": {"B": 500}, "base": {}, "output": {"root": "results/benchmark"},
         "description": "", **raw}
    lt = c["locked_test_fraction"]
    if not (isinstance(lt, (int, float)) and 0.1 <= lt <= 0.5):
        raise ConfigError("locked_test_fraction must be in [0.1, 0.5]")
    n_splits = c["screening"].get("n_splits")
    if not (isinstance(n_splits, int) and n_splits >= 2):
        raise ConfigError("screening.n_splits must be an integer >= 2")
    sel = c["selection"]
    if sel.get("metric") not in {"log_loss", "brier", "roc_auc", "average_precision"}:
        raise ConfigError("selection.metric must be log_loss, brier, roc_auc or average_precision")
    if not (isinstance(sel.get("top_k"), int) and sel["top_k"] >= 1):
        raise ConfigError("selection.top_k must be a positive integer")
    names = [x.get("name") for x in c["candidates"]]
    if len(set(names)) != len(names) or not all(isinstance(n, str) and n for n in names):
        raise ConfigError("every candidate needs a unique non-empty name")
    if sel.get("reference") not in names:
        raise ConfigError("selection.reference must name one of the candidates")
    for cand in c["candidates"]:
        bad = set(cand.get("overrides", {})) - OVERRIDABLE
        if bad:
            raise ConfigError(f"candidate {cand['name']!r} may not override {sorted(bad)} (dataset/seed/output are shared)")
    c["_candidate_cfgs"] = {}
    for cand in c["candidates"]:
        cc = _deep_update(c["base"], cand.get("overrides", {}))
        cc.update({"experiment_id": cand["name"], "seed": c["seed"], "dataset": c["dataset"], "metrics": c["metrics"],
                   "output": c["output"]})
        cc.setdefault("validation", {})
        cc["validation"] = {**cc["validation"], "outer": {"type": "group_kfold", "n_splits": n_splits}}
        c["_candidate_cfgs"][cand["name"]] = load_config(cc)       # full validation of every candidate before any data exist
    c["metrics_prob"] = [m for m in c["metrics"] if METRICS[m].kind == "probability"]
    if c["selection"]["metric"] not in c["metrics_prob"]:
        raise ConfigError("selection.metric must also be listed in metrics")
    return c


def _val_rel(cfg, n_splits):
    """Validation partition as the same share of the fitting participants in Pass A and Pass B."""
    vf = cfg["validation"]["validation_fraction"]
    return vf / (1 - 1 / n_splits) if vf else None


def _score(metric, y, p):
    v = METRICS[metric].fn(np.asarray(y).astype(int), p)
    return -v if metric in {"log_loss", "brier"} else v


def run_campaign_on(c: dict, data) -> dict:
    """Core campaign on an already-loaded (synthetic) bundle. Returns the aggregate record."""
    y, g = data.y.to_numpy(), data.groups.to_numpy()
    seed = c["seed"]
    part = group_partition(y, g, {"development": 1 - c["locked_test_fraction"], "locked_test": c["locked_test_fraction"]},
                           seed, True, require_both_classes=("development", "locked_test"))
    dev, test = part["development"], part["locked_test"]
    check_disjoint(part, g)
    n_splits = c["screening"]["n_splits"]
    folds = [(dev[tr], dev[te]) for tr, te in group_kfold_splits(y[dev], g[dev], n_splits, seed + 1)]
    rec = {"partitions": {"development": {"n_participants": int(len(set(g[dev]))), "n_observations": int(dev.size),
                                          "participants_sha256": _partition_fingerprint(g, dev)},
                          "locked_test": {"n_participants": int(len(set(g[test]))), "n_observations": int(test.size),
                                          "participants_sha256": _partition_fingerprint(g, test)}},
           "screening": [], "selection": None, "confirmation": None}
    oof = {}
    for name, cfg in c["_candidate_cfgs"].items():
        t0 = time.perf_counter()
        row = {"candidate": name, "model_id": cfg["model"]["id"], "feature_set": "+".join(cfg["features"]["groups"] or ["all"]),
               "fusion": cfg["fusion"]["type"], "sampling": cfg["sampling"]["method"], "config_hash": config_hash(cfg)[:16]}
        try:
            data_c = data.select_groups(cfg["features"]["groups"]) if cfg["features"]["groups"] else data
            p = np.full(len(y), np.nan)
            val_rel = _val_rel(cfg, n_splits)
            for k, (tr, te) in enumerate(folds):
                _, p[te] = fit_predict_fold(cfg, data_c, tr, te, seed, k, val_rel=val_rel)
            vals, fails = compute_metrics(c["metrics_prob"], y[dev], p[dev], 0.5)   # pooled OOF: probability metrics only
            row.update({"status": "success", **{f"oof_{m}": v for m, v in vals.items()},
                        "failed_metrics": ";".join(sorted(fails)), "screen_score": _score(c["selection"]["metric"], y[dev], p[dev])})
            oof[name] = p
        except Exception as e:
            row.update({"status": "failed", "error": f"{type(e).__name__}: {str(e)[:200]}",
                        "where": traceback.extract_tb(e.__traceback__)[-1].name})
        row["runtime_seconds"] = round(time.perf_counter() - t0, 2)
        rec["screening"].append(row)
    ok = [r for r in rec["screening"] if r["status"] == "success"]
    ranked = sorted(ok, key=lambda r: (-r["screen_score"], r["candidate"]))           # ties broken by name, deterministically
    ref = c["selection"]["reference"]
    finalists = [r["candidate"] for r in ranked[: c["selection"]["top_k"]]]
    if ref not in finalists and ref in oof:
        finalists.append(ref)
    rec["selection"] = {"rule": f"top {c['selection']['top_k']} by development OOF {c['selection']['metric']} + reference",
                        "ranking": [r["candidate"] for r in ranked], "finalists": finalists, "reference": ref,
                        "reference_failed": ref not in oof}
    if ref in oof:
        dev_boot = cluster_bootstrap(y[dev], {n: oof[n][dev] for n in oof}, g[dev], c["metrics_prob"],
                                     B=c["bootstrap"]["B"], seed=seed + 2, reference=ref)
        rec["screening_paired_vs_reference"] = dev_boot["paired_differences"]
    preds_test, final_rows = {}, []
    for name in finalists:
        cfg = c["_candidate_cfgs"][name]
        data_c = data.select_groups(cfg["features"]["groups"]) if cfg["features"]["groups"] else data
        val_rel = _val_rel(cfg, n_splits)
        fold_rec, p_te = fit_predict_fold(cfg, data_c, dev, test, seed, 0, val_rel=val_rel)
        preds_test[name] = p_te
        final_rows.append({"candidate": name, "threshold": fold_rec["threshold"], "metrics": fold_rec["metrics"],
                           "failed_metrics": fold_rec["failed_metrics"], "selected_params": fold_rec["selected_params"]})
    boot = cluster_bootstrap(y[test], preds_test, g[test], c["metrics_prob"], B=c["bootstrap"]["B"], seed=seed + 3,
                             reference=ref if ref in preds_test else None)
    rec["confirmation"] = {"finalists": final_rows, "bootstrap": boot,
                           "locked_test_prevalence_observations": float(y[test].mean())}
    return rec


def run_campaign(path_or_dict, results_base: Path | None = None, run_id: str | None = None) -> dict:
    c = load_campaign(path_or_dict)
    chash = config_hash({k: v for k, v in c.items() if not k.startswith("_") and k != "metrics_prob"})
    ts = datetime.now(timezone.utc)
    run_id = run_id or f"{ts.strftime('%Y%m%dT%H%M%SZ')}_{chash[:10]}"
    base = Path(results_base or REPO_ROOT)
    out = base / c["output"]["root"] / c["campaign_id"] / run_id
    if out.exists():
        raise RunExistsError(f"campaign run already exists: {out.relative_to(base)}")
    out.mkdir(parents=True)
    rec = {"campaign_id": c["campaign_id"], "run_id": run_id, "timestamp_utc": ts.isoformat(), "config_hash": chash,
           "seed": c["seed"], "status": "running", "synthetic_data": None, "error": None}
    t0 = time.perf_counter()
    try:
        data = load_dataset(c["dataset"].get("adapter", "synthetic"), c["dataset"].get("params", {}))
        if not data.synthetic:
            raise GovernanceError("non-synthetic data reached the campaign")
        rec["synthetic_data"], rec["data_provenance"] = True, data.provenance
        rec.update(run_campaign_on(c, data))
        rec["status"] = "success"
    except Exception as e:
        rec["status"] = "failed"
        rec["error"] = {"type": type(e).__name__, "message": str(e)[:500]}
    rec["runtime_seconds"] = round(time.perf_counter() - t0, 2)
    with open(out / "campaign.resolved.yaml", "w") as f:
        yaml.safe_dump({k: v for k, v in c.items() if not k.startswith("_")}, f, sort_keys=True)
    with open(out / "campaign_result.json", "w") as f:
        json.dump(rec, f, indent=2, sort_keys=True, default=str)
    if rec.get("screening"):
        cols = sorted({k for r in rec["screening"] for k in r})
        with open(out / "screening.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rec["screening"])
    if rec.get("confirmation"):
        with open(out / "confirmation.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["model", "metric", "point", "ci_low", "ci_high"])
            for e in rec["confirmation"]["bootstrap"]["estimates"]:
                w.writerow([e["model"], e["metric"], e["point"], e["ci_low"], e["ci_high"]])
        with open(out / "paired_differences.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["model", "reference", "metric", "difference", "ci_low", "ci_high"])
            for e in rec["confirmation"]["bootstrap"]["paired_differences"]:
                w.writerow([e["model"], e["reference"], e["metric"], e["difference"], e["ci_low"], e["ci_high"]])
    return rec


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m src.benchmark.campaign <campaign.yaml>")
    r = run_campaign(sys.argv[1])
    summ = {"run_id": r["run_id"], "status": r["status"], "error": r["error"]}
    if r["status"] == "success":
        summ["screening"] = [(s["candidate"], s["status"], round(s.get("screen_score", float("nan")), 4)) for s in r["screening"]]
        summ["finalists"] = r["selection"]["finalists"]
    print(json.dumps(summ, indent=1, default=str))
    sys.exit(0 if r["status"] == "success" else 1)
