"""Full-pipeline (refitting) participant-cluster bootstrap for TRAINING-DESIGN contrasts (label rule x anchoring).

Historical motivation: synthetic design experiments investigated conditional versus refitting uncertainty.
Numerical annotations are withheld because those historical diagnostics were not reproduced for this release.
Contrasts between training designs should account for model-refitting variability.

Design (one replicate):
  1. resample TRAINING participants with replacement (clusters: all of a sampled participant's anchors come along; each copy
     gets a distinct pseudo-identifier);
  2. for every configuration, rebuild its training rows from the resample (baseline-only or all eligible anchors, rule-specific
     labels, indeterminate rows dropped) and REFIT the complete pipeline from scratch (fresh estimator from `make_pipeline`);
  3. resample the FIXED evaluation participants with replacement (same resample for every configuration -> paired), keep their
     FIXED evaluation target, and compute metrics; differences versus the reference configuration are taken within the replicate.
The point estimate uses the original training participants and the original evaluation participants. Evaluation participants
never enter training. Failed replicates (single-class training rows, fit errors, undefined metrics) are counted, not hidden.

Contrasts: by default every configuration minus the reference. `contrasts` may instead name linear combinations of configurations
(e.g. the anchoring contrast A2 - A1 or the interaction (A2 - A1) - (A4 - A3)); each is computed inside every replicate, so its
interval is paired. Optional visit-weighted evaluation (a SEPARATE, secondary estimand): `visit_eval_frame` holds every eligible
anchor of the evaluation participants with its own target; the same refitted models score it, its participants are resampled as
clusters (all of a participant's anchors together) from a separate random stream, and its results are returned under "visit_weighted",
never mixed with the baseline results.
Pure functions; no file access. Synthetic-tested in tests/test_design_bootstrap.py.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from yoad.benchmark.metrics import METRICS


def training_rows(anchors: pd.DataFrame, rule: str, anchoring: str, positive: str = "EVENT", negative: str = "NONEVENT"):
    """Rows used to train one configuration: baseline anchors only or all eligible anchors, rule-determinate rows only."""
    if anchoring not in ("baseline", "all"):
        raise ValueError("anchoring must be 'baseline' or 'all'")
    t = anchors if anchoring == "all" else anchors[anchors["is_baseline"]]
    t = t[t[rule].isin([positive, negative])]
    return t, (t[rule] == positive).astype(int).to_numpy()


def _fit_all(train: pd.DataFrame, configs: dict, features: list, make_pipeline: Callable, seed: int):
    fitted = {}
    for name, (rule, anch) in configs.items():
        t, y = training_rows(train, rule, anch)
        if len(np.unique(y)) < 2:
            raise ValueError(f"configuration {name}: training labels contain a single class")
        fitted[name] = make_pipeline(name, seed).fit(t[features], y)
    return fitted


def _metrics(y, preds: dict, metrics, contrasts: dict):
    est, diff = {}, {}
    for c, p in preds.items():
        for m in metrics:
            est[(c, m)] = METRICS[m].fn(y, p)
    for name, coefs in contrasts.items():
        for m in metrics:
            diff[(name, m)] = sum(w * est[(c, m)] for c, w in coefs.items())
    return est, diff


def _cluster_index(groups: dict, ids, rng):
    """Row positions of a participant-cluster resample (all rows of each drawn participant)."""
    pick = rng.choice(ids, size=len(ids), replace=True)
    return np.concatenate([groups[pid] for pid in pick])


def _summ(point_est, point_diff, reps_est, reps_diff, configs, metrics, contrasts, reference, q):
    est_rows = [{"config": c, "metric": m, "point": point_est[(c, m)],
                 "ci_low": q([r[(c, m)] for r in reps_est])[0], "ci_high": q([r[(c, m)] for r in reps_est])[1]}
                for c in configs for m in metrics]
    diff_rows = []
    for name, coefs in contrasts.items():
        default = set(coefs) == {name, reference} and coefs[name] == 1 and coefs[reference] == -1
        for m in metrics:
            diff_rows.append({"contrast": name, "config": name if default else None, "reference": reference if default else None,
                              "coefficients": dict(coefs), "metric": m, "difference": point_diff[(name, m)],
                              "ci_low": q([r[(name, m)] for r in reps_diff])[0], "ci_high": q([r[(name, m)] for r in reps_diff])[1]})
    return est_rows, diff_rows


def design_contrast_bootstrap(train_anchors: pd.DataFrame, eval_frame: pd.DataFrame, target: str, configs: dict,
                              features: list, make_pipeline: Callable, reference: str, metrics=("roc_auc", "brier", "log_oe"),
                              B: int = 200, seed: int = 0, alpha: float = 0.05, on_fit: Callable | None = None,
                              contrasts: dict | None = None, visit_eval_frame: pd.DataFrame | None = None,
                              visit_target: str | None = None) -> dict:
    """train_anchors: eligible anchors of TRAINING participants (columns participant_id, is_baseline, rule columns, features).
    eval_frame: one row per EVALUATION participant (baseline), with the fixed binary `target` column and the features.
    configs: {name: (rule_column, 'baseline'|'all')}. make_pipeline(name, seed) must return a NEW unfitted estimator.
    on_fit (testing hook): called with (replicate, config_name, training_rows_frame) after each refit; the frame carries both the
    pseudo-identifier (participant_id) and the original identifier (orig_participant_id).
    contrasts: {name: {config: coefficient}}; default {c: {c: 1, reference: -1}} for every c != reference.
    visit_eval_frame / visit_target: optional visit-weighted evaluation (several rows per evaluation participant)."""
    if reference not in configs:
        raise ValueError("reference must be one of the configurations")
    if contrasts is None:
        contrasts = {c: {c: 1, reference: -1} for c in configs if c != reference}
    for name, coefs in contrasts.items():
        if not coefs or any(c not in configs for c in coefs):
            raise ValueError(f"contrast {name} refers to unknown configurations")
    overlap = set(train_anchors["participant_id"]) & set(eval_frame["participant_id"])
    if overlap:
        raise ValueError(f"{len(overlap)} evaluation participant(s) also appear in the training anchors")
    if eval_frame["participant_id"].duplicated().any():
        raise ValueError("eval_frame must have one row per evaluation participant")
    y_eval = eval_frame[target].to_numpy().astype(int)
    if len(np.unique(y_eval)) < 2:
        raise ValueError("evaluation target must contain both classes")
    vw = visit_eval_frame is not None
    if vw:
        if visit_target is None:
            raise ValueError("visit_target is required with visit_eval_frame")
        if set(train_anchors["participant_id"]) & set(visit_eval_frame["participant_id"]):
            raise ValueError("visit-weighted evaluation participants also appear in the training anchors")
        y_vis = visit_eval_frame[visit_target].to_numpy().astype(int)
        if len(np.unique(y_vis)) < 2:
            raise ValueError("visit-weighted evaluation target must contain both classes")
        v_groups = {pid: np.asarray(ix) for pid, ix in
                    visit_eval_frame.reset_index(drop=True).groupby("participant_id").indices.items()}
        v_ids = np.array(list(v_groups))
    metrics = list(metrics)
    # point estimate: original training participants, original evaluation participants
    fitted = _fit_all(train_anchors, configs, features, make_pipeline, seed)
    p0 = {c: f.predict_proba(eval_frame[features])[:, 1] for c, f in fitted.items()}
    point_est, point_diff = _metrics(y_eval, p0, metrics, contrasts)
    if vw:
        pv0 = {c: f.predict_proba(visit_eval_frame[features])[:, 1] for c, f in fitted.items()}
        v_point_est, v_point_diff = _metrics(y_vis, pv0, metrics, contrasts)
        v_reps_est, v_reps_diff, v_failures = [], [], []
        rng_v = np.random.default_rng([seed, 1])                       # own stream: baseline results do not depend on it

    rng = np.random.default_rng(seed)
    ids = train_anchors["participant_id"].unique()
    by_id = {pid: g for pid, g in train_anchors.groupby("participant_id")}
    n_eval = len(eval_frame)
    reps_est, reps_diff, failures = [], [], []
    for b in range(B):
        pick = rng.choice(ids, size=len(ids), replace=True)
        boot = pd.concat([by_id[pid].assign(orig_participant_id=pid, participant_id=f"b{k}") for k, pid in enumerate(pick)],
                         ignore_index=True)
        e_idx = rng.integers(0, n_eval, n_eval)
        try:
            fitted_b = {}
            for name, (rule, anch) in configs.items():
                t, y = training_rows(boot, rule, anch)
                if len(np.unique(y)) < 2:
                    raise ValueError(f"single-class training labels in {name}")
                fitted_b[name] = make_pipeline(name, seed + b + 1).fit(t[features], y)
                if on_fit is not None:
                    on_fit(b, name, t)
            ye = y_eval[e_idx]
            if len(np.unique(ye)) < 2:
                raise ValueError("single-class evaluation resample")
            Xe = eval_frame[features].iloc[e_idx]
            pb = {c: f.predict_proba(Xe)[:, 1] for c, f in fitted_b.items()}
            est, diff = _metrics(ye, pb, metrics, contrasts)
            if not all(np.isfinite(v) for v in list(est.values()) + list(diff.values())):
                raise ValueError("non-finite metric")
            reps_est.append(est); reps_diff.append(diff)
        except Exception as e:                                         # recorded, never silently replaced
            failures.append({"replicate": b, "error": f"{type(e).__name__}: {str(e)[:160]}"})
            continue
        if vw:                                                         # same refitted models, independent cluster draw
            try:
                v_idx = _cluster_index(v_groups, v_ids, rng_v)
                yv = y_vis[v_idx]
                if len(np.unique(yv)) < 2:
                    raise ValueError("single-class visit-weighted resample")
                Xv = visit_eval_frame[features].iloc[v_idx]
                vest, vdiff = _metrics(yv, {c: f.predict_proba(Xv)[:, 1] for c, f in fitted_b.items()}, metrics, contrasts)
                if not all(np.isfinite(v) for v in list(vest.values()) + list(vdiff.values())):
                    raise ValueError("non-finite metric")
                v_reps_est.append(vest); v_reps_diff.append(vdiff)
            except Exception as e:
                v_failures.append({"replicate": b, "error": f"{type(e).__name__}: {str(e)[:160]}"})
    q = lambda vals: (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))) if vals else (None, None)
    est_rows, diff_rows = _summ(point_est, point_diff, reps_est, reps_diff, configs, metrics, contrasts, reference, q)
    out = {"method": "full-pipeline participant-cluster bootstrap (training refit + paired evaluation resample)",
           "B_requested": B, "B_used": len(reps_est), "failures": failures, "n_train_participants": int(len(ids)),
           "n_eval_participants": int(n_eval), "confidence": 1 - alpha, "estimates": est_rows, "paired_differences": diff_rows}
    if vw:
        v_est_rows, v_diff_rows = _summ(v_point_est, v_point_diff, v_reps_est, v_reps_diff, configs, metrics, contrasts, reference, q)
        out["visit_weighted"] = {"estimand": "all eligible anchors of the evaluation participants, equally weighted; "
                                             "participant-cluster resampling; secondary, never compared with the baseline estimand",
                                 "B_used": len(v_reps_est), "failures": v_failures, "n_eval_participants": int(len(v_ids)),
                                 "n_eval_anchors": int(len(visit_eval_frame)), "estimates": v_est_rows,
                                 "paired_differences": v_diff_rows}
    return out
