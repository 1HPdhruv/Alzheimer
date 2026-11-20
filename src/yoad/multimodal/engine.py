"""Nested participant-level cross-validation, stage orchestration and the locked held-out evaluation (PROTOCOL.md §§5-9).

Development data = baseline anchors of the frozen TRAINING partition only. Every candidate is scored by repeated (3 x 5-fold)
outer cross-validation on identical participant folds; everything learned (imputation, scaling, tuning over the prespecified
grid by inner 3-fold CV, Platt calibration on inner out-of-fold predictions, the Youden threshold, stacking, resampling) uses the
outer-training rows only, and a run-time check (FitGuard / fitted_rows_) verifies after every fit that no outer-test row was used.
Outer-test labels never reach a worker. The frozen held-out partition is used once, after the selection is written: the winner,
the frozen references (exact reproduction of A3 against the stored estimates), the transportable model and the MRI-subset
winner are refitted on the development data and scored on the held-out REF-determinate baseline participants.

Pure: inputs are in-memory tables; outputs are aggregate tables, plus out-of-fold arrays that the caller may only keep in a
PRIVATE participant-level folder. Synthetic-tested in multimodal_benchmark/tests.
"""
from __future__ import annotations

import hashlib
import json
import math
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from yoad.multimodal import metrics as MT
from yoad.multimodal import models as M
from yoad.multimodal import registry as R
from yoad.multimodal import selection as S
from yoad.multimodal.cohort import RISK_HORIZON_DAYS
from yoad.data.reference_outcome import EVENT, NONEVENT
from yoad.evaluation.safeguards import LeakageError

PLATT_MODELS = ("svm_rbf", "xgb_tuned_platt")


class Population:
    def __init__(self, name, frame, y, strict=None, survival=None):
        self.name, self.frame, self.strict, self.survival = name, frame, strict, survival
        self.y = pd.Series(np.asarray(y, int), index=frame.index)


def build_populations(rows: dict, landmark: pd.DataFrame | None = None, survival: pd.DataFrame | None = None) -> dict:
    prim, strict = rows["primary"], rows["strict"]
    dev = prim[(prim["partition"] == "train") & prim["is_baseline"]]
    lab = dev[dev["state_REF"].isin([EVENT, NONEVENT])].set_index("participant_id").sort_index()
    sdev = strict[(strict["partition"] == "train") & strict["is_baseline"]].set_index("participant_id")
    y = (lab["state_REF"] == EVENT).astype(int).to_numpy()
    pops = {"P_ALL": Population("P_ALL", lab, y, sdev.loc[lab.index])}
    for pop, prefix in R.INSTRUMENT_OF_POPULATION.items():
        col = f"{prefix}_date"
        if col in lab.columns:
            m = lab[col].notna().to_numpy()
            pops[pop] = Population(pop, lab[m], y[m], sdev.loc[lab.index[m]])
    if landmark is not None and len(landmark):
        lm = landmark[landmark["state_REF"].isin([EVENT, NONEVENT])].set_index("participant_id").sort_index()
        pops["P_LM"] = Population("P_LM", lm, (lm["state_REF"] == EVENT).astype(int).to_numpy())
    if survival is not None and len(survival):
        sv = survival.set_index("participant_id")
        pops["P_ALL"].survival = {"primary": sv.join(dev.set_index("participant_id"), how="inner").sort_index(),
                                  "strict": sv.join(sdev, how="inner").sort_index()}
    return pops


def _cox_frame(pop: Population, timing: str, feats: list, keep=None) -> pd.DataFrame:
    sv = pop.survival[timing]
    sv = sv if keep is None else sv.loc[keep]
    return sv[feats].assign(time_days=sv["time_days"], event=sv["event"], _state=sv["state_REF"].astype(str))


def gate_table(pops: dict) -> pd.DataFrame:
    rows = []
    for name, p in pops.items():
        e, n = int(p.y.sum()), int(len(p.y) - p.y.sum())
        ok = e >= R.MIN_EVENTS and n >= R.MIN_NONEVENTS
        rows.append({"population": name, "participants": MT.sup(len(p.y)), "events": MT.sup(e), "nonevents": MT.sup(n),
                     "status": ("ESTIMATION_ONLY" if e < R.FULL_FROM_EVENTS else "FULL_ANALYSIS") if ok else "STOP",
                     "_events": e, "_nonevents": n})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ candidate preparation
def prepare(cand: dict, pops: dict, mri_full=()) -> tuple[dict | None, dict]:
    """Resolves features (coverage rule), participants (complete case), groups and the event gate. Returns (spec, record)."""
    rec = {k: cand[k] for k in ("id", "stage", "population", "feature_set", "model", "imputation", "timing")}
    pop = pops.get(cand["population"])
    if pop is None:
        return None, rec | {"status": "skipped", "reason": "population unavailable"}
    frame = pop.frame if cand["timing"] == "primary" else pop.strict
    wanted = R.resolve(cand["feature_set"], mri_full)
    dropped, feats = [], []
    for f in wanted:
        if f not in frame.columns:
            if f in R.FROZEN_FEATURES:
                raise ValueError(f"frozen feature {f} missing from the analysis frame")
            dropped.append(f"{f} (not available)"); continue
        miss = float(frame[f].isna().mean())
        if f not in R.FROZEN_FEATURES and miss > R.MAX_MISSING_FRACTION:
            dropped.append(f"{f} ({miss:.0%} missing)"); continue
        feats.append(f)
    groups = {}
    for g in R.groups_of(cand["feature_set"]):
        if g == "LONGITUDINAL":
            names = list(feats)
        elif g == "MRIFULL":
            names = [f for f in feats if f.startswith("fs_")]
        else:
            names = [f for f in feats if f in R.GROUPS[g]]
        if not names:
            return None, rec | {"status": "skipped", "reason": f"modality {g} unavailable after the coverage rule",
                                "features_dropped": "; ".join(dropped)}
        groups[g] = names
    ids = frame.index
    if cand["imputation"] == "complete_case":
        ids = frame.index[frame[feats].notna().all(axis=1).to_numpy()]
    y = pop.y.loc[ids]
    e, n = int(y.sum()), int(len(y) - y.sum())
    rec.update(n_features=len(feats), features_dropped="; ".join(dropped), participants=MT.sup(len(ids)), events=MT.sup(e),
               nonevents=MT.sup(n))
    if e < R.MIN_EVENTS or n < R.MIN_NONEVENTS:
        return None, rec | {"status": "skipped", "reason": f"fewer than {R.MIN_EVENTS} events or non-events"}
    if cand["model"] == "lr_interact" and len(feats) > R.INTERACTION_MAX_BASE:
        return None, rec | {"status": "skipped", "reason": f"more than {R.INTERACTION_MAX_BASE} base features"}
    if cand["model"] == "cox_l2" and pop.survival is None:
        return None, rec | {"status": "skipped", "reason": "no follow-up data for the censoring-aware model"}
    spec = {**{k: cand[k] for k in ("id", "stage", "population", "feature_set", "model", "imputation", "timing")},
            "features": feats, "groups": groups, "ids": np.asarray(ids)}
    return spec, rec | {"status": "prepared"}


def fold_maps(folds: pd.DataFrame) -> dict:
    return {r: g.set_index("participant_id")["fold"] for r, g in folds.groupby("repeat")}


def task_settings(base_model: str) -> dict:
    """Everything a worker needs from the registry, resolved in the parent process (workers never read module settings)."""
    m = R.MODELS[base_model]
    return {"configs": [m["fixed"]] if "fixed" in m else (R.grid_configs(base_model) if "grid" in m else []),
            "tune_metric": m.get("tune_metric", "log_loss"), "inner_splits": R.INNER_SPLITS,
            "search_n_iter": m.get("n_iter"), "search_inner_splits": m.get("inner_splits")}


def build_tasks(spec: dict, pop: Population, fmaps: dict, platt_also: bool = False) -> list[dict]:
    frame = pop.frame if spec["timing"] == "primary" else pop.strict
    feats, ids = spec["features"], spec["ids"]
    base_model = "xgb_tuned" if spec["model"] == "xgb_tuned_platt" else spec["model"]
    settings = task_settings(base_model)
    tasks = []
    for r, fm in fmaps.items():
        fold = fm.reindex(ids).to_numpy()
        if np.isnan(fold.astype(float)).any():
            raise AssertionError("a candidate participant has no outer fold")
        for k in range(R.OUTER_SPLITS):
            te, tr = ids[fold == k], ids[fold != k]
            t = {"cid": spec["id"], "model": base_model, "features": feats, "groups": spec["groups"], "imputation":
                 spec["imputation"], "platt": platt_also or spec["model"] in PLATT_MODELS, "repeat": int(r), "fold": int(k),
                 "seed": R.SEED, "inner_seed": R.SEED + 100 * int(r) + k + 1,
                 "X_tr": frame.loc[tr, feats], "y_tr": pop.y.loc[tr].to_numpy(), "X_te": frame.loc[te, feats], **settings}
            if base_model == "cox_l2":
                sv_index = pop.survival[spec["timing"]].index
                f = fm.reindex(sv_index).to_numpy().astype(float)
                t["S_tr"] = _cox_frame(pop, spec["timing"], feats, sv_index[~np.isnan(f) & (f != k)])
            tasks.append(t)
    return tasks


# ------------------------------------------------------------------------------------------------ workers
def _check(est, allowed, forbidden):
    rows = M.fitted_rows(est)
    if rows & frozenset(forbidden):
        raise LeakageError(f"fitted on {len(rows & frozenset(forbidden))} outer-test row(s)")
    if rows != frozenset(allowed):
        raise LeakageError("estimator was not fitted on exactly the intended rows")


def _logloss(y, p):
    q = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def _inner(y, seed, k):
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def _generic(t):
    from yoad.benchmark.metrics import fit_calibrator
    X, y, Xte = t["X_tr"], t["y_tr"], t["X_te"]
    configs, metric = t["configs"], t["tune_metric"]
    inner = _inner(y, t["inner_seed"], t["inner_splits"])
    scores, oofs = [], []
    for cfg in configs:
        oof = np.full(len(y), np.nan)
        for itr, iva in inner:
            est = M.build(t["model"], cfg, t["features"], t["seed"], t["imputation"], t["groups"]).fit(X.iloc[itr], y[itr])
            _check(est, X.index[itr], X.index[iva])
            oof[iva] = est.predict_proba(X.iloc[iva])[:, 1]
        scores.append(_logloss(y, oof) if metric == "log_loss" else -MT.auroc(y, oof)); oofs.append(oof)
    j = int(np.nanargmin(scores))
    final = M.build(t["model"], configs[j], t["features"], t["seed"], t["imputation"], t["groups"]).fit(X, y)
    _check(final, X.index, Xte.index)
    p = final.predict_proba(Xte)[:, 1]
    out = {"params": configs[j], "p_test": p, "threshold": MT.youden(y, oofs[j]),
           "inner_scores": [{"config": json.dumps(c, sort_keys=True), "inner_score": s, "metric": metric}
                            for c, s in zip(configs, scores)]}
    if t["platt"]:
        cal = fit_calibrator("sigmoid", y, oofs[j], t["seed"])
        out.update(p_test_platt=cal(p), threshold_platt=MT.youden(y, cal(oofs[j])))
    return out


def _cox(t):
    st, Xte = t["S_tr"], t["X_te"]
    grid = t["configs"]
    lab = st["_state"].isin([EVENT, NONEVENT]).to_numpy()
    yl = (st["_state"] == EVENT).astype(int).to_numpy()
    inner = list(StratifiedKFold(t["inner_splits"], shuffle=True, random_state=t["inner_seed"]).split(np.zeros(len(st)),
                                                                                                       st["_state"]))
    scores, oofs = [], []
    for cfg in grid:
        oof = np.full(len(st), np.nan)
        for itr, iva in inner:
            cp = M.CoxPipeline(t["features"], cfg["penalty"], RISK_HORIZON_DAYS).fit(
                st.iloc[itr], st["time_days"].to_numpy()[itr], st["event"].to_numpy()[itr])
            _check(cp, st.index[itr], st.index[iva])
            va = iva[lab[iva]]
            oof[va] = cp.predict_proba(st.iloc[va])[:, 1]
        scores.append(_logloss(yl[lab], oof[lab])); oofs.append(oof)
    j = int(np.nanargmin(scores))
    final = M.CoxPipeline(t["features"], grid[j]["penalty"], RISK_HORIZON_DAYS).fit(st, st["time_days"].to_numpy(),
                                                                                   st["event"].to_numpy())
    _check(final, st.index, Xte.index)
    return {"params": grid[j], "p_test": final.predict_proba(Xte)[:, 1], "threshold": MT.youden(yl[lab], oofs[j][lab]),
            "inner_scores": [{"config": json.dumps(c), "inner_score": s, "metric": "log_loss"} for c, s in zip(grid, scores)],
            "cox_training_participants": int(len(st)), "cox_training_events": int(st["event"].sum())}


def _simkhada(t):
    from yoad.experiments.simkhada_baseline import baseline_lib as bl
    feats = t["features"]
    num, cat = [f for f in feats if f != "female"], [f for f in feats if f == "female"]
    X, Xte, y = t["X_tr"][num + cat], t["X_te"][num + cat], t["y_tr"]
    groups = np.asarray(X.index)
    pipe = bl.make_pipeline("xgboost", "smotenc", num, cat, t["seed"])
    cv = bl.sgkf(y, groups, t["search_inner_splits"], t["seed"])
    s, failed = bl.run_search(pipe, bl.search_space("xgboost", "smotenc"), X, y, groups, t["search_n_iter"], cv, t["seed"], n_jobs=1)
    chosen = clone(s.best_estimator_)
    thr, _ = bl.oof_threshold(chosen, X, y, groups, t["seed"])
    cal = bl.grouped_calibrated(chosen, X, y, groups, t["seed"])
    rows = frozenset(bl.fitted_rows(cal))
    if rows & frozenset(Xte.index) or not rows <= frozenset(X.index):
        raise LeakageError("Simkhada-style pipeline touched outer-test rows")
    return {"params": {k: repr(v) for k, v in s.best_params_.items()}, "p_test": cal.predict_proba(Xte)[:, 1], "threshold": thr,
            "inner_scores": [{"config": "best of randomized search", "inner_score": float(s.best_score_),
                              "metric": "balanced_accuracy"}], "search_failed": int(failed)}


def run_task(t: dict) -> dict:
    import warnings
    t0 = time.time()
    out = {"cid": t["cid"], "repeat": t["repeat"], "fold": t["fold"], "test_ids": np.asarray(t["X_te"].index)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out.update(_cox(t) if t["model"] == "cox_l2" else _simkhada(t) if t["model"] == "xgb_simkhada" else _generic(t))
        except Exception as e:                                    # recorded, never hidden
            out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    out["convergence_warnings"] = sum("converge" in str(w.message).lower() for w in caught)
    out["seconds"] = round(time.time() - t0, 2)
    return out


# ------------------------------------------------------------------------------------------------ assembly and summaries
def assemble(spec: dict, pop: Population, results: list[dict], use_platt: bool) -> dict:
    ids = spec["ids"]; pos = {pid: i for i, pid in enumerate(ids)}
    P = np.full((R.OUTER_REPEATS, len(ids)), np.nan); T = np.full_like(P, np.nan)
    errors, params, inner = [], [], []
    for res in results:
        if "error" in res:
            errors.append({"candidate": spec["id"], "repeat": res["repeat"], "fold": res["fold"], "error": res["error"]}); continue
        ix = [pos[i] for i in res["test_ids"]]
        P[res["repeat"], ix] = res["p_test_platt"] if use_platt else res["p_test"]
        T[res["repeat"], ix] = res["threshold_platt"] if use_platt else res["threshold"]
        params.append({"candidate": spec["id"], "repeat": res["repeat"], "fold": res["fold"],
                       "selected_config": json.dumps(res["params"], sort_keys=True, default=str), "seconds": res["seconds"],
                       "convergence_warnings": res.get("convergence_warnings", 0),
                       **{k: MT.sup(res[k]) if k.startswith("cox_training") else res[k]
                          for k in ("cox_training_participants", "cox_training_events", "search_failed") if k in res}})
        inner += [{"candidate": spec["id"], "repeat": res["repeat"], "fold": res["fold"], **s} for s in res.get("inner_scores", [])]
    return {"spec": spec, "ids": ids, "y": pop.y.loc[ids].to_numpy(), "P": P, "T": T, "errors": errors, "params": params,
            "inner": inner, "complete": not errors and not np.isnan(P).any()}


def summarise(a: dict, B: int) -> tuple[dict, list, pd.DataFrame]:
    s = a["spec"]; y, P, T = a["y"], a["P"], a["T"]
    rs = MT.repeat_summary(y, P, T)
    ci = MT.bootstrap(y, P, B=B)
    row = {"candidate": s["id"], "stage": s["stage"], "population": s["population"], "feature_set": s["feature_set"],
           "model": s["model"], "imputation": s["imputation"], "timing": s["timing"], "n_features": len(s["features"]),
           "participants": MT.sup(len(y)), "events": MT.sup(int(y.sum())), "complete": a["complete"]}
    for m, (mean, sd) in rs.items():
        row[m] = mean; row[f"{m}_sd_repeats"] = sd
        if m in ci:
            row[f"{m}_ci_low"], row[f"{m}_ci_high"] = ci[m]
    row["bootstrap_B_used"] = ci["_B_used"]
    folds = []
    for r in range(P.shape[0]):
        for k, idx in _fold_index(a, r).items():
            yk, pk = y[idx], P[r, idx]
            folds.append({"candidate": s["id"], "repeat": r, "fold": k, "participants": MT.sup(len(idx)),
                          "events": MT.sup(int(yk.sum())), "roc_auc": MT.auroc(yk, pk), "brier": float(np.mean((pk - yk) ** 2))})
    bins = MT.calibration_bins(y, np.nanmean(P, axis=0)).assign(candidate=s["id"])
    return row, folds, bins


def _fold_index(a: dict, r: int) -> dict:
    return a.get("_fold_index", {}).get(r, {})


def _aligned(a: dict, b: dict):
    common = np.intersect1d(a["ids"], b["ids"])
    ia, ib = np.searchsorted(a["ids"], common), np.searchsorted(b["ids"], common)
    if not (np.array_equal(a["ids"][ia], common) and np.array_equal(b["ids"][ib], common)):
        raise AssertionError("participant alignment failed")
    return a["y"][ia], a["P"][:, ia], b["P"][:, ib]


def _contrast(p: dict, y, Pa, Pb, family_size: int, B: int) -> list:
    res = MT.paired_bootstrap(y, Pa, Pb, B=B)
    out = []
    for m in MT.CONTRAST_METRICS:
        d = res[m]; lo_b, hi_b = MT.bonferroni(d["difference"], d["se"], family_size)
        out.append({"family": p["family"], "family_size": family_size, "candidate": p["candidate"],
                    "comparator": p["comparator"], "metric": m, "difference": d["difference"], "ci_low": d["ci_low"],
                    "ci_high": d["ci_high"], "se": d["se"], "ci_low_bonferroni": lo_b, "ci_high_bonferroni": hi_b,
                    "margin": R.MARGINS.get(m), "classification": MT.classify(d["ci_low"], d["ci_high"], R.MARGINS[m])
                    if m in R.MARGINS else None, "participants": MT.sup(res["n"]), "events": MT.sup(res["events"]),
                    "bootstrap_B_used": res["B_used"]})
    return out


def contrast_rows(plan: list, cands: dict, B: int, n_jobs: int = 1) -> list:
    """Paired contrasts, computed in parallel; each uses its own fixed bootstrap seed, so results do not depend on n_jobs."""
    from joblib import Parallel, delayed
    fam = pd.Series([p["family"] for p in plan]).value_counts().to_dict()
    jobs = [delayed(_contrast)(p, *_aligned(cands[p["candidate"]], cands[p["comparator"]]), fam[p["family"]], B) for p in plan]
    return [r for rows in Parallel(n_jobs=n_jobs)(jobs) for r in rows]


# ------------------------------------------------------------------------------------------------ orchestration
def _execute(specs: list, pops: dict, fmaps: dict, n_jobs: int, cache, log) -> dict:
    from joblib import Parallel, delayed
    all_ids = {s["id"] for s in specs}
    cands = {}
    for s in specs:
        if s["model"] == "xgb_tuned_platt" and s["id"].replace("xgb_tuned_platt", "xgb_tuned") in all_ids:
            continue                                         # produced by its xgb_tuned twin
        pop = pops[s["population"]]
        twin = s["id"].replace("|xgb_tuned|", "|xgb_tuned_platt|") if s["model"] == "xgb_tuned" else None
        has_twin = twin in all_ids if twin else False
        cached = cache.load(s["id"]) if cache else None
        if cached is None:
            t0 = time.time()
            results = Parallel(n_jobs=n_jobs)(delayed(run_task)(t) for t in build_tasks(s, pop, fmaps, platt_also=has_twin))
            if cache:
                cache.save(s["id"], results)
            log(f"  {s['id']}: {len(results)} fold fits, {sum('error' in r for r in results)} failed, {time.time() - t0:.0f}s")
        else:
            results = cached
            log(f"  {s['id']}: reused from the cache of this run")
        cands[s["id"]] = assemble(s, pop, results, use_platt=s["model"] in PLATT_MODELS)
        if has_twin:
            ts = next(x for x in specs if x["id"] == twin)
            cands[twin] = assemble(ts, pop, results, use_platt=True)
    for a in cands.values():                                  # fold membership, for per-fold summaries
        fi = {}
        for r, fm in fmaps.items():
            f = fm.reindex(a["ids"]).to_numpy()
            fi[r] = {k: np.flatnonzero(f == k) for k in range(R.OUTER_SPLITS)}
        a["_fold_index"] = fi
    return cands


def selection_pool(cands: dict, population: str) -> list:
    pool = []
    for cid, a in cands.items():
        s = a["spec"]
        if s["population"] != population or s["stage"] not in ("A", "B") or s["imputation"] != "median" or s["timing"] != "primary":
            continue
        rs = MT.repeat_summary(a["y"], a["P"], None, names=("brier", "roc_auc", "calibration_slope", "log_oe"))
        pool.append({"id": cid, "model": s["model"], "features": s["features"], "brier": rs["brier"][0],
                     "roc_auc": rs["roc_auc"][0], "calibration_slope": rs["calibration_slope"][0], "log_oe": rs["log_oe"][0],
                     "complete": a["complete"]})
    return pool


def _vs_best(y, Pa, Pb, B):
    res = MT.paired_bootstrap(y, Pa, Pb, names=("brier", "roc_auc"), B=B)
    return res["brier"]["difference"], res["brier"]["se"], res["roc_auc"]["difference"]


def run_selection(cands: dict, population: str, B: int, n_jobs: int = 1) -> dict:
    from joblib import Parallel, delayed
    pool = selection_pool(cands, population)
    ok = [p for p in pool if p.get("complete") and p.get("brier") is not None and math.isfinite(p["brier"])]
    if not ok:
        return S.select(pool, lambda b, c: (np.nan, np.nan, np.nan))
    best = min(ok, key=lambda p: (p["brier"], S.order_key(p)))["id"]
    for p in ok:
        if not np.array_equal(cands[p["id"]]["ids"], cands[best]["ids"]):
            raise AssertionError("selection candidates must be scored on identical participants")
    others = [p["id"] for p in ok if p["id"] != best]
    vals = Parallel(n_jobs=n_jobs)(delayed(_vs_best)(cands[c]["y"], cands[c]["P"], cands[best]["P"], B) for c in others)
    pre = dict(zip(others, vals))
    return S.select(pool, lambda b, c: pre[c])


def run_benchmark(rows: dict, folds: pd.DataFrame, landmark=None, survival=None, mri_full=(), n_jobs: int = 1, cache=None,
                  B: int = R.BOOT_B, log=print) -> dict:
    pops = build_populations(rows, landmark, survival)
    gates = gate_table(pops)
    feasible = set(gates.loc[gates["status"] != "STOP", "population"])
    matrix = R.experiment_matrix(feasible)
    specs, registry_rows = [], []
    for c in matrix:
        spec, rec = prepare(c, pops, mri_full)
        registry_rows.append(rec)
        if spec:
            specs.append(spec)
    for p in sorted(set(R.POPULATIONS) - feasible):
        registry_rows.append({"id": f"population {p}", "population": p, "status": "skipped",
                              "reason": "population unavailable or below the event gate"})
    fmaps = fold_maps(folds)
    log(f"Stages A-D: {len(specs)} candidates on populations {sorted(feasible)}")
    cands = _execute(specs, pops, fmaps, n_jobs, cache, log)
    sel = {p: run_selection(cands, p, B, n_jobs) for p in ("P_ALL", "P_MRI") if p in feasible}
    # stage E: strict-timing sensitivity of the P_ALL winner and the reference
    e_specs = []
    if sel.get("P_ALL", {}).get("winner"):
        for cid in dict.fromkeys([sel["P_ALL"]["winner"], R.candidate_id("A", "P_ALL", *R.REFERENCE)]):
            p = R.parse_candidate(cid)
            c = {**p, "stage": "E", "timing": "strict"}
            c["id"] = R.candidate_id("E", p["population"], p["feature_set"], p["model"], p["imputation"], "strict")
            spec, rec = prepare(c, pops, mri_full)
            registry_rows.append(rec)
            if spec:
                e_specs.append((spec, cid))
    if e_specs:
        log(f"Stage E: strict-timing sensitivity for {len(e_specs)} candidate(s)")
        cands.update(_execute([s for s, _ in e_specs], pops, fmaps, n_jobs, cache, log))
    plan = R.contrast_plan(list(cands))
    for spec, orig in e_specs:
        if spec["id"] in cands:
            plan.append({"candidate": spec["id"], "comparator": orig, "family": "timing_sensitivity"})
    for p, d in sel.items():
        if d["winner"] and d["winner"] != R.candidate_id("A", p, *R.REFERENCE):
            plan.append({"candidate": d["winner"], "comparator": R.candidate_id("A", p, *R.REFERENCE),
                         "family": "winner_vs_reference"})
    log(f"Summaries and {len(plan)} paired contrasts (B = {B})")
    from joblib import Parallel, delayed
    results, folds_out, bins = [], [], []
    for row, f, b in Parallel(n_jobs=n_jobs)(delayed(summarise)(a, B) for a in cands.values()):
        results.append(row); folds_out += f; bins.append(b)
    status = {r["id"]: r for r in registry_rows}
    for cid, a in cands.items():
        status[cid] = {**status.get(cid, {}), "status": "completed" if a["complete"] else "failed",
                       "folds_failed": len(a["errors"]),
                       "seconds": round(sum(x["seconds"] for x in a["params"]), 1)}
    reg_cols = ["id", "stage", "population", "feature_set", "model", "imputation", "timing", "status", "reason", "n_features",
                "features_dropped", "participants", "events", "nonevents", "folds_failed", "seconds"]
    tables = {"population_gate": gates.drop(columns=["_events", "_nonevents"]),
              "experiment_registry": pd.DataFrame(list(status.values())).reindex(columns=reg_cols),
              "candidate_results": pd.DataFrame(results), "outer_fold_metrics": pd.DataFrame(folds_out),
              "calibration_bins": pd.concat(bins, ignore_index=True) if bins else pd.DataFrame(),
              "paired_contrasts": pd.DataFrame(contrast_rows(plan, cands, B, n_jobs)),
              "selected_configurations": pd.DataFrame([p for a in cands.values() for p in a["params"]]),
              "tuning_log": pd.DataFrame([x for a in cands.values() for x in a["inner"]]),
              "failures": pd.DataFrame([e for a in cands.values() for e in a["errors"]],
                                       columns=["candidate", "repeat", "fold", "error"])}
    for p, d in sel.items():
        tables[f"selection_{p}"] = pd.DataFrame(d["table"])
    selection = {p: {k: d[k] for k in ("winner", "best_brier", "reason")} for p, d in sel.items()}
    selection["rule"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in R.SELECTION.items()}
    selection["sha256"] = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()
    return {"tables": tables, "selection": selection, "cands": cands, "pops": pops, "specs": {s["id"]: s for s in specs}}


# ------------------------------------------------------------------------------------------------ locked held-out stage
HELDOUT_METRICS = ("roc_auc", "average_precision", "brier", "log_oe", "calibration_intercept", "calibration_slope")


def _frozen_reference(rows_primary: pd.DataFrame, model_id: str):
    """Exactly the A3 fit of the frozen 2 x 2 point estimate: REF-labelled baseline rows of the training partition, in frame
    order, frozen pipeline factory, seed 20261017; scored on held-out REF-determinate baseline participants in frame order."""
    from yoad.experiments.visit_anchor_audit import design_lib as frozen
    from yoad.benchmark.design_bootstrap import training_rows
    feats = frozen.PREDICTOR_SETS["primary"]
    train_anchors = rows_primary[rows_primary["partition"] == "train"].copy()
    t, y = training_rows(train_anchors, "state_REF", "baseline")
    params = frozen.DEFAULT_CONFIG["models"][M.SB_KEY[model_id]]
    pipe = frozen.make_pipeline_factory(M.SB_KEY[model_id], params, feats)("A3", frozen.DEFAULT_CONFIG["seed"]).fit(t[feats], y)
    held = rows_primary[rows_primary["partition"] == "held_out"]
    ev = held[held["is_baseline"] & held["state_REF"].isin([EVENT, NONEVENT])]
    yv = (ev["state_REF"] == EVENT).astype(int).to_numpy()
    if set(t["participant_id"]) & set(ev["participant_id"]):
        raise LeakageError("reference training and held-out participants overlap")
    return ev["participant_id"].to_numpy(), yv, pipe.predict_proba(ev[feats])[:, 1]


def run_heldout(rows: dict, bench: dict, stored_a3: dict | None, B: int = R.BOOT_B) -> dict:
    """bench: output of run_benchmark (its selection must already be written). stored_a3: {model: {metric: stored point}}."""
    from yoad.benchmark.metrics import METRICS as FROZEN_METRICS
    prim = rows["primary"]
    held = prim[(prim["partition"] == "held_out") & prim["is_baseline"] & prim["state_REF"].isin([EVENT, NONEVENT])]
    held = held.set_index("participant_id").sort_index()
    yh = (held["state_REF"] == EVENT).astype(int)
    preds, recs, repro = {}, [], []
    for model_id in ("lr_l2_ref", "xgb_ref"):
        ids, y, p = _frozen_reference(prim, model_id)
        key = f"REFERENCE_A3|{model_id}"
        order = np.argsort(ids, kind="mergesort")
        preds[key] = (ids[order], y[order], p[order], np.full(len(p), np.nan))
        for m in HELDOUT_METRICS:
            v = float(FROZEN_METRICS[m].fn(y, p))
            st = (stored_a3 or {}).get(model_id, {}).get(m)
            repro.append({"model": model_id, "metric": m, "reproduced": v, "stored_A3": st,
                          "abs_difference": None if st is None else abs(v - st),
                          "status": "not compared (no stored value supplied)" if st is None else
                          ("reproduced" if abs(v - st) <= 1e-8 else "DIFFERS")})
    to_fit = []
    for p, d in bench["selection"].items():
        if p in ("P_ALL", "P_MRI") and d.get("winner"):
            to_fit.append((f"WINNER_{p}", d["winner"]))
    to_fit.append(("LOCKED_TRANSPORT", R.candidate_id("A", "P_ALL", "FS08b_TRANSPORT", "lr_l2_ref")))
    if "P_MRI" in bench["selection"]:
        to_fit.append(("MRI_SUBSET_REFERENCE", R.candidate_id("A", "P_MRI", *R.REFERENCE)))
    locked = {}
    for label, cid in to_fit:
        spec = bench["specs"].get(cid) or bench["specs"].get(cid.replace("xgb_tuned_platt", "xgb_tuned"))
        if spec is None:
            recs.append({"label": label, "candidate": cid, "status": "not available (candidate was not run)"}); continue
        pop = bench["pops"][spec["population"]]
        Xh = held.reindex(columns=spec["features"])
        if spec["population"] != "P_ALL":
            inst = R.INSTRUMENT_OF_POPULATION[spec["population"]]
            Xh = Xh[held[f"{inst}_date"].notna().to_numpy()] if f"{inst}_date" in held else Xh.iloc[0:0]
        if not len(Xh):
            recs.append({"label": label, "candidate": cid, "status": "no held-out participant has the required modality"}); continue
        base_model = "xgb_tuned" if spec["model"] == "xgb_tuned_platt" else spec["model"]
        t = {"cid": cid, "model": base_model, "features": spec["features"], "groups": spec["groups"],
             "imputation": spec["imputation"], "platt": spec["model"] in PLATT_MODELS, "repeat": 0, "fold": 0,
             "seed": R.SEED, "inner_seed": R.SEED + 99991, "X_tr": pop.frame.loc[spec["ids"], spec["features"]],
             "y_tr": pop.y.loc[spec["ids"]].to_numpy(), "X_te": Xh, **task_settings(base_model)}
        if base_model == "cox_l2":
            t["S_tr"] = _cox_frame(pop, spec["timing"], spec["features"])
        res = run_task(t)
        if "error" in res:
            recs.append({"label": label, "candidate": cid, "status": f"failed: {res['error']}"}); continue
        use_platt = spec["model"] in PLATT_MODELS
        p = res["p_test_platt"] if use_platt else res["p_test"]
        thr = res["threshold_platt"] if use_platt else res["threshold"]
        ids = np.asarray(Xh.index)
        preds[label] = (ids, yh.loc[ids].to_numpy(), p, np.full(len(p), thr))
        locked[label] = {"candidate": cid, "selected_config": res["params"], "threshold": thr}
        recs.append({"label": label, "candidate": cid, "status": "evaluated"})
    rows_out, bins = [], []
    for label, (ids, y, p, thr) in preds.items():
        pm = MT.point_metrics(y, p)
        ci = MT.bootstrap(y, p[None, :], B=B)
        row = {"model": label, "candidate": locked.get(label, {}).get("candidate", label), "participants": MT.sup(len(y)),
               "events": MT.sup(int(y.sum())), **pm, **{f"{m}_ci_low": ci[m][0] for m in MT.CORE_METRICS},
               **{f"{m}_ci_high": ci[m][1] for m in MT.CORE_METRICS},
               "uncertainty": "fixed-model participant bootstrap of the held-out set (no refitting)"}
        if not np.isnan(thr).all():
            row.update(MT.threshold_metrics(y, p, thr))
        rows_out.append(row)
        bins.append(MT.calibration_bins(y, p).assign(model=label))
    contrasts = []
    pairs = [(f"WINNER_P_ALL", "REFERENCE_A3|lr_l2_ref"), ("LOCKED_TRANSPORT", "REFERENCE_A3|lr_l2_ref"),
             ("WINNER_P_MRI", "MRI_SUBSET_REFERENCE")]
    for a, b in pairs:
        if a in preds and b in preds:
            ia, ib = preds[a][0], preds[b][0]
            common = np.intersect1d(ia, ib)
            xa, xb = np.searchsorted(ia, common), np.searchsorted(ib, common)
            res = MT.paired_bootstrap(preds[a][1][xa], preds[a][2][xa][None, :], preds[b][2][xb][None, :], B=B)
            for m in MT.CONTRAST_METRICS:
                d = res[m]
                contrasts.append({"candidate": a, "comparator": b, "metric": m, "difference": d["difference"],
                                  "ci_low": d["ci_low"], "ci_high": d["ci_high"], "margin": R.MARGINS.get(m),
                                  "classification": MT.classify(d["ci_low"], d["ci_high"], R.MARGINS[m]) if m in R.MARGINS else None,
                                  "participants": MT.sup(res["n"]), "events": MT.sup(res["events"])})
    return {"tables": {"heldout_results": pd.DataFrame(rows_out), "heldout_contrasts": pd.DataFrame(contrasts),
                       "heldout_calibration_bins": pd.concat(bins, ignore_index=True) if bins else pd.DataFrame(),
                       "heldout_models": pd.DataFrame(recs), "reference_reproduction": pd.DataFrame(repro)},
            "locked": locked}


def transport_parameters(bench: dict) -> dict | None:
    """Coefficients, intercept, imputation medians and scaling of the locked transportable model (FS08b, lr_l2_ref) fitted on
    all development participants: aggregate model parameters for a later locked NACC validation (no participant rows)."""
    cid = R.candidate_id("A", "P_ALL", "FS08b_TRANSPORT", "lr_l2_ref")
    spec = bench["specs"].get(cid)
    if spec is None:
        return None
    pop = bench["pops"]["P_ALL"]
    pipe = M.sb_pipeline("lr_l2_ref", R.MODELS["lr_l2_ref"]["fixed"], spec["features"], R.SEED).fit(
        pop.frame.loc[spec["ids"], spec["features"]], pop.y.loc[spec["ids"]].to_numpy())
    ct = pipe.named_steps["columns"].named_transformers_["num"]
    lr = pipe.named_steps["model"].estimator_
    return {"candidate": cid, "features": spec["features"], "imputation_median": ct.named_steps["impute"].statistics_.tolist(),
            "scaling_mean": ct.named_steps["scale"].mean_.tolist(), "scaling_scale": ct.named_steps["scale"].scale_.tolist(),
            "coefficients": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
            "note": "fitted on all REF-labelled development participants; for a locked external validation without refitting"}
