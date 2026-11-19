"""Outer-fold fitting, metrics and paired contrasts (PROTOCOL.md §§6-7). No file access.

Outer folds: the frozen participant folds (multimodal_benchmark.cohort.outer_folds, 5 x 3) broadcast to every anchor of the
participant, so every candidate is scored on identical anchors and no participant is ever in a training and a test part of
the same split. Inner tuning, thresholds, recalibration, preprocessing, feature activity and early stopping use outer-training
anchors only.

Estimands: PRIMARY anchor-weighted (each REF-determinate anchor weight 1); SENSITIVITY participant-weighted (each participant's
anchors sum to 1). Point estimate = metric on the pooled out-of-fold predictions of a repeat, averaged over repeats.
Intervals: participant-cluster bootstrap (participants resampled with all their anchors; every repeat scored on the same
resample and averaged), percentile 95%, conditional on the fitted fold models. Paired contrasts use the same resamples.
Modality contribution: REFITTED ablations on identical folds (drop a block in training and evaluation); the earlier
permutation diagnostic is withdrawn because permuting values did not carry the matching times, lengths and availability.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.special import expit

from yoad.multimodal import metrics as MT
from yoad.evaluation.safeguards import LeakageError
from yoad.longitudinal import baselines as BL
from yoad.longitudinal import config as CF
from yoad.longitudinal import sequences as SQ

# ------------------------------------------------------------------------------------------------ folds


def anchor_folds(meta: pd.DataFrame, folds: pd.DataFrame) -> np.ndarray:
    """(REPEATS, N) outer fold of every anchor."""
    out = np.full((CF.REPEATS, len(meta)), -1, int)
    for r in range(CF.REPEATS):
        f = folds[folds["repeat"] == r].set_index("participant_id")["fold"]
        out[r] = meta["participant_id"].map(f).fillna(-1).astype(int).to_numpy()
    if (out < 0).any():
        raise LeakageError("an anchor's participant has no outer fold")
    return out


def outer_tasks(meta: pd.DataFrame, af: np.ndarray) -> list:
    """[(r, k, train idx, test idx)] with participant disjointness and the per-fold class minimum checked."""
    pid, y = meta["participant_id"].to_numpy(), meta["y"].to_numpy()
    out = []
    for r in range(af.shape[0]):
        for k in range(CF.SPLITS):
            te = np.flatnonzero(af[r] == k); tr = np.flatnonzero(af[r] != k)
            if set(pid[tr]) & set(pid[te]):
                raise LeakageError("participant in both outer training and test parts")
            if min(int(y[te].sum()), int((1 - y[te]).sum())) < CF.MIN_FOLD_CLASS:
                raise ValueError(f"outer fold ({r}, {k}) lacks events or non-events")
            out.append((r, k, tr, te))
    return out


# ------------------------------------------------------------------------------------------------ ablation settings
def ablation_settings(name: str | None, latest: bool = False) -> dict:
    """Preprocessor arguments of a candidate (latest = E_SNAP) and of a refitted ablation (None = full model). A dropped block
    is disabled in the Preprocessor, so its values, dates, availability and every timing summary are removed before any
    timeline is built (sequences.restrict)."""
    if name is None:
        return {"latest": latest}
    if name.startswith("drop_"):
        return {"drop": (name[5:],), "latest": latest}
    if name == "values_masked_biomarkers":
        return {"values_masked": CF.BIOMARKER_BLOCKS, "latest": latest}
    raise ValueError(name)


def ablate_frame(X: pd.DataFrame, name: str | None) -> pd.DataFrame:
    """Tabular counterpart for A1_XGB_MATCHED (applied to training and evaluation rows alike)."""
    if name is None:
        return X
    out = X.copy()
    if name.startswith("drop_"):
        b = name[5:]
        if b in CF.STATIC_BLOCKS:
            cols = [c for c in CF.STATIC_BLOCKS[b] if c in X.columns]
        else:                                           # values, staleness, availability and counts of the block
            cols = [c for c in X.columns if BL.column_block(c) == b]
        return out.drop(columns=cols)
    if name == "values_masked_biomarkers":
        return out.drop(columns=[f for b in CF.BIOMARKER_BLOCKS for f in CF.TV_BLOCKS[b] if f in out.columns])
    raise ValueError(name)


# ------------------------------------------------------------------------------------------------ tasks
def neural_task(model_id: str, s: SQ.Samples, tr, te, r: int, k: int, ablation: str | None = None,
                latest: bool = False) -> dict:
    """Inner grouped tuning over NEURAL_GRID with early stopping; common threshold / recalibration rule on inner OOF; final
    refit with the median inner best epoch for every seed in NEURAL_SEEDS; outer-test predictions = seed mean."""
    from yoad.longitudinal import models as MD, train as TR
    t0 = time.time()
    y = s.meta["y"].to_numpy(float); pid = s.meta["participant_id"].to_numpy()
    w = SQ.weights(pid)
    kw = ablation_settings(ablation, latest)
    inner = BL.inner_splits(y[tr], pid[tr], CF.INNER_SPLITS, CF.inner_seed(r, k))
    scores, oofs, epochs = [], [], []
    for cfg in CF.NEURAL_GRID:
        oof = np.full(len(tr), np.nan); eps = []
        for i, (itr, iva) in enumerate(inner):
            pre = SQ.Preprocessor(**kw).fit(s, tr[itr])                # inner-training anchors only
            fit = TR.fit(model_id, cfg["hidden"], pre.transform(s), y, w, tr[itr], tr[iva], seed=CF.torch_seed(r, k, i, "inner"))
            oof[iva] = fit["p_val"]; eps.append(fit["best_epoch"])
        scores.append(TR.weighted_log_loss(y[tr], oof, w[tr])); oofs.append(oof); epochs.append(eps)
    j = int(np.argmin(scores))
    hidden, n_ep = CF.NEURAL_GRID[j]["hidden"], max(1, int(np.median(epochs[j])))
    pre = SQ.Preprocessor(**kw).fit(s, tr)                               # outer-training anchors only
    if pre.train_ids_ & frozenset(pid[te]):
        raise LeakageError("preprocessing fitted on an outer-test participant")
    arrays = pre.transform(s)
    fits = [TR.fit(model_id, hidden, arrays, y, w, tr, None, epochs=n_ep, seed=CF.torch_seed(r, k, sd, "final"))["model"]
            for sd in CF.NEURAL_SEEDS]
    ps = np.stack([TR.predict(m, arrays, te) for m in fits])
    p = ps.mean(0)
    return {"p_test": p, "p_seeds": ps, **BL.common_rule(y[tr].astype(int), oofs[j], p, CF.SEED),
            "params": {"hidden": hidden, "epochs": n_ep}, "n_params": MD.n_params(fits[0]),
            "preprocessing": pre.summary(), "_fits": fits, "_pre": pre,
            "inner_scores": [{"config": f"hidden={c['hidden']}", "inner_weighted_log_loss": sc,
                              "median_best_epoch": float(np.median(e))} for c, sc, e in zip(CF.NEURAL_GRID, scores, epochs)],
            "seconds": round(time.time() - t0, 1)}


def tables(s: SQ.Samples) -> dict:
    af = s.anchor_frame.copy()
    for c in BL.FS16 + BL.CORE7 + CF.SIMKHADA_FEATURES:
        if c not in af.columns:
            af[c] = np.nan
    idx = pd.RangeIndex(s.n)
    bs = SQ.block_summary(s)
    num, cat = BL.simkhada_columns(af)
    return {"LR_REF": af[list(BL.CORE7)].set_index(idx), "A0_XGB_INCUMBENT": af[list(BL.FS16)].set_index(idx),
            "A_SIMKHADA": af[num + cat].set_index(idx), "A1_XGB_MATCHED": SQ.snapshot_frame(s).set_index(idx),
            "AVAIL": bs[[f"{p}_{b}" for b in CF.TV_BLOCKS for p in ("has", "n")]].set_index(idx)}


def baseline_task(cid: str, X: pd.DataFrame, s: SQ.Samples, tr, te, r: int, k: int, ablation: str | None = None) -> dict:
    t0 = time.time()
    y = s.meta["y"].to_numpy(int); pid = s.meta["participant_id"].to_numpy()
    X = ablate_frame(X, ablation)
    if cid == "A_SIMKHADA":
        out = BL.simkhada_task(X.iloc[tr], y[tr], pid[tr], X.iloc[te], CF.SEED, CF.inner_seed(r, k))
    else:
        model = "xgb_ref" if cid in ("A0_XGB_INCUMBENT", "A1_XGB_MATCHED") else "lr_l2_ref"
        out = BL.frozen_task(model, X.iloc[tr], y[tr], pid[tr], X.iloc[te], CF.SEED, CF.inner_seed(r, k))
    out["seconds"] = round(time.time() - t0, 1)
    return out


def run_task(spec: dict) -> dict:
    """One (candidate, ablation, repeat, fold); errors are recorded, never hidden; models are not returned."""
    import warnings
    out = {"cid": spec["cid"], "ablation": spec.get("ablation"), "repeat": spec["r"], "fold": spec["k"], "te": spec["te"]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        try:
            if spec["cid"] in CF.NEURAL:
                res = neural_task(spec["model"], spec["s"], spec["tr"], spec["te"], spec["r"], spec["k"], spec.get("ablation"),
                                  spec.get("latest", False))
            else:
                res = baseline_task(spec["cid"], spec["X"], spec["s"], spec["tr"], spec["te"], spec["r"], spec["k"],
                                    spec.get("ablation"))
            out.update({k: v for k, v in res.items() if not k.startswith("_")})
        except LeakageError:
            raise
        except Exception as e:                                           # recorded in the failure table
            out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    return out


def build_specs(cids, s: SQ.Samples, tasks: list, ablations=(None,)) -> list:
    tabs = tables(s)
    specs = []
    for cid in cids:
        for ab in ablations:
            for r, k, tr, te in tasks:
                sp = {"cid": cid, "ablation": ab, "r": r, "k": k, "tr": tr, "te": te}
                if cid in CF.NEURAL:
                    sp.update(model="E_COMBINED" if cid == "E_SNAP" else cid, s=s, latest=cid == "E_SNAP")
                else:
                    sp.update(X=tabs[cid], s=s)
                specs.append(sp)
    return specs


def execute(specs: list, n_jobs: int = 1) -> list:
    if n_jobs <= 1:
        return [run_task(sp) for sp in specs]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(delayed(run_task)(sp) for sp in specs)


def assemble(results: list, n: int) -> dict:
    """{(cid, ablation): {"P", "T", "PR", "TR" (R, N), "PS": (R, S, N) | None, "folds": [...], "errors": [...]}}"""
    out = {}
    for res in results:
        key = res["cid"] if res.get("ablation") is None else (res["cid"], res["ablation"])
        c = out.setdefault(key, {m: np.full((CF.REPEATS, n), np.nan) for m in ("P", "T", "PR", "TR")}
                           | {"PS": None, "folds": [], "errors": []})
        r, te = res["repeat"], res["te"]
        if "error" in res:
            c["errors"].append({"repeat": r, "fold": res["fold"], "error": res["error"]}); continue
        c["P"][r, te], c["T"][r, te] = res["p_test"], res["threshold"]
        c["PR"][r, te], c["TR"][r, te] = res["p_test_recal"], res["threshold_recal"]
        if res.get("p_seeds") is not None:
            if c["PS"] is None:
                c["PS"] = np.full((CF.REPEATS, len(CF.NEURAL_SEEDS), n), np.nan)
            c["PS"][r][:, te] = res["p_seeds"]
        c["folds"].append({"repeat": r, "fold": res["fold"], "threshold": res["threshold"], "n_params": res.get("n_params"),
                           "params": str(res.get("params")), "seconds": res.get("seconds"),
                           "preprocessing": res.get("preprocessing"), "inner_scores": res.get("inner_scores"),
                           "search_failed": res.get("search_failed")})
    return out


# ------------------------------------------------------------------------------------------------ metrics (incl. weighted)
NAMES = CF.METRICS + CF.THRESHOLD_METRICS


def _wauroc(y, p, w):
    o = np.argsort(p, kind="mergesort"); y, p, w = y[o], p[o], w[o]
    W1, W0 = (w * y).sum(), (w * (1 - y)).sum()
    if W1 <= 0 or W0 <= 0:
        return np.nan
    u, inv = np.unique(p, return_inverse=True)
    pos = np.bincount(inv, w * y, len(u)); neg = np.bincount(inv, w * (1 - y), len(u))
    below = np.concatenate([[0.0], np.cumsum(neg)[:-1]])
    return float((pos * (below + 0.5 * neg)).sum() / (W1 * W0))


def _wcal(y, p, w, slope: bool):
    off = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    X = np.column_stack([np.ones(len(y)), off]) if slope else np.ones((len(y), 1))
    b = np.array([0.0, 1.0]) if slope else np.array([0.0])
    for _ in range(100):
        mu = expit(X @ b + (0 if slope else off)); W = w * mu * (1 - mu)
        H = X.T @ (X * W[:, None])
        try:
            step = np.linalg.solve(H, X.T @ (w * (y - mu)))
        except np.linalg.LinAlgError:
            return np.nan
        b = b + step
        if not np.all(np.isfinite(b)) or np.abs(b).max() > 1e6:
            return np.nan
        if np.abs(step).max() < 1e-10:
            break
    return float(b[1] if slope else b[0])


def weighted_metrics(y, p, t, w, names) -> dict:
    y, p, t, w = np.asarray(y, float), np.asarray(p, float), np.asarray(t, float), np.asarray(w, float)
    d = (p >= t).astype(float)
    W1, W0 = (w * y).sum(), (w * (1 - y)).sum()
    q = np.clip(p, 1e-15, 1 - 1e-15)
    sens = (w * d * y).sum() / W1 if W1 else np.nan
    spec = (w * (1 - d) * (1 - y)).sum() / W0 if W0 else np.nan
    f = {"roc_auc": lambda: _wauroc(y, p, w), "brier": lambda: float((w * (p - y) ** 2).sum() / w.sum()),
         "log_loss": lambda: float(-(w * (y * np.log(q) + (1 - y) * np.log(1 - q))).sum() / w.sum()),
         "calibration_intercept": lambda: _wcal(y, p, w, False), "calibration_slope": lambda: _wcal(y, p, w, True),
         "sensitivity": lambda: sens, "specificity": lambda: spec, "balanced_accuracy": lambda: (sens + spec) / 2,
         "accuracy": lambda: float((w * (d == y)).sum() / w.sum())}
    return {n: f[n]() for n in names}


def all_metrics(y, p, t, names=None, w=None) -> dict:
    """Unweighted (anchor-weighted) metrics, or participant-weighted ones when `w` is given."""
    names = NAMES if names is None else names
    if w is not None:
        return weighted_metrics(y, p, t, w, names)
    pm = [n for n in names if n in CF.METRICS]
    m = MT.point_metrics(y, p, pm) if pm else {}
    if any(n in CF.THRESHOLD_METRICS for n in names):
        m.update(MT.threshold_metrics(y, p, t))
        d = (np.asarray(p) >= np.asarray(t)).astype(int)
        m["accuracy"] = float((d == np.asarray(y)).mean())
    return {n: m[n] for n in names}


def _repeat_avg(y, P, T, idx, names=NAMES, w=None) -> np.ndarray:
    return np.nanmean([[all_metrics(y[idx], P[r, idx], T[r, idx], names, None if w is None else w[idx])[n] for n in names]
                       for r in range(P.shape[0])], axis=0)


def cluster_index(groups: np.ndarray):
    ids, inv = np.unique(groups, return_inverse=True)
    return ids, [np.flatnonzero(inv == i) for i in range(len(ids))]


def resamples(groups: np.ndarray, B: int, seed: int) -> list:
    """B participant-cluster resamples as anchor index arrays (the same list for every candidate)."""
    ids, members = cluster_index(groups)
    rng = np.random.default_rng(seed)
    return [np.concatenate([members[i] for i in rng.integers(0, len(ids), len(ids))]) for _ in range(B)]


def summarise(y, P, T, mask, draws, w=None, names=NAMES) -> dict:
    """Repeat-mean metrics on anchors in `mask` with SD over repeats and cluster-bootstrap percentile CIs."""
    names = tuple(names)
    base = np.flatnonzero(mask & ~np.isnan(P).any(axis=0))
    point = _repeat_avg(y, P, T, base, names, w)
    per = np.array([[all_metrics(y[base], P[r, base], T[r, base], names, None if w is None else w[base])[n] for n in names]
                    for r in range(P.shape[0])])
    sd = per.std(axis=0, ddof=1) if P.shape[0] > 1 else np.full(len(names), np.nan)
    inset = np.zeros(len(y), bool); inset[base] = True
    D = []
    for d in draws:
        idx = d[inset[d]]
        if len(idx) == 0 or y[idx].min() == y[idx].max():
            continue
        D.append(_repeat_avg(y, P, T, idx, names, w))
    D = np.array(D)
    out = {"anchors": int(len(base)), "event_anchors": int(y[base].sum()), "B_used": len(D)}
    for i, n in enumerate(names):
        col = D[:, i][np.isfinite(D[:, i])] if len(D) else np.array([])
        out[n] = float(point[i]); out[f"{n}_sd_repeats"] = float(sd[i])
        out[f"{n}_ci_low"] = float(np.quantile(col, CF.ALPHA / 2)) if len(col) else np.nan
        out[f"{n}_ci_high"] = float(np.quantile(col, 1 - CF.ALPHA / 2)) if len(col) else np.nan
    return out


def paired(y, Pa, Ta, Pb, Tb, mask, draws, names=CF.CONTRAST_METRICS) -> dict:
    """Difference a - b of repeat-mean metrics on anchors predicted by both, with the same cluster resamples."""
    base = np.flatnonzero(mask & ~(np.isnan(Pa).any(axis=0) | np.isnan(Pb).any(axis=0)))
    names = tuple(names)
    pt = _repeat_avg(y, Pa, Ta, base, names) - _repeat_avg(y, Pb, Tb, base, names)
    inset = np.zeros(len(y), bool); inset[base] = True
    D = []
    for d in draws:
        idx = d[inset[d]]
        if len(idx) == 0 or y[idx].min() == y[idx].max():
            continue
        D.append(_repeat_avg(y, Pa, Ta, idx, names) - _repeat_avg(y, Pb, Tb, idx, names))
    D = np.array(D)
    out = {"anchors": int(len(base)), "B_used": len(D)}
    for i, n in enumerate(names):
        col = D[:, i][np.isfinite(D[:, i])] if len(D) else np.array([])
        out[n] = {"difference": float(pt[i]), "ci_low": float(np.quantile(col, CF.ALPHA / 2)) if len(col) else np.nan,
                  "ci_high": float(np.quantile(col, 1 - CF.ALPHA / 2)) if len(col) else np.nan,
                  "se": float(np.std(col, ddof=1)) if len(col) > 1 else np.nan}
    return out


def seed_sd(y, PS, T, mask) -> float:
    """Mean over repeats of the SD across seeds of balanced accuracy (each seed's predictions, the fold threshold)."""
    if PS is None:
        return np.nan
    vals = []
    for r in range(PS.shape[0]):
        base = np.flatnonzero(mask & ~np.isnan(PS[r]).any(axis=0))
        ba = [MT.threshold_metrics(y[base], PS[r, s, base], T[r, base])["balanced_accuracy"] for s in range(PS.shape[1])]
        vals.append(np.std(ba, ddof=1))
    return float(np.mean(vals))


def fold_metrics(y, P, T, af: np.ndarray) -> pd.DataFrame:
    rows = []
    for r in range(P.shape[0]):
        for k in range(CF.SPLITS):
            te = np.flatnonzero(af[r] == k)
            if np.isnan(P[r, te]).any():
                continue
            m = all_metrics(y[te], P[r, te], T[r, te])
            rows.append({"repeat": r, "fold": k, **{n: m[n] for n in ("balanced_accuracy", "accuracy", "roc_auc", "brier")}})
    return pd.DataFrame(rows)
