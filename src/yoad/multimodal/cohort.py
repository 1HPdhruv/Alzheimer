"""Benchmark analysis frames, built only through the FROZEN cohort, label and split functions (PROTOCOL.md §§2-5).

Frozen steps, called exactly as experiments/visit_anchor_audit/design_lib.run_design calls them: eligible_anchors -> attach_predictors
(frozen MMSE / ADAS-Cog13 / FAQ windows) -> label_anchors (SIM and REF, endpoint 'any') -> split_participants (seed 20261017, 60/40,
stratified on the baseline any-cause REF state). The frozen feature columns, labels, row order and partitions are therefore those
of the five stored runs; the benchmark only ADDS columns:
  * added long-format instruments through the same frozen attach_predictors rule (separate call, so frozen values are untouched);
  * wide instruments (MRI, CSF, plasma, amyloid PET) through `attach_instrument`: the assessment dated in [anchor + lo, anchor +
    post_days] nearest to the anchor, ties to the earlier date; undated, post-cut-off and same-date-conflicting assessments
    removed (the frozen measurement conventions applied per assessment);
  * landmark rows: per participant the EARLIEST eligible anchor at protocol month >= 12 (chosen without looking at its label),
    with baseline values, per-variable OLS slopes and years since baseline from measurements dated on or before the landmark
    index time only (run-time check);
  * right-censored follow-up from the baseline anchor for the censoring-aware model: first dementia-only date = event, otherwise
    the last later assessment = censoring, administrative censoring at 24 months + 90 days, visits after death discarded.
Outer folds: every development participant (all REF states) is assigned to one of 5 folds in each of 3 repeats, stratified on the
baseline REF state, so every candidate model is scored on identical participants and indeterminate participants are only ever
training data. Pure functions; no file access.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from yoad.experiments.visit_anchor_audit import design_lib as frozen
from yoad.multimodal import registry as R
from yoad.features.anchor_predictors import attach_predictors, clean_measurements, eligible_anchors
from yoad.data.reference_outcome import EVENT, NONEVENT

KEY = ["participant_id", "anchor_month"]
FROZEN_LONG = ("mmse", "adas13", "faq")
ADMIN_CENSOR_DAYS = 24 * 30.4375 + 90          # 24 months + 90 days, in days
RISK_HORIZON_DAYS = 730.5                       # 24 months, the REF horizon, in days


def _windows(timing: str, variables) -> tuple[dict, int]:
    lo, hi = R.WINDOWS[timing]
    return {v: (lo, hi) for v in variables}, hi


def attach_instrument(anchors: pd.DataFrame, table: pd.DataFrame, features: list, timing: str, data_cutoff,
                      prefix: str) -> tuple[pd.DataFrame, dict]:
    """One assessment (participant, date) per anchor from a wide table; returns (KEY + features + <prefix>_date,
    <prefix>_offset_days) and an aggregate flow."""
    (lo, hi) = R.WINDOWS[timing]
    t = table[["participant_id", "date", *features]].copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    flow = {f"{prefix}_assessments_in": int(len(t)), f"{prefix}_undated": int(t["date"].isna().sum())}
    t = t.dropna(subset=["participant_id", "date"])
    allnan = t[features].isna().all(axis=1)
    flow[f"{prefix}_no_values"] = int(allnan.sum()); t = t[~allnan]
    future = t["date"] > pd.Timestamp(data_cutoff)
    flow[f"{prefix}_after_data_cutoff"] = int(future.sum()); t = t[~future].drop_duplicates()
    clash = t.duplicated(["participant_id", "date"], keep=False)
    flow[f"{prefix}_conflicting_same_date"] = int(clash.sum()); t = t[~clash]
    key = anchors[KEY + ["anchor_date"]]
    j = key.merge(t, on="participant_id", how="inner")
    j["off"] = (j["date"] - j["anchor_date"]).dt.days
    flow[f"{prefix}_candidates_after_index_time"] = int((j["off"] > hi).sum())
    j = j[(j["off"] >= lo) & (j["off"] <= hi)].copy()
    j["absoff"] = j["off"].abs()
    j = j.sort_values(KEY + ["absoff", "off"]).groupby(KEY).head(1)
    out = key.merge(j[KEY + features + ["date", "off"]].rename(columns={"date": f"{prefix}_date", "off": f"{prefix}_offset_days"}),
                    on=KEY, how="left").drop(columns=["anchor_date"])
    late = out[f"{prefix}_date"].notna() & (out[f"{prefix}_offset_days"] > hi)
    early = out[f"{prefix}_date"].notna() & (out[f"{prefix}_offset_days"] < lo)
    if late.any() or early.any():
        raise AssertionError(f"{prefix}: assessment outside the predictor window")
    flow[f"{prefix}_anchors_with_assessment"] = int(out[f"{prefix}_date"].notna().sum())
    return out, flow


def build_frames(visits: pd.DataFrame, measurements: pd.DataFrame, static: pd.DataFrame, data_cutoff, death_dates: dict | None,
                 extra_long: pd.DataFrame | None = None, wide: dict | None = None) -> dict:
    """wide: {prefix: (table, features)}. Returns {'rows': {timing: frame of all eligible anchors}, 'train', 'test', 'flow',
    'labels'}; the frames carry partition = 'train' / 'held_out'."""
    cfg = frozen.DEFAULT_CONFIG
    anchors, flow = eligible_anchors(visits, data_cutoff, u_days=cfg["u_days"], death_dates=death_dates)
    frames, flows = {}, dict(flow)
    labels = None
    phase = (visits[["participant_id", "visit_month", "phase"]].drop_duplicates(["participant_id", "visit_month"])
             .rename(columns={"visit_month": "anchor_month"}) if "phase" in visits.columns else None)
    for timing in R.WINDOWS:
        win = frozen.PREDICTOR_WINDOWS[timing]
        rows, mflow = attach_predictors(anchors, measurements, static, windows={k: tuple(v) for k, v in win["windows"].items()},
                                        post_days=win["post_days"], data_cutoff=data_cutoff)
        if labels is None:                        # labels depend on anchors only; computed once, exactly as run_design
            rows = frozen.label_anchors(visits, rows, data_cutoff, death_dates, "any")
            labels = rows[KEY + ["state_SIM", "reason_SIM", "state_REF", "reason_REF", "split_stratum"]].copy()
        else:
            rows = rows.merge(labels, on=KEY, how="left")
        if phase is not None:
            rows = rows.merge(phase, on=KEY, how="left")
        flows.update({f"{timing}_{k}": v for k, v in mflow.items()})
        if extra_long is not None and len(extra_long):
            ex_vars = sorted(set(extra_long["variable"]))
            w, post = _windows(timing, ex_vars)
            ex, eflow = attach_predictors(anchors, extra_long, None, windows=w, post_days=post, data_cutoff=data_cutoff)
            keep = KEY + [c for v in ex_vars for c in (v, f"{v}_date", f"{v}_offset_days")]
            rows = rows.merge(ex[keep], on=KEY, how="left")
            flows.update({f"{timing}_{k}": v for k, v in eflow.items()})
        for prefix, (table, feats) in (wide or {}).items():
            att, iflow = attach_instrument(anchors, table, list(feats), timing, data_cutoff, prefix)
            rows = rows.merge(att, on=KEY, how="left")
            flows.update({f"{timing}_{k}": v for k, v in iflow.items()})
        frames[timing] = rows
    train, test = frozen.split_participants(frames["primary"], cfg["seed"], cfg["train_fraction"])
    for t in frames.values():
        t["partition"] = np.where(t["participant_id"].isin(train), "train", "held_out")
    return {"rows": frames, "train": train, "test": test, "flow": flows}


def outer_folds(dev_baseline: pd.DataFrame) -> pd.DataFrame:
    """participant_id, repeat, fold for every development participant; stratified on the baseline REF state."""
    from sklearn.model_selection import StratifiedKFold
    b = dev_baseline[["participant_id", "state_REF"]].drop_duplicates("participant_id").sort_values("participant_id")
    ids, strata = b["participant_id"].to_numpy(), b["state_REF"].astype(str).to_numpy()
    out = []
    for r in range(R.OUTER_REPEATS):
        skf = StratifiedKFold(n_splits=R.OUTER_SPLITS, shuffle=True, random_state=R.SEED + 1000 * (r + 1))
        for k, (_, te) in enumerate(skf.split(np.zeros(len(ids)), strata)):
            out.append(pd.DataFrame({"participant_id": ids[te], "repeat": r, "fold": k}))
    f = pd.concat(out, ignore_index=True)
    if f.duplicated(["participant_id", "repeat"]).any() or f.groupby("repeat")["participant_id"].nunique().ne(len(ids)).any():
        raise AssertionError("every development participant must be in exactly one test fold per repeat")
    return f


def landmark_frame(rows: pd.DataFrame, measurements: pd.DataFrame, data_cutoff, partition: str = "train") -> tuple[pd.DataFrame, dict]:
    """One landmark row per participant of `partition` (primary timing). Columns: frozen features at the landmark, baseline values
    (<v>_bl, age_bl), OLS slopes per year (<v>_slope_per_year, >= 2 dated assessments spanning >= 180 days), <v>_n_assessments,
    years_since_baseline, state_REF / reason_REF at the landmark."""
    post = frozen.PREDICTOR_WINDOWS["primary"]["post_days"]
    r = rows[rows["partition"] == partition]
    base = r[r["is_baseline"]].set_index("participant_id")
    later = r[(~r["is_baseline"]) & (r["anchor_month"] >= R.LANDMARK_MIN_MONTH)].sort_values(KEY)
    lm = later.groupby("participant_id", as_index=False).head(1).copy()          # earliest; label not consulted
    flow = {"landmark_participants": int(len(lm)), "landmark_partition_participants": int(r["participant_id"].nunique())}
    for v in R.HISTORY_VARIABLES:
        lm[f"{v}_bl"] = lm["participant_id"].map(base[v])
    lm["age_bl"] = lm["participant_id"].map(base["age"])
    lm["baseline_date"] = lm["participant_id"].map(base["anchor_date"])
    lm["years_since_baseline"] = (lm["anchor_date"] - lm["baseline_date"]).dt.days / 365.25
    m, _ = clean_measurements(measurements, data_cutoff)                   # the frozen cleaning rules
    m = m[m["variable"].isin(R.HISTORY_VARIABLES)]
    j = lm[["participant_id", "anchor_date", "baseline_date"]].merge(m, on="participant_id", how="inner")
    j["index_time"] = j["anchor_date"] + pd.Timedelta(days=post)
    j = j[(j["date"] >= j["baseline_date"] - pd.Timedelta(days=90)) & (j["date"] <= j["index_time"])]
    if (j["date"] > j["index_time"]).any():
        raise AssertionError("history summary uses a measurement after the landmark index time")
    for v in R.HISTORY_VARIABLES:
        g = j[j["variable"] == v]
        stats = {}
        for pid, h in g.groupby("participant_id"):
            x = (h["date"] - h["baseline_date"]).dt.days.to_numpy() / 365.25
            y = h["value"].to_numpy(float)
            ok = len(np.unique(x)) >= 2 and (x.max() - x.min()) * 365.25 >= 180
            stats[pid] = (float(np.polyfit(x, y, 1)[0]) if ok else np.nan, len(h))
        lm[f"{v}_slope_per_year"] = lm["participant_id"].map({k: s[0] for k, s in stats.items()})
        lm[f"{v}_n_assessments"] = lm["participant_id"].map({k: s[1] for k, s in stats.items()}).fillna(0).astype(int)
    flow["landmark_REF_determinate"] = int(lm["state_REF"].isin([EVENT, NONEVENT]).sum())
    return lm.drop(columns=["baseline_date"]).reset_index(drop=True), flow


def survival_data(visits: pd.DataFrame, baseline: pd.DataFrame, data_cutoff, death_dates: dict | None) -> pd.DataFrame:
    """participant_id, time_days, event from the harmonised diagnosis visits after each baseline anchor (see module docstring).
    Participants without any later assessment carry no follow-up and are omitted."""
    v = visits.copy(); v["date"] = pd.to_datetime(v["date"], errors="coerce")
    v = v[v["date"].notna() & (v["date"] <= pd.Timestamp(data_cutoff))]
    by = {pid: g for pid, g in v.groupby("participant_id")}
    out = []
    for pid, t0 in zip(baseline["participant_id"], pd.to_datetime(baseline["anchor_date"])):
        g = by.get(pid)
        if g is None:
            continue
        g = g[g["date"] > t0]
        dd = (death_dates or {}).get(pid)
        if dd is not None and pd.notna(dd):
            g = g[g["date"] <= pd.Timestamp(dd)]
        if g.empty:
            continue
        cls = g.groupby("date")["dx"].agg(lambda s: "DEM" if set(s) == {"DEM"} else ("CONFLICT" if "DEM" in set(s) else "NONDEM"))
        dem = cls[cls == "DEM"]
        if len(dem):
            t, e = (dem.index[0] - t0).days, 1
        else:
            t, e = (cls.index.max() - t0).days, 0
        if t > ADMIN_CENSOR_DAYS:
            t, e = ADMIN_CENSOR_DAYS, 0
        if t > 0:
            out.append((pid, float(t), int(e)))
    return pd.DataFrame(out, columns=["participant_id", "time_days", "event"])
