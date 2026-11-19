"""Longitudinal sequences for temporal multimodal v1 (PROTOCOL.md §3). Pure functions; no file access.

Definitions
  anchor        an eligible MCI visit of the frozen cohort (multimodal_benchmark.cohort.build_frames, STRICT timing rows); one
                prediction per anchor; label = frozen REF state at that anchor (EVENT 1, NONEVENT 0, INDETERMINATE excluded)
  prediction time  the anchor date itself. Only assessments dated ON OR BEFORE it are used (the frozen +14-day assessment
                window is excluded from this primary experiment). Enforced by construction and asserted.
  modality stream  every cleaned, dated assessment of a modality (cognition/function, MRI, CSF, plasma, amyloid) dated in
                [anchor - HISTORY_YEARS, anchor], most recent MAX_PER_MODALITY kept, oldest first. Each entry keeps its OWN
                elapsed time (assessment date - anchor date) and its gap to the previous assessment of the same modality;
                nothing is snapped to diagnosis visits.
  merged view   (Models B and C) one token per distinct observation date across the streams (most recent MAX_DATES - 1),
                plus a final ANCHOR token at elapsed time 0 that carries no measurement; the anchor token is asserted last
  cleaning      frozen rules: undated, non-numeric, post-cut-off values removed; exact duplicates collapse; a participant /
                variable (or assessment) / date with conflicting values is removed; +/-inf -> missing (counted)
  equal dates   two anchors of one participant on the same date -> the lower protocol month is kept (identical inputs and
                label window); anchors whose DXSUM participant-visit carries conflicting exam dates are excluded (flagged)
  masks         per feature (observed and ACTIVE in the training fold) and per token (real / padding); an unobserved or
                inactive value is stored as 0 together with a 0 mask, never as an unmarked zero
Training-only preprocessing: Preprocessor.fit sees training anchors only and decides which features are active, which
branches are enabled, the MRI plausibility retention, the log / mean / SD statistics and the staleness scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from yoad.features.anchor_predictors import clean_measurements
from yoad.data.reference_outcome import EVENT, NONEVENT
from yoad.longitudinal import config as CF
from yoad.longitudinal import mri_adapter as MA

KEY = ["participant_id", "anchor_month"]
STATIC = tuple(f for b in ("DEM", "APOE") for f in CF.STATIC_BLOCKS[b])
TV = tuple(CF.TV_BLOCKS)
RAW_MRI = tuple(f"raw_{k}" for k in MA.RAW_KEYS)


class TemporalLeakageError(AssertionError):
    """An assessment dated after the prediction anchor, or a mis-ordered sequence, reached the model inputs."""


# ------------------------------------------------------------------------------------------------ dated streams
def _finite(df: pd.DataFrame, cols, flow: dict, name: str) -> pd.DataFrame:
    v = df[cols].apply(pd.to_numeric, errors="coerce")
    inf = np.isinf(v.to_numpy(float))
    flow[f"{name}_infinite_values_set_missing"] = int(inf.sum())
    df = df.copy()
    df[cols] = v.mask(np.isinf(v))
    return df


def _clean_wide(t: pd.DataFrame, feats: list, extra: list, cutoff, flow: dict, name: str) -> pd.DataFrame:
    """The frozen attach_instrument cleaning, applied to the whole table (no anchor window)."""
    t = _finite(t[["participant_id", "date", *feats, *extra]], feats + extra, flow, name)
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    flow[f"{name}_undated"] = int(t["date"].isna().sum())
    t = t.dropna(subset=["participant_id", "date"])
    allnan = t[feats].isna().all(axis=1)
    flow[f"{name}_no_values"] = int(allnan.sum()); t = t[~allnan]
    late = t["date"] > pd.Timestamp(cutoff)
    flow[f"{name}_after_data_cutoff"] = int(late.sum()); t = t[~late].drop_duplicates()
    clash = t.duplicated(["participant_id", "date"], keep=False)
    flow[f"{name}_conflicting_same_date"] = int(clash.sum())
    return t[~clash].sort_values(["participant_id", "date"]).reset_index(drop=True)


def dated_streams(measurements: pd.DataFrame, extra_long: pd.DataFrame | None, wide: dict, data_cutoff) -> tuple[dict, dict]:
    """{block: DataFrame(participant_id, date, features[, raw MRI columns])} with one row per participant and date."""
    flow = {}
    long = pd.concat([measurements] + ([extra_long] if extra_long is not None and len(extra_long) else []), ignore_index=True)
    long = long[long["variable"].isin(CF.TV_BLOCKS["COGFUNC"])].copy()
    vals = pd.to_numeric(long["value"], errors="coerce")
    flow["COGFUNC_infinite_values_set_missing"] = int(np.isinf(vals.to_numpy(float)).sum())
    long["value"] = vals.mask(np.isinf(vals))
    m, mflow = clean_measurements(long, data_cutoff)                 # frozen cleaning rules
    flow.update({f"COGFUNC_{k}": v for k, v in mflow.items()})
    cog = m.pivot_table(index=["participant_id", "date"], columns="variable", values="value", aggfunc="first").reset_index()
    for f in CF.TV_BLOCKS["COGFUNC"]:
        if f not in cog.columns:
            cog[f] = np.nan
    streams = {"COGFUNC": cog[["participant_id", "date", *CF.TV_BLOCKS["COGFUNC"]]].sort_values(["participant_id", "date"])
               .reset_index(drop=True)}
    for b, prefix in CF.WIDE_PREFIX.items():
        feats = list(CF.TV_BLOCKS[b])
        if prefix not in (wide or {}):
            streams[b] = pd.DataFrame(columns=["participant_id", "date", *feats]); continue
        t = wide[prefix][0].copy()
        for f in feats:
            if f not in t.columns:
                t[f] = np.nan
        extra = [c for c in RAW_MRI if c in t.columns] if b == "MRI" else []
        streams[b] = _clean_wide(t, feats, extra, data_cutoff, flow, b)
    for b, t in streams.items():
        if t.duplicated(["participant_id", "date"]).any():
            raise TemporalLeakageError(f"{b}: more than one assessment per participant and date after cleaning")
    return streams, flow


# ------------------------------------------------------------------------------------------------ anchor samples
@dataclass
class Samples:
    """Raw (unstandardised) per-anchor modality streams. Days are (assessment date - anchor date) in days (<= 0)."""
    meta: pd.DataFrame                       # participant_id, anchor_month, anchor_date, is_baseline, partition, y
    SX: dict                                 # block -> (N, MAX_PER_MODALITY, d) values, NaN = not observed / padding
    SD: dict                                 # block -> (N, MAX_PER_MODALITY) days before the anchor (NaN = padding)
    SL: dict                                 # block -> (N,) number of assessments kept
    raw_mri: np.ndarray                      # (N, MAX_PER_MODALITY, len(RAW_MRI)) raw MRI sums (fold-level plausibility)
    static: np.ndarray                       # (N, len(STATIC))
    anchor_frame: pd.DataFrame = field(repr=False, default=None)   # frozen strict anchor-row features (baselines)
    flow: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.meta)

    def subset(self, idx) -> "Samples":
        idx = np.asarray(idx)
        return Samples(self.meta.iloc[idx].reset_index(drop=True), {b: a[idx] for b, a in self.SX.items()},
                       {b: a[idx] for b, a in self.SD.items()}, {b: a[idx] for b, a in self.SL.items()}, self.raw_mri[idx],
                       self.static[idx], None if self.anchor_frame is None else self.anchor_frame.iloc[idx].reset_index(drop=True),
                       dict(self.flow))


def label_from_state(state: pd.Series) -> np.ndarray:
    return np.where(state == EVENT, 1.0, np.where(state == NONEVENT, 0.0, np.nan))


def select_anchors(rows: pd.DataFrame, state_col: str, flagged_visits=frozenset(), determinate_only=True) -> tuple[pd.DataFrame, dict]:
    r = rows.copy()
    r["y"] = label_from_state(r[state_col])
    flow = {"anchors_in": int(len(r))}
    if determinate_only:
        r = r[r["y"].notna()]
    flow["anchors_ref_determinate"] = int(len(r))
    bad = r.set_index(KEY).index.isin(list(flagged_visits)) if len(flagged_visits) else np.zeros(len(r), bool)
    flow["anchors_excluded_conflicting_dxsum_dates"] = int(bad.sum())
    r = r[~bad]
    r = r.sort_values(["participant_id", "anchor_date", "anchor_month"])
    same = r.duplicated(["participant_id", "anchor_date"], keep="first")
    flow["anchors_excluded_equal_date_duplicate"] = int(same.sum())
    r = r[~same].sort_values(KEY).reset_index(drop=True)
    flow["anchors_used"] = int(len(r))
    return r, flow


def build_samples(rows: pd.DataFrame, streams: dict, state_col: str = "state_REF", flagged_visits=frozenset(),
                  determinate_only: bool = True) -> Samples:
    """rows: frozen anchor rows (STRICT timing) with KEY, anchor_date, is_baseline, partition, state_col, static and anchor
    features; streams: dated_streams(). Returns Samples in anchor order."""
    r, flow = select_anchors(rows, state_col, flagged_visits, determinate_only)
    N, L = len(r), CF.MAX_PER_MODALITY
    r["_aid"] = np.arange(N)
    SX, SD, SL = {}, {}, {}
    raw = np.full((N, L, len(RAW_MRI)), np.nan)
    for b, feats in CF.TV_BLOCKS.items():
        t = streams[b]
        j = r[["_aid", "participant_id", "anchor_date"]].merge(t, on="participant_id", how="inner")
        j["day"] = (pd.to_datetime(j["date"]) - j["anchor_date"]).dt.days
        flow[f"{b}_anchor_assessment_pairs_after_anchor_excluded"] = int((j["day"] > CF.POST_DAYS).sum())
        j = j[(j["day"] <= CF.POST_DAYS) & (j["day"] >= -CF.HISTORY_YEARS * 365.25)]
        if (j["day"] > 0).any():
            raise TemporalLeakageError(f"{b}: assessment after the prediction anchor")
        j = j.sort_values(["_aid", "day"])
        j["_from_end"] = j.groupby("_aid").cumcount(ascending=False)
        j = j[j["_from_end"] < L].copy()
        j["_pos"] = j.groupby("_aid").cumcount()
        a, p = j["_aid"].to_numpy(), j["_pos"].to_numpy()
        x = np.full((N, L, len(feats)), np.nan); d = np.full((N, L), np.nan)
        x[a, p] = j[list(feats)].to_numpy(float)
        d[a, p] = j["day"].to_numpy(float)
        if b == "MRI" and all(c in j.columns for c in RAW_MRI):
            raw[a, p] = j[list(RAW_MRI)].to_numpy(float)
        SX[b], SD[b] = x, d
        SL[b] = np.bincount(a, minlength=N).astype(int) if len(a) else np.zeros(N, int)
    static = r[list(STATIC)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    flow["static_infinite_values_set_missing"] = int(np.isinf(static).sum())
    static[np.isinf(static)] = np.nan
    af = r.drop(columns=["_aid"])
    num = af.select_dtypes(include=[np.number]).columns
    flow["anchor_frame_infinite_values_set_missing"] = int(np.isinf(af[num].to_numpy(float)).sum())
    af[num] = af[num].mask(np.isinf(af[num].to_numpy(float)))
    meta = r[["participant_id", "anchor_month", "anchor_date", "is_baseline", "partition", "y"]].copy()
    s = Samples(meta, SX, SD, SL, raw, static, af, flow)
    check_sequences(s)
    return s


def check_sequences(s: Samples) -> None:
    """Run-time checks: no assessment after the anchor, streams strictly increasing in time, padding after real entries."""
    for b in TV:
        d, n = s.SD[b], s.SL[b]
        real = np.arange(d.shape[1])[None, :] < n[:, None]
        if (d[real] > CF.POST_DAYS).any():
            raise TemporalLeakageError(f"{b}: assessment after the prediction anchor")
        if np.isnan(d[real]).any() or not np.isnan(d[~real]).all():
            raise TemporalLeakageError(f"{b}: padding is not after the real entries")
        dd = np.diff(np.where(real, d, 1e12), axis=1)
        if ((dd <= 0) & real[:, 1:]).any():
            raise TemporalLeakageError(f"{b}: assessments not strictly increasing in time")


# ------------------------------------------------------------------------------------------------ derived views
def restrict(s: Samples, keep: dict, latest: bool = False) -> Samples:
    """The EFFECTIVE sample of one candidate / ablation / fold: features not permitted in `keep` ({block: (d,) bool}; all
    False = modality removed) are deleted, assessments left without any permitted finite value are removed, and every stream
    is re-packed (oldest first). With `latest`, each modality keeps only its most recent remaining assessment. Timelines,
    lengths, elapsed times, last observations and staleness built from the result can therefore never be influenced by a
    disabled modality or an inactive feature."""
    SX, SD, SL = {}, {}, {}
    raw = np.full_like(s.raw_mri, np.nan)
    i = np.arange(s.n)[:, None]
    for b in TV:
        k = np.asarray(keep.get(b, np.ones(len(CF.TV_BLOCKS[b]), bool)), bool)
        x = np.where(k[None, None, :], s.SX[b], np.nan)
        L = x.shape[1]
        real = (np.arange(L)[None, :] < s.SL[b][:, None]) & np.isfinite(x).any(axis=2)
        if latest:
            last = np.where(real, np.arange(L)[None, :], -1).max(axis=1)
            real = (np.arange(L)[None, :] == last[:, None]) & (last[:, None] >= 0)
        order = np.argsort(~real, axis=1, kind="stable")             # kept entries first, original (time) order
        n = real.sum(axis=1)
        pad = np.arange(L)[None, :] >= n[:, None]
        nx, nd = x[i, order], s.SD[b][i, order]
        nx[pad] = np.nan; nd[pad] = np.nan
        SX[b], SD[b], SL[b] = nx, nd, n.astype(int)
        if b == "MRI":
            raw = s.raw_mri[i, order]; raw[pad] = np.nan
    return Samples(s.meta, SX, SD, SL, raw, s.static.copy(), s.anchor_frame, dict(s.flow))


def latest_only(s: Samples) -> Samples:
    """History ablation: each modality keeps only its most recent assessment (still dated, still before the anchor)."""
    return restrict(s, {}, latest=True)


def merged_view(SX: dict, SD: dict, SL: dict, max_dates: int = CF.MAX_DATES) -> dict:
    """Distinct observation dates across the given streams (most recent max_dates - 1) + the anchor token (last, elapsed 0).
    The anchor token exists for every anchor, whatever the streams contain."""
    N = len(next(iter(SL.values())))
    parts = []
    for b in TV:
        L = SD[b].shape[1]
        aid, pos = np.nonzero(np.arange(L)[None, :] < SL[b][:, None])
        parts.append(pd.DataFrame({"aid": aid, "day": SD[b][aid, pos], "b": b, "pos": pos}))
    p = pd.concat(parts, ignore_index=True)
    days = p[["aid", "day"]].drop_duplicates().sort_values(["aid", "day"])
    days["from_end"] = days.groupby("aid").cumcount(ascending=False)
    days = days[days["from_end"] < max_dates - 1].copy()
    days["slot"] = days.groupby("aid").cumcount()
    n_obs = days.groupby("aid").size().reindex(range(N), fill_value=0).to_numpy()
    p = p.merge(days[["aid", "day", "slot"]], on=["aid", "day"], how="inner")
    out = {"lengths": n_obs + 1}
    T = max_dates
    dt = np.zeros((N, T)); valid = np.arange(T)[None, :] < out["lengths"][:, None]
    dt[days["aid"].to_numpy(), days["slot"].to_numpy()] = days["day"].to_numpy() / 365.25
    for b in TV:
        q = p[p["b"] == b]
        x = np.full((N, T, SX[b].shape[2]), np.nan)
        x[q["aid"].to_numpy(), q["slot"].to_numpy()] = SX[b][q["aid"].to_numpy(), q["pos"].to_numpy()]
        out[f"X_{b}"] = x
    gap = np.zeros((N, T))
    for n in np.flatnonzero(out["lengths"] > 1):
        L = out["lengths"][n]
        gap[n, 1:L] = np.diff(dt[n, :L])                         # last gap = time from the latest observation to the anchor
    anchor_pos = out["lengths"] - 1
    if not (dt[np.arange(N), anchor_pos] == 0).all() or (dt[valid] > 0).any():
        raise TemporalLeakageError("the anchor token must be last, at elapsed time 0, after sorting and truncation")
    out.update(dt=dt, gap=gap, valid=valid)
    return out


def last_observation(s: Samples) -> tuple[dict, dict]:
    """Per block: (N, d) last observed value over the stream and its staleness in years from the ASSESSMENT date (>= 0)."""
    val, stale = {}, {}
    for b, arr in s.SX.items():
        obs = np.isfinite(arr)
        pos = np.where(obs, np.arange(arr.shape[1])[None, :, None], -1).max(axis=1)
        n_idx, f_idx = np.nonzero(pos >= 0)
        v = np.full(pos.shape, np.nan); t = np.full(pos.shape, np.nan)
        v[n_idx, f_idx] = arr[n_idx, pos[n_idx, f_idx], f_idx]
        t[n_idx, f_idx] = np.maximum(-s.SD[b][n_idx, pos[n_idx, f_idx]] / 365.25, 0.0)
        val[b], stale[b] = v, t
    return val, stale


def block_summary(s: Samples) -> pd.DataFrame:
    """Per anchor and block, each computed from that block ONLY (so removing a block removes all of its columns):
    has_<b> (any assessment), has_<b>_at_anchor (one within 90 days before the anchor), n_<b> (assessments kept)."""
    out = {}
    i = np.arange(s.n)
    for b in TV:
        has = s.SL[b] > 0
        last_day = np.where(has, s.SD[b][i, np.maximum(s.SL[b] - 1, 0)], np.nan)
        out[f"has_{b}"] = has.astype(float)
        out[f"has_{b}_at_anchor"] = (has & (last_day >= CF.WINDOW[0])).astype(float)
        out[f"n_{b}"] = s.SL[b].astype(float)
    return pd.DataFrame(out)


def availability(s: Samples) -> pd.DataFrame:
    """DESCRIPTIVE only (cohort tables, evaluation subsets; never a model input): block_summary plus n_dates (distinct
    observation dates over all modalities) and history_years."""
    out = block_summary(s)
    mv = merged_view(s.SX, s.SD, s.SL)
    out["n_dates"] = (mv["lengths"] - 1).astype(float)
    earliest = np.nanmin(np.column_stack([s.SD[b][:, 0] for b in TV]), axis=1) if s.n else np.zeros(0)
    out["history_years"] = np.where(np.isfinite(earliest), -earliest / 365.25, 0.0)
    return out


def snapshot_frame(s: Samples) -> pd.DataFrame:
    """Matched-feature table for A1_XGB_MATCHED: per block its last observations, their staleness, has_<b>, has_<b>_at_anchor
    and n_<b>, plus the static values. No all-modality summary (e.g. a count of dates over every modality) is included."""
    val, stale = last_observation(s)
    cols = {}
    for b, feats in CF.TV_BLOCKS.items():
        for i, f in enumerate(feats):
            cols[f] = val[b][:, i]
            cols[f"{f}_staleness_years"] = stale[b][:, i]
    for i, f in enumerate(STATIC):
        cols[f] = s.static[:, i]
    return pd.concat([pd.DataFrame(cols), block_summary(s)], axis=1)


# ------------------------------------------------------------------------------------------------ training-only preprocessing
class Preprocessor:
    """Fitted on TRAINING anchors only. Decides, per fold:
      active_[b][k]     feature k of block b has >= MIN_FEATURE_TRAIN_OBS finite training assessments (unique participant /
                        date) and, for MRI, passes the frozen plausibility-median rule on TRAINING raw values
      enabled_[b]       branch b has >= MIN_BRANCH_TRAIN_PARTICIPANTS training participants with an active feature observed,
                        and is not dropped by an ablation
      stats_            log (prespecified list), mean and SD over finite training values of active features
    transform() first builds the EFFECTIVE sample (restrict): disabled or dropped modalities and inactive features are
    deleted BEFORE any timeline, length, elapsed time, last observation or staleness is derived, so a disabled modality can
    change nothing; the anchor token is added independently. `values_masked` keeps masks and times but zeroes values;
    `latest` keeps each modality's most recent remaining assessment (E_SNAP)."""

    def __init__(self, drop=(), values_masked=(), latest: bool = False):
        self.drop = tuple(drop)
        self.values_masked = tuple(values_masked)
        self.latest = bool(latest)

    @staticmethod
    def _log(b, a):
        a = np.array(a, float, copy=True)
        a[~np.isfinite(a)] = np.nan
        for k, f in enumerate(CF.TV_BLOCKS[b]):
            if f in CF.LOG_FEATURES:
                with np.errstate(invalid="ignore", divide="ignore"):
                    a[..., k] = np.log(np.where(a[..., k] > 0, a[..., k], np.nan))
        return a

    def _unique_entries(self, s: Samples, idx, b):
        """Training assessments of block b, one row per (participant, date): values (n, d) and raw MRI (n, r)."""
        L = s.SD[b].shape[1]
        sub_n, sub_p = np.nonzero(np.arange(L)[None, :] < s.SL[b][idx][:, None])
        n = idx[sub_n]
        pid = s.meta["participant_id"].to_numpy()[n]
        date = pd.to_datetime(s.meta["anchor_date"].to_numpy()[n]) + pd.to_timedelta(s.SD[b][n, sub_p], unit="D")
        key = pd.DataFrame({"pid": pid, "date": date})
        keep = ~key.duplicated().to_numpy()
        return s.SX[b][n, sub_p][keep], s.raw_mri[n, sub_p][keep] if b == "MRI" else None, pid[keep]

    def keep(self) -> dict:
        """{block: (d,) bool} permitted features of the fitted fold (active and branch enabled)."""
        return {b: self.active_[b] & self.enabled_[b] for b in TV}

    def fit(self, s: Samples, train_idx) -> "Preprocessor":
        idx = np.asarray(train_idx)
        self.train_ids_ = frozenset(s.meta["participant_id"].to_numpy()[idx])
        self.stats_, self.active_, self.enabled_, self.n_train_obs_ = {}, {}, {}, {}
        for b, feats in CF.TV_BLOCKS.items():
            vals, raw, pid = self._unique_entries(s, idx, b)
            lv = self._log(b, vals)
            fin = np.isfinite(lv)
            n_obs = fin.sum(axis=0)
            active = n_obs >= CF.MIN_FEATURE_TRAIN_OBS
            if b == "MRI" and len(vals):
                med = {k: (float(np.nanmedian(raw[:, i])) if np.isfinite(raw[:, i]).any() else None)
                       for i, k in enumerate(MA.RAW_KEYS)}
                plaus = MA.plausible_in_fold(med)
                active &= np.array([plaus[f] for f in feats])
                self.mri_medians_ = med
            mu, sd = np.zeros(len(feats)), np.ones(len(feats))
            for k in range(len(feats)):
                x = lv[fin[:, k], k]
                if active[k] and len(x) >= 2:
                    mu[k] = x.mean(); sd[k] = x.std() if x.std() > 1e-8 else 1.0
            n_part = len(set(pid[(fin & active[None, :]).any(axis=1)])) if len(pid) else 0
            self.active_[b] = active
            self.enabled_[b] = bool(active.any() and n_part >= CF.MIN_BRANCH_TRAIN_PARTICIPANTS and b not in self.drop)
            self.stats_[b] = (mu, sd)
            self.n_train_obs_[b] = n_obs
        st = s.static[idx].astype(float)
        st[~np.isfinite(st)] = np.nan
        fin = np.isfinite(st)
        self.static_active_ = fin.sum(axis=0) >= CF.MIN_FEATURE_TRAIN_OBS
        for blk, feats in CF.STATIC_BLOCKS.items():
            if blk in self.drop:
                self.static_active_[[STATIC.index(f) for f in feats]] = False
        mu = np.array([st[fin[:, k], k].mean() if fin[:, k].sum() else 0.0 for k in range(st.shape[1])])
        sd = np.array([st[fin[:, k], k].std() if fin[:, k].sum() > 1 else 1.0 for k in range(st.shape[1])])
        self.stats_["STATIC"] = (mu, np.where(sd > 1e-8, sd, 1.0))
        _, stale = last_observation(restrict(s.subset(idx), self.keep(), self.latest))
        self.stale_scale_ = {b: max(float(np.nanmax(t)) if np.isfinite(t).any() else 1.0, 1.0) for b, t in stale.items()}
        return self

    def _z(self, b, a):
        mu, sd = self.stats_[b]
        z = (self._log(b, a) - mu) / sd
        m = np.isfinite(z) & self.active_[b] & self.enabled_[b]
        if b in self.values_masked:
            z = np.zeros_like(z)
        return np.where(m, np.clip(np.nan_to_num(z), -CF.CLIP_Z, CF.CLIP_Z), 0.0).astype(np.float32), m.astype(np.float32)

    def transform(self, s: Samples) -> dict:
        e = restrict(s, self.keep(), self.latest)                      # effective sample: nothing disabled survives
        out = {}
        mv = merged_view(e.SX, e.SD, e.SL)
        out.update(lengths=mv["lengths"], valid=mv["valid"], dt=mv["dt"].astype(np.float32), gap=mv["gap"].astype(np.float32))
        for b in TV:
            out[f"x_{b}"], out[f"m_{b}"] = self._z(b, mv[f"X_{b}"])
            out[f"m_{b}"] *= mv["valid"][:, :, None]
            out[f"sx_{b}"], out[f"sm_{b}"] = self._z(b, e.SX[b])
            L = e.SD[b].shape[1]
            real = np.arange(L)[None, :] < e.SL[b][:, None]
            out[f"sm_{b}"] *= real[:, :, None]
            out[f"sv_{b}"] = real & (out[f"sm_{b}"].sum(-1) > 0)
            t = np.where(real, e.SD[b] / 365.25, 0.0)
            g = np.zeros_like(t); g[:, 1:] = np.where(real[:, 1:], np.diff(np.nan_to_num(e.SD[b]), axis=1) / 365.25, 0.0)
            out[f"st_{b}"], out[f"sg_{b}"] = t.astype(np.float32), g.astype(np.float32)
        val, stale = last_observation(e)
        for b in TV:
            out[f"lx_{b}"], out[f"lm_{b}"] = self._z(b, val[b][:, None, :])
            out[f"lx_{b}"], out[f"lm_{b}"] = out[f"lx_{b}"][:, 0], out[f"lm_{b}"][:, 0]
            out[f"ls_{b}"] = np.where(out[f"lm_{b}"] > 0, np.nan_to_num(stale[b]) / self.stale_scale_[b], 0.0).astype(np.float32)
        mu, sd = self.stats_["STATIC"]
        st = s.static.astype(float); st[~np.isfinite(st)] = np.nan
        z = (st - mu) / sd
        sm = np.isfinite(z) & self.static_active_
        out["x_STATIC"] = np.where(sm, np.clip(np.nan_to_num(z), -CF.CLIP_Z, CF.CLIP_Z), 0.0).astype(np.float32)
        out["m_STATIC"] = sm.astype(np.float32)
        for k, v in out.items():
            if v.dtype.kind == "f" and not np.isfinite(v).all():
                raise ValueError(f"non-finite model input {k}")
        return out

    def summary(self) -> dict:
        return {"active": {b: [f for f, a in zip(CF.TV_BLOCKS[b], self.active_[b]) if a] for b in TV},
                "enabled": dict(self.enabled_), "static_active": [f for f, a in zip(STATIC, self.static_active_) if a],
                "drop": list(self.drop), "values_masked": list(self.values_masked), "latest": self.latest}


def weights(participants: np.ndarray) -> np.ndarray:
    """Participant weights: each participant's anchors sum to 1, so no participant counts more because of more visits."""
    _, inv, cnt = np.unique(participants, return_inverse=True, return_counts=True)
    return (1.0 / cnt)[inv]
