"""Aggregate modality-overlap counts for the biomarker experiment (Prompt 22). NOT executed by the AI assistant on real files.

Input (built locally, inside the isolated environment):
  cohort: one row per participant - participant_id, index_date, state in {EVENT, NONEVENT, INDETERMINATE} (from
          src.cohort.reference_outcome), reason (reason code)
  measurements: dict modality -> DataFrame(participant_id, date) with one row per measurement that passed the modality's
          own completeness rule (e.g. all three Roche analytes present; pT217_F not a -4 code)
Timing rule (prespecified): a measurement counts as BASELINE only if  index_date - lookback_days <= date <= index_date
(no post-index measurement is ever a baseline predictor). `post_index_grace_days` > 0 is a labelled sensitivity count only.
Output: long-format aggregate rows (unique participants, never visits or assay rows), small cells suppressed (< k).
"""
from __future__ import annotations

from itertools import combinations

import pandas as pd

from yoad.audit.local_runner import K_DEFAULT

STATES = ("EVENT", "NONEVENT", "INDETERMINATE")


def baseline_availability(cohort: pd.DataFrame, measurements: dict[str, pd.DataFrame], lookback_days: int = 180,
                          post_index_grace_days: int = 0) -> pd.DataFrame:
    """Participant x modality boolean matrix (index participant_id). Unique participants only."""
    if cohort["participant_id"].duplicated().any():
        raise ValueError("cohort must have one row per participant")
    if lookback_days < 0 or post_index_grace_days < 0:
        raise ValueError("windows must be non-negative")
    idx = cohort.set_index("participant_id")["index_date"].map(pd.Timestamp)
    out = pd.DataFrame(index=idx.index)
    for m, df in measurements.items():
        d = df[["participant_id", "date"]].copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d[d["date"].notna() & d["participant_id"].isin(idx.index)]
        t0 = d["participant_id"].map(idx)
        ok = (d["date"] >= t0 - pd.Timedelta(days=lookback_days)) & (d["date"] <= t0 + pd.Timedelta(days=post_index_grace_days))
        out[m] = out.index.isin(d.loc[ok, "participant_id"].unique())
    return out


def _counts(mask: pd.Series, states: pd.Series) -> dict:
    s = states[mask]
    return {"n": int(mask.sum()), **{f"n_{k.lower()}": int((s == k).sum()) for k in STATES}}


def overlap_matrix(cohort: pd.DataFrame, avail: pd.DataFrame, core: tuple[str, ...], candidate: str | None = None,
                   max_order: int = 3, k: int = K_DEFAULT) -> pd.DataFrame:
    """Rows: every modality alone, every combination up to `max_order`, the core set, core + each other modality, and
    core + candidate. Each row: unique participants with ALL listed modalities at baseline, split by reference state.
    Denominator = the eligible baseline MCI cohort supplied. Cells 1..k-1 suppressed (NA) and flagged."""
    states = cohort.set_index("participant_id")["state"].reindex(avail.index)
    rows, mods = [], list(avail.columns)
    sets = [("all_eligible", ())]
    for r in range(1, max_order + 1):
        sets += [("+".join(c), c) for c in combinations(mods, r)]
    sets.append(("CORE:" + "+".join(core), tuple(core)))
    for m in mods:
        if m not in core:
            sets.append(("CORE+" + m, tuple(core) + (m,)))
    if candidate:
        sets.append(("CORE+CANDIDATE:" + candidate, tuple(core) + (candidate,)))
    seen = set()
    for label, cols in sets:
        if label in seen:
            continue
        seen.add(label)
        mask = avail[list(cols)].all(axis=1) if cols else pd.Series(True, index=avail.index)
        rows.append({"set": label, "modalities": "+".join(cols), **_counts(mask, states)})
    out = pd.DataFrame(rows)
    out["denominator_n"] = len(avail)
    num = [c for c in out.columns if c.startswith("n")]
    out["suppressed"] = False
    for c in num:
        small = (out[c] > 0) & (out[c] < k)
        out.loc[small, "suppressed"] = True
        out[c] = out[c].astype("Int64").mask(small)
    return out


def missingness_summary(avail: pd.DataFrame, k: int = K_DEFAULT) -> pd.DataFrame:
    n = len(avail)
    rows = [{"modality": m, "available_n": int(avail[m].sum()), "missing_n": int((~avail[m]).sum()), "denominator_n": n}
            for m in avail.columns]
    out = pd.DataFrame(rows)
    for c in ("available_n", "missing_n"):
        out[c] = out[c].astype("Int64").mask((out[c] > 0) & (out[c] < k))
    return out
