"""Reference 24-month outcome (three-state), derived from the prediction target - independent of src/cohort/endpoint.py.

Version history (kept, not overwritten; earlier code is in research/history/):
  v1.1 (Prompt 18): no upper bound on the non-event ascertainment window.
  v1.2 (Prompt 19): upper bound u, explicit reason precedence, withdrawal / administrative censoring as unknown-status codes.
  v1.3 (Prompt 20, this file): records are first CONSOLIDATED PER DATE, so no result depends on row order or on the string ordering
    of labels. Changes: (a) same-date dementia records with different etiology (AD + non-AD) or with missing etiology give an UNKNOWN
    dementia type -> INDETERMINATE `dementia_type_uncertain` for the AD endpoint (v1.2 let the alphabetical order 'AD' < 'OTHER'
    decide, and filled a missing etiology with 'AD'); (b) a same-date dementia/non-dementia conflict is checked against the event
    cut-off H + e, not H (v1.2 let a conflict in (H, H + e] pass as non-event in the e > 0 sensitivity); (c) death after H without a
    qualifying assessment has its own code `died_before_ascertainment` (v1.2: `insufficient_followup`); (d) among death / withdrawal /
    data cut-off the EARLIEST dated cause is reported (ties: death > withdrawal > cut-off); (e) uncertain diagnoses count as
    "uncertain in window" only when they precede the first dementia diagnosis; (f) optional sensitivity `type_rule="any_ad"`.
    Full specification: research/reference_outcome_specification.md (v1.3).

Estimand target (primary): was a clinical diagnosis of AD dementia RECORDED at a dated visit on or before H = index date + 24
calendar months (pandas DateOffset; 29 Feb + 24 months -> 28 Feb)? This is the recorded-diagnosis estimand, not the underlying
(interval-censored) conversion time.
  EVENT          - the first dated dementia diagnosis (after per-date consolidation) is on/before H and is AD-type for endpoint="ad";
                   later diagnoses never change this in the primary analysis (reversion only matters under confirm=True)
  NONEVENT       - no dementia, death or conflicting diagnosis on/before the event cut-off, AND a dated non-dementia diagnosis in
                   [H - w, H + u] that precedes any dementia diagnosis
  INDETERMINATE  - everything else, with exactly one reason code chosen by REASON_PRECEDENCE; never coded NONEVENT
Per-date consolidation (all records of one participant on one calendar date become one diagnosis):
  any DEM and any NONDEM -> CONFLICT (uncertain; neither event nor non-event evidence, and a potential event);
  DEM only               -> DEM with type AD (all AD), OTHER (all non-AD) or UNKNOWN (mixed or missing etiology);
  NONDEM only            -> NONDEM;  neither (missing/unclassifiable diagnosis) -> UNK (no information).
Primary w = u = 90 days (operational tolerance, NOT the ADNI protocol visit window). u_days=None removes the upper bound.
event_grace_days e (default 0) extends the EVENT/conflict cut-off to H + e (visit-window sensitivity; 28 d = ADNI-2/ADNI3 14-day
scheduling window + 14 days to complete); death keeps H.
Data-quality rules: undated visits and visits on/before the index date are not outcome observations; visits dated after a recorded
death are discarded and flagged; exact duplicate rows collapse. If no dem_type column is supplied, dementia is assumed AD-type
(documented ADNI assumption, flagged `dem_type_assumed_ad`); if the column is supplied, a missing value is UNKNOWN, never AD.
What it cannot establish: true onset or pathology; it reflects when a diagnosis was recorded under the cohort's diagnostic process,
which uses some of the same instruments as the predictors (MMSE, FAQ, CDR).
"""
from __future__ import annotations

import pandas as pd

EVENT, NONEVENT, INDET = "EVENT", "NONEVENT", "INDETERMINATE"

# Evaluated top to bottom; the first rule that applies fixes (state, reason). The order is part of the protocol.
REASON_PRECEDENCE = (
    (INDET, "competing_non_ad_dementia"),               # R1  first dementia on/before cut-off is non-AD (AD endpoint only)
    (INDET, "dementia_type_uncertain"),                 # R2  first dementia on/before cut-off has UNKNOWN etiology (AD endpoint only)
    (INDET, "unconfirmed_dementia_reverted"),           # R3  confirm=True sensitivity only
    (EVENT, "dementia_on_or_before_H"),                 # R4
    (INDET, "competing_death"),                         # R5  death on/before H without prior dementia
    (INDET, "conflicting_diagnoses_on_or_before_H"),    # R6  a DEM/NONDEM same-date conflict on/before the event cut-off
    (NONEVENT, "nondementia_observed_in_window"),       # R7  qualifying assessment in [H - w, H + u]
    (INDET, "uncertain_diagnosis_in_window"),           # R8  uncertain/conflicting diagnosis in the window, before any dementia
    (INDET, "nondementia_only_beyond_window"),          # R9  non-dementia observed only after H + u
    (INDET, "straddles_H"),                             # R10 first dementia after the cut-off, no qualifying assessment before it
    (INDET, "died_before_ascertainment"),               # R11 unknown status, cause = death in (H, H + u]   } earliest dated cause
    (INDET, "withdrawn_before_ascertainment"),          # R11 unknown status, cause = withdrawal <= H + u   } wins; ties death >
    (INDET, "administratively_censored"),               # R11 unknown status, cause = data cut-off < H + u  } withdrawal > cut-off
    (INDET, "insufficient_followup"),                   # R12 loss to follow-up / missed visits without a recorded reason
)
REASONS = tuple(r for _, r in REASON_PRECEDENCE)


def _ts(x):
    return None if x is None or pd.isna(x) else pd.Timestamp(x)


def _consolidate(v: pd.DataFrame) -> pd.DataFrame:
    """One row per date: cls in {DEM, NONDEM, CONFLICT, UNK}, typ in {AD, OTHER, UNKNOWN, ''}. Independent of row order."""
    rows = []
    for d, g in v.groupby("date", sort=True):
        dx = set(g["dx"])
        if "DEM" in dx and "NONDEM" in dx:
            rows.append((d, "CONFLICT", ""))
        elif "DEM" in dx:
            t = set(g.loc[g["dx"] == "DEM", "typ"])
            rows.append((d, "DEM", next(iter(t)) if len(t) == 1 else "UNKNOWN"))
        elif "NONDEM" in dx:
            rows.append((d, "NONDEM", ""))
        else:
            rows.append((d, "UNK", ""))
    return pd.DataFrame(rows, columns=["date", "cls", "typ"])


def reference_state(index_date, visits: pd.DataFrame, horizon_months: int = 24, w_days: int = 90, u_days: int | None = 90,
                    endpoint: str = "ad", confirm: bool = False, death_date=None, withdrawal_date=None,
                    data_cutoff=None, event_grace_days: int = 0, type_rule: str = "first") -> tuple[str, str, dict]:
    """visits: columns date, dx in {'DEM','NONDEM', other/NaN}, dem_type in {'AD','OTHER'} (optional). Returns (state, reason, flags).
    type_rule: 'first' (primary: the etiology of the FIRST dementia diagnosis decides) or 'any_ad' (sensitivity: an AD-type dementia
    diagnosis on/before the cut-off makes it an event even if a non-AD or unknown-type diagnosis came first)."""
    if endpoint not in ("ad", "any"):
        raise ValueError("endpoint must be 'ad' or 'any'")
    if type_rule not in ("first", "any_ad"):
        raise ValueError("type_rule must be 'first' or 'any_ad'")
    if w_days < 0 or event_grace_days < 0 or (u_days is not None and u_days < 0):
        raise ValueError("tolerances must be non-negative")
    t0 = pd.Timestamp(index_date); H = t0 + pd.DateOffset(months=horizon_months)
    lo = H - pd.Timedelta(days=w_days)
    hi = None if u_days is None else H + pd.Timedelta(days=u_days)
    He = H + pd.Timedelta(days=event_grace_days)
    in_win = lambda d: (d >= lo) & (d <= (hi if hi is not None else pd.Timestamp.max))
    death, wd, cut = _ts(death_date), _ts(withdrawal_date), _ts(data_cutoff)

    v = visits.copy()
    has_type = "dem_type" in v
    if not has_type: v["dem_type"] = "AD"
    v["typ"] = v["dem_type"].where(v["dem_type"].isin(["AD", "OTHER"]), "UNKNOWN")
    v["date"] = pd.to_datetime(v["date"], errors="coerce")
    v = v[["date", "dx", "typ"]].drop_duplicates()
    flags = {"n_undated": int(v["date"].isna().sum()), "n_pre_index": int((v["date"] <= t0).sum()), "n_after_death": 0,
             "reverted": False, "same_date_conflict": False, "dem_type_assumed_ad": not has_type}
    v = v[v["date"].notna() & (v["date"] > t0)]
    if death is not None:
        flags["n_after_death"] = int((v["date"] > death).sum())
        v = v[v["date"] <= death]
    v["dx"] = v["dx"].where(v["dx"].isin(["DEM", "NONDEM"]), "UNK")
    days = _consolidate(v)
    flags["same_date_conflict"] = bool((days["cls"] == "CONFLICT").any())
    dem = days[days["cls"] == "DEM"]
    first = dem.iloc[0] if len(dem) else None
    if first is not None:
        flags["reverted"] = bool(((days["date"] > first["date"]) & (days["cls"] == "NONDEM")).any())
        flags["first_dementia_type"] = first["typ"]

    # R1-R4: first dementia diagnosis on/before the event cut-off
    if first is not None and first["date"] <= He:
        ad_by_cut = bool(((dem["date"] <= He) & (dem["typ"] == "AD")).any())
        if endpoint == "ad" and first["typ"] != "AD" and not (type_rule == "any_ad" and ad_by_cut):
            return INDET, ("competing_non_ad_dementia" if first["typ"] == "OTHER" else "dementia_type_uncertain"), flags
        if confirm and flags["reverted"]:
            return INDET, "unconfirmed_dementia_reverted", flags
        return EVENT, "dementia_on_or_before_H", flags
    # R5 death on/before H (no dementia recorded before it)
    if death is not None and death <= H:
        return INDET, "competing_death", flags
    # R6 conflicting diagnoses on/before the event cut-off
    if ((days["cls"] == "CONFLICT") & (days["date"] <= He)).any():
        return INDET, "conflicting_diagnoses_on_or_before_H", flags
    pre = days if first is None else days[days["date"] < first["date"]]       # only diagnoses preceding any dementia
    # R7 qualifying assessment
    nd = pre[pre["cls"] == "NONDEM"]
    if in_win(nd["date"]).any():
        return NONEVENT, "nondementia_observed_in_window", flags
    # R8 uncertain / conflicting diagnosis inside the window (conflicts after the cut-off can still reach here)
    unc = pre[pre["cls"].isin(["UNK", "CONFLICT"])]
    if in_win(unc["date"]).any():
        return INDET, "uncertain_diagnosis_in_window", flags
    # R9 non-dementia observed only beyond the window
    if hi is not None and (nd["date"] > hi).any():
        return INDET, "nondementia_only_beyond_window", flags
    # R10 first dementia after the cut-off, no qualifying assessment before it
    if first is not None:
        return INDET, "straddles_H", flags
    # R11-R12 unknown status: earliest dated recorded cause within the ascertainment horizon, else unexplained
    end = hi if hi is not None else pd.Timestamp.max
    causes = []
    if death is not None and H < death <= end: causes.append((death, 0, "died_before_ascertainment"))
    if wd is not None and wd <= end: causes.append((wd, 1, "withdrawn_before_ascertainment"))
    if cut is not None and cut < end: causes.append((cut, 2, "administratively_censored"))
    if causes:
        return INDET, min(causes)[2], flags
    return INDET, "insufficient_followup", flags
