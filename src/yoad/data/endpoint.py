"""Fixed-horizon endpoint labelling with explicit handling of insufficient ascertainment.

Two NON-interchangeable definitions (pending supervisor approval; synthetic-tested only):

Rule "A" - cumulative incidence by the horizon H:
    label 1  if the first dementia diagnosis is dated on/before t0 + H;
    label 0  if no dementia is documented by H AND non-dementia status is actually observed at/after t0 + H - g
             (g = ascertainment grace in days; g = 0 is strict);
    label NaN (unlabelled, never 0) when ascertainment is insufficient or the status at H is interval-censored,
             when death precedes H without dementia (default), or the only covering visits have missing/uncertain diagnoses.
Rule "B" - dementia status at a visit inside a window [lo, hi] months around H (default 21-27):
    label = dementia status at the in-window visit closest to H. It can include events after H and it excludes
    participants without an in-window visit; it is NOT a 24-month cumulative-incidence endpoint.

Per-date consolidation (Prompt 20): all records of one participant on one date are merged BEFORE labelling, so row order cannot matter. A date
with both dementia and non-dementia is UNCERTAIN (v1.0-1.2 counted it as dementia); a conflict on/before H with no clean dementia on/before H
makes the participant unlabelled (`conflicting_diagnoses_on_or_before_H`); a dementia date whose records disagree on etiology, or lack one when
a dem_type column is supplied, has type UNKNOWN (unlabelled for the AD endpoint, reason `dementia_type_uncertain`). Known remaining difference
from src/cohort/reference_outcome.py: no upper bound on the confirming non-dementia visit (Rule A) - see tests.
Dementia -> non-dementia reversion: under A the first dementia diagnosis still defines the event (flagged `reverted`);
under B the in-window status is used. Non-AD dementia is a competing event for the AD endpoint (-> unlabelled).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEM, NONDEM, UNCERTAIN = "DEM", "NONDEM", "UNCERTAIN"


def _add_months(d: pd.Timestamp, m: int) -> pd.Timestamp:
    return d + pd.DateOffset(months=m)


def _consolidate_per_date(v: pd.DataFrame) -> pd.DataFrame:
    """Merge records of the same participant and date (order-independent). Returns columns participant_id, visit_date, dx, dem_type."""
    v = v.copy(); v["visit_date"] = pd.to_datetime(v["visit_date"])
    v["dem_type"] = v["dem_type"].where(v["dem_type"].isin(["AD", "OTHER"]), "UNKNOWN")
    rows = []
    for (pid, d), g in v.groupby(["participant_id", "visit_date"], sort=True):
        dx = set(g["dx"])
        if DEM in dx and NONDEM in dx: rows.append((pid, d, UNCERTAIN, "", True))
        elif DEM in dx:
            t = set(g.loc[g["dx"] == DEM, "dem_type"]); rows.append((pid, d, DEM, next(iter(t)) if len(t) == 1 else "UNKNOWN", False))
        elif NONDEM in dx: rows.append((pid, d, NONDEM, "", False))
        else: rows.append((pid, d, UNCERTAIN, "", False))
    return pd.DataFrame(rows, columns=["participant_id", "visit_date", "dx", "dem_type", "conflict"])


def label_endpoint(base: pd.DataFrame, visits: pd.DataFrame, rule: str = "A", horizon_months: int = 24,
                   grace_days: int = 90, window_months: tuple[int, int] = (21, 27), endpoint: str = "any_dementia",
                   death_handling: str = "unlabelled") -> pd.DataFrame:
    """base: participant_id, baseline_date, [death_date]. visits: participant_id, visit_date, dx in {DEM, NONDEM, UNCERTAIN/NaN},
    [dem_type in {AD, OTHER}]. Returns one row per participant: label (1/0/NaN), reason, flags and bounds."""
    if rule not in ("A", "B") or endpoint not in ("any_dementia", "ad_dementia") or death_handling not in ("unlabelled", "nonevent"):
        raise ValueError("bad option")
    v = visits.copy(); v["dx"] = v["dx"].where(v["dx"].isin([DEM, NONDEM]), UNCERTAIN)
    if "dem_type" not in v: v["dem_type"] = "AD"
    v = _consolidate_per_date(v)
    out = []
    for _, b in base.iterrows():
        pid, t0 = b["participant_id"], pd.Timestamp(b["baseline_date"]); H = _add_months(t0, horizon_months)
        death = pd.Timestamp(b["death_date"]) if "death_date" in b and pd.notna(b.get("death_date")) else None
        pv = v[(v["participant_id"] == pid) & (pd.to_datetime(v["visit_date"]) > t0)].sort_values("visit_date")
        pv = pv.assign(visit_date=pd.to_datetime(pv["visit_date"]))
        flags = {"reverted": False, "uncertain_dx_seen": bool((pv["dx"] == UNCERTAIN).any()), "interval_censored": False,
                 "competing_nonAD": False, "death_before_H": False}
        label, reason = np.nan, ""
        dem = pv[pv["dx"] == DEM]
        if len(dem) and (dem["visit_date"].iloc[-1] < pv["visit_date"].iloc[-1]) and (pv[pv["visit_date"] > dem["visit_date"].iloc[0]]["dx"] == NONDEM).any():
            flags["reverted"] = True
        if rule == "A":
            first_dem = dem.iloc[0] if len(dem) else None
            if first_dem is not None and first_dem["visit_date"] <= H:
                if endpoint == "ad_dementia" and first_dem["dem_type"] != "AD":
                    flags["competing_nonAD"] = first_dem["dem_type"] == "OTHER"
                    reason = "competing_non_AD_dementia" if first_dem["dem_type"] == "OTHER" else "dementia_type_uncertain"
                else:
                    label, reason = 1.0, "dementia_on_or_before_H"
            elif (pv["conflict"] & (pv["visit_date"] <= H)).any() and not (death is not None and death <= H):
                reason = "conflicting_diagnoses_on_or_before_H"
            else:
                if death is not None and death <= H:
                    flags["death_before_H"] = True
                    label, reason = (0.0, "death_before_H_coded_nonevent_SENSITIVITY") if death_handling == "nonevent" else (np.nan, "death_before_H_competing")
                else:
                    nd = pv[(pv["dx"] == NONDEM) & ((first_dem is None) | (pv["visit_date"] < (first_dem["visit_date"] if first_dem is not None else H)))]
                    last_nd = nd["visit_date"].max() if len(nd) else pd.NaT
                    if pd.notna(last_nd) and last_nd >= H - pd.Timedelta(days=grace_days):
                        label, reason = 0.0, "nondementia_observed_at_or_after_H_minus_grace"
                    else:
                        flags["interval_censored"] = first_dem is not None
                        reason = "interval_censored_across_H" if first_dem is not None else "insufficient_followup"
        else:
            lo, hi = _add_months(t0, window_months[0]), _add_months(t0, window_months[1])
            inw = pv[(pv["visit_date"] >= lo) & (pv["visit_date"] <= hi) & (pv["dx"] != UNCERTAIN)]
            if len(inw):
                row = inw.iloc[(inw["visit_date"] - H).abs().argmin()]
                if row["dx"] == DEM and endpoint == "ad_dementia" and row["dem_type"] != "AD":
                    reason = "competing_non_AD_dementia" if row["dem_type"] == "OTHER" else "dementia_type_uncertain"
                else: label, reason = (1.0 if row["dx"] == DEM else 0.0), "status_at_in_window_visit"
                flags["event_after_H"] = bool(row["dx"] == DEM and row["visit_date"] > H)
            else:
                reason = "no_valid_in_window_visit"
        out.append({"participant_id": pid, "label": label, "reason": reason, "rule": rule, **flags})
    return pd.DataFrame(out)


def audit_labels(labels: pd.DataFrame, base: pd.DataFrame, visits: pd.DataFrame, horizon_months: int = 24, grace_days: int = 90) -> list[str]:
    """Independent check: no negative label may rest on insufficient follow-up, and no unlabelled flag may be silently 0."""
    problems = []
    for _, r in labels[labels["label"] == 0].iterrows():
        b = base[base["participant_id"] == r["participant_id"]].iloc[0]
        t0 = pd.Timestamp(b["baseline_date"]); H = _add_months(t0, horizon_months)
        vv = visits[(visits["participant_id"] == r["participant_id"]) & (visits["dx"] == NONDEM)]
        last = pd.to_datetime(vv["visit_date"]).max() if len(vv) else pd.NaT
        if r["rule"] == "A" and "SENSITIVITY" not in r["reason"] and (pd.isna(last) or last < H - pd.Timedelta(days=grace_days)):
            problems.append(f"{r['participant_id']}: negative label with follow-up ending {last} before H-grace")
    if labels["label"].isna().any() and not labels.loc[labels["label"].isna(), "reason"].astype(bool).all():
        problems.append("unlabelled participants without a reason code")
    return problems


def label_from_summary_dates(df: pd.DataFrame, horizon_months: int = 24, grace_days: int = 90, endpoint: str = "any_dementia",
                             death_handling: str = "unlabelled") -> pd.DataFrame:
    """Vectorised Definition A from participant-level summary dates (re-derivation check for supplied labels).

    Required columns: participant_id, baseline_date, first_dementia_date (NaT if none observed), last_nondementia_date
    (last documented non-dementia visit BEFORE the first dementia visit; NaT if none). Optional: death_date,
    first_dementia_type ('AD'/'OTHER'). Equivalent to rule 'A' of `label_endpoint` (tested)."""
    if endpoint not in ("any_dementia", "ad_dementia") or death_handling not in ("unlabelled", "nonevent"):
        raise ValueError("bad option")
    b = pd.to_datetime(df["baseline_date"]); H = b + pd.DateOffset(months=horizon_months)
    fd = pd.to_datetime(df["first_dementia_date"]); ln = pd.to_datetime(df["last_nondementia_date"])
    dd = pd.to_datetime(df["death_date"]) if "death_date" in df else pd.Series(pd.NaT, index=df.index)
    typ = df["first_dementia_type"] if "first_dementia_type" in df else pd.Series("AD", index=df.index)
    ev_by_h = fd.notna() & (fd <= H)
    comp_nonAD = ev_by_h & (endpoint == "ad_dementia") & (typ == "OTHER")
    unk_type = ev_by_h & (endpoint == "ad_dementia") & ~typ.isin(["AD", "OTHER"])      # missing/mixed etiology is never read as AD
    label = pd.Series(np.nan, index=df.index); reason = pd.Series("", index=df.index)
    label[ev_by_h & ~comp_nonAD & ~unk_type] = 1.0; reason[ev_by_h & ~comp_nonAD & ~unk_type] = "dementia_on_or_before_H"
    reason[comp_nonAD] = "competing_non_AD_dementia"; reason[unk_type] = "dementia_type_uncertain"
    rest = ~ev_by_h
    died = rest & dd.notna() & (dd <= H)
    if death_handling == "nonevent":
        label[died] = 0.0; reason[died] = "death_before_H_coded_nonevent_SENSITIVITY"
    else:
        reason[died] = "death_before_H_competing"
    open_ = rest & ~died
    observed = open_ & ln.notna() & (ln >= H - pd.Timedelta(days=grace_days)) & (fd.isna() | (fd > ln))
    label[observed] = 0.0; reason[observed] = "nondementia_observed_at_or_after_H_minus_grace"
    bad = open_ & ~observed
    reason[bad & fd.notna()] = "interval_censored_across_H"; reason[bad & fd.isna()] = "insufficient_followup"
    return pd.DataFrame({"participant_id": df["participant_id"], "label": label, "reason": reason})
