"""Pure functions for the visit-anchored label audit (Workstream B). No file access; synthetic-tested in
tests/test_visit_anchor_audit_lib.py. The guarded real-data entry point is run_visit_anchor_audit.py (researcher only)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from yoad.data.reference_outcome import EVENT, NONEVENT
from yoad.data.visit_anchor_labels import anchors, label_calendar_reference, label_visitcode_sim2026

DX_CODE = {1: "CN", 2: "MCI", 3: "DEM"}
K = 11


def viscode_month(code) -> float:
    """'bl' -> 0, 'm06' -> 6, 'm120' -> 120; anything else (sc, scmri, init, uns1, y1, ...) -> NaN (not a protocol month)."""
    if not isinstance(code, str):
        return np.nan
    c = code.strip().lower()
    if c == "bl":
        return 0.0
    m = re.fullmatch(r"m(\d{1,3})", c)
    return float(m.group(1)) if m else np.nan


def harmonise_dxsum(df: pd.DataFrame, id_col="RID", visit_col="VISCODE2", date_col="EXAMDATE", dx_col="DIAGNOSIS",
                    phase_col="PHASE") -> tuple[pd.DataFrame, dict]:
    """Map a DXSUM-like table to the visit format of src.cohort.visit_anchor_labels. Returns (visits, aggregate flow)."""
    need = [id_col, visit_col, date_col, dx_col]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"DXSUM-like table lacks columns {miss}")
    flow = {"rows_in": int(len(df))}
    v = pd.DataFrame({"participant_id": df[id_col], "visit_month": df[visit_col].map(viscode_month),
                      "date": pd.to_datetime(df[date_col], errors="coerce"),
                      "dx": pd.to_numeric(df[dx_col], errors="coerce").map(DX_CODE),
                      "phase": df[phase_col] if phase_col in df.columns else "NA"})
    flow["rows_non_protocol_visit"] = int(v["visit_month"].isna().sum())
    flow["rows_missing_dx"] = int(v["dx"].isna().sum())
    for ph, n in v.loc[v["dx"].isna(), "phase"].value_counts().items():      # DIAGNOSIS is documented only for ADNI3/4
        flow[f"rows_missing_dx_phase_{ph}"] = int(n)
    flow["rows_missing_date"] = int(v["date"].isna().sum())
    v = v.dropna(subset=["visit_month", "dx", "date"]).copy()
    v["visit_month"] = v["visit_month"].astype(int)
    # identical duplicates collapse; a participant-visit with conflicting diagnoses is dropped and counted
    v = v.drop_duplicates(["participant_id", "visit_month", "dx"])
    clash = v.duplicated(["participant_id", "visit_month"], keep=False)
    flow["rows_conflicting_same_visit"] = int(clash.sum())
    v = v[~clash].copy()
    flow["rows_used"] = int(len(v)); flow["participants_used"] = int(v["participant_id"].nunique())
    return v, flow


DEATH_IMPUTED = ("as_recorded", "earliest", "latest")


def adsl_death_dates(adsl: pd.DataFrame, imputed: str = "as_recorded") -> tuple[dict, dict]:
    """Death dates from ADNIMERGE2 ADSL. Documented semantics (ADNIMERGE2 manual and Analysis-Meta-Specs vignette):
    SUBJID = as.character(RID); DTHFL = 'Yes' if died; DTHDT = DM.DTHDTC as a date; DTHDTF = 'M' when the day or month was
    missing and imputed. Returns ({RID: death date}, aggregate flow). A flagged death without a usable date cannot be applied
    and is counted, never guessed.
    imputed: PRIMARY 'as_recorded' uses DTHDT as released. Prespecified sensitivity bounds for dates with DTHDTF = 'M' (only the
    year is certain): 'earliest' = 1 January, 'latest' = 31 December of that year. Non-imputed dates are never changed."""
    if imputed not in DEATH_IMPUTED:
        raise ValueError(f"imputed must be one of {DEATH_IMPUTED}")
    a = adsl.copy()
    a["RID"] = pd.to_numeric(a["SUBJID"], errors="coerce")
    a = a.dropna(subset=["RID"]).drop_duplicates("RID")
    d = pd.to_datetime(a["DTHDT"], errors="coerce") if "DTHDT" in a else pd.Series(pd.NaT, index=a.index)
    flagged = a["DTHFL"].astype(str).str.strip().str.lower().eq("yes") if "DTHFL" in a else d.notna()
    imputed_flag = imputed_mask(a)
    if imputed != "as_recorded":
        yr = d.dt.year
        bound = pd.to_datetime(yr.astype("Int64").astype(str) + ("-01-01" if imputed == "earliest" else "-12-31"), errors="coerce")
        d = d.where(~(d.notna() & imputed_mask(a)), bound)
    flow = {"adsl_death_imputation_handling": imputed,
            "adsl_deaths_flagged": int(flagged.sum()), "adsl_deaths_with_date": int(d.notna().sum()),
            "adsl_death_dates_month_or_day_imputed": int((d.notna() & imputed_flag).sum()),
            "adsl_deaths_flagged_without_date": int((flagged & d.isna()).sum())}
    return {int(r): t for r, t in zip(a["RID"], d) if pd.notna(t)}, flow


def imputed_mask(adsl: pd.DataFrame) -> pd.Series:
    return (adsl["DTHDTF"].astype(str).str.strip().str.upper().eq("M") if "DTHDTF" in adsl
            else pd.Series(False, index=adsl.index))


def death_imputation_sensitivity(visits: pd.DataFrame, adsl: pd.DataFrame, data_cutoff=None) -> dict:
    """Aggregate: how many calendar-rule anchor labels differ when imputed death dates are moved to the earliest vs the latest
    date of their year (prespecified sensitivity; the primary analysis uses the recorded dates)."""
    early, _ = adsl_death_dates(adsl, "earliest"); late, _ = adsl_death_dates(adsl, "latest")
    out = {}
    for policy in ("first_mci_visit", "all_mci_visits"):
        a = anchors(visits, policy)
        e = label_calendar_reference(visits, a, data_cutoff=data_cutoff, death_dates=early)
        l = label_calendar_reference(visits, a, data_cutoff=data_cutoff, death_dates=late)
        out[f"death_imputation_label_changes_{policy}"] = int((e["state"] != l["state"]).sum())
    return out


def sup(n):
    n = int(n)
    return n if n == 0 or n >= K else f"<{K}"


def flow_table(flow: dict) -> pd.DataFrame:
    """Export a cohort-flow dict as columns step / n / value. Counts (integers, incl. numpy and whole-number floats) go to `n`
    with the small-count suppression of sup(); descriptive strings (e.g. the death rule used) and None go to `value`, unchanged.
    Anything else (booleans, fractional numbers, other types) raises, so an unexpected value can never bypass suppression."""
    rows = []
    for k, v in flow.items():
        if isinstance(v, (bool, np.bool_)):
            raise TypeError(f"flow value {k!r} is boolean; record it as a count or a descriptive string")
        if isinstance(v, (int, np.integer)) or (isinstance(v, (float, np.floating)) and float(v).is_integer()):
            rows.append({"step": k, "n": sup(v), "value": None})
        elif v is None or isinstance(v, str):
            rows.append({"step": k, "n": None, "value": v})
        else:
            raise TypeError(f"flow value {k!r} has unsupported type {type(v).__name__} ({v!r})")
    return pd.DataFrame(rows, columns=["step", "n", "value"])


def audit(visits: pd.DataFrame, data_cutoff=None, sim_visits: pd.DataFrame | None = None,
          death_dates: dict | None = None) -> dict[str, pd.DataFrame]:
    """All aggregate tables of the audit (cells 1..10 suppressed). sim_visits: optional visit table that also contains undated
    diagnosis rows; the visit-code rule looks outcomes up by visit code only, so it uses them as outcome visits (as the
    reference code and src/cohort/simkhada_dataset do). Anchors always come from dated visits."""
    out = {}
    rows, cross, contrib = [], [], []
    for policy in ("first_mci_visit", "all_mci_visits"):
        a = anchors(visits, policy)
        sim = label_visitcode_sim2026(visits if sim_visits is None else sim_visits, a)
        cal = label_calendar_reference(visits, a, data_cutoff=data_cutoff, death_dates=death_dates)  # deaths: competing (REF only)
        for name, lab in (("visitcode_sim2026", sim), ("calendar_reference_v13", cal)):
            g = lab.groupby(["state", "reason"]).agg(n_anchors=("participant_id", "size"), n_participants=("participant_id", "nunique")).reset_index()
            g.insert(0, "rule", name); g.insert(0, "anchor_policy", policy)
            rows.append(g)
            lbl = lab[lab["state"].isin([EVENT, NONEVENT])]
            contrib.append({"anchor_policy": policy, "rule": name, "n_labelled_anchors": len(lbl),
                            "n_labelled_participants": lbl["participant_id"].nunique(),
                            "event_share_of_labelled_anchors": round(float((lbl["state"] == EVENT).mean()), 4) if len(lbl) else None,
                            "anchors_per_participant_mean": round(float(lbl.groupby("participant_id").size().mean()), 3) if len(lbl) else None,
                            "participants_with_both_labels": lbl.groupby("participant_id")["state"].nunique().gt(1).sum()})
        x = pd.crosstab(sim["state"].rename("sim"), cal["state"].rename("cal")).stack().reset_index()
        x.columns = ["visitcode_sim2026", "calendar_reference_v13", "n_anchors"]
        x.insert(0, "anchor_policy", policy)
        cross.append(x)
        # asymmetry: anchors WITHOUT a diagnosis recorded at the exact m+24 visit code
        has24 = set(zip(visits["participant_id"], visits["visit_month"]))
        no24 = sim[[(p, m + 24) not in has24 for p, m in zip(sim["participant_id"], sim["anchor_month"])]]
        out[f"asymmetry_{policy}"] = no24.groupby("state").size().rename("n_anchors_without_m24_visit").reset_index()
    t = pd.concat(rows, ignore_index=True)
    for c in ("n_anchors", "n_participants"):
        t[c] = t[c].map(sup)
    out["label_counts"] = t
    xc = pd.concat(cross, ignore_index=True); xc["n_anchors"] = xc["n_anchors"].map(sup); out["rule_crosstab"] = xc
    cb = pd.DataFrame(contrib)
    for c in ("n_labelled_anchors", "n_labelled_participants", "participants_with_both_labels"):
        cb[c] = cb[c].map(sup)
    out["design_summary"] = cb
    for k in [k for k in out if k.startswith("asymmetry_")]:
        out[k]["n_anchors_without_m24_visit"] = out[k]["n_anchors_without_m24_visit"].map(sup)
    return out
