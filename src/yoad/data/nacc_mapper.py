"""NACC UDS -> the visit and feature format of the ADNI reference pipeline, for independent external validation of an
ADNI-developed, NACC-transportable model. Pure functions; no file access. Synthetic-tested in tests/test_nacc_validation.py.
Specification: research/source_target_variable_mapping.md (2026-10-19 section).

Column names are those used successfully by experiments/master/e05_nacc_extract.py on the #19886 investigator file, plus the
standard UDS names CDRGLOB and HANDED; the runner refuses to start if any is missing.

Mapping (NACC -> ADNI feature name)
  NACCUDSD    1 normal -> CN; 2 impaired-not-MCI -> CN (non-dementia, not MCI; never an anchor); 3 MCI -> MCI; 4 dementia -> DEM
              (all-cause, as in the reference outcome); other codes -> missing diagnosis (dropped, counted)
  visit date  VISITDATE; visit month = months since the participant's first visit (rounded; the first visit = month 0)
  MMSCORE     NACCMMSE 0..30 (UDS v1-3 only; MoCA-era visits have none -> excluded by complete case, counted)
  CDGLOBAL    CDRGLOB in {0, 0.5, 1, 2, 3};  CDRSB  CDRSUM 0..18
  FAQTOTAL    sum of the 10 FAQ items scored 0..3 (8 = not applicable, 9/-4 = missing), prorated to 10 items when >= 8 are
              valid (the project's existing NACC rule, e05); ADNI FAQTOTAL is also a 0..30 total of 10 items
  PTGENDER    NACCSEX 1 male / 2 female -> 0 / 1 (ADNI PTGENDER - 1 uses the same direction)
  AGE         NACCAGE (integer years; ADNI age is continuous)
  PTHAND      HANDED 2 right -> 0, 1 left -> 1 (ADNI PTHAND - 1); 3 ambidextrous and 9 unknown -> missing
  PTEDUCAT    EDUC 0..36 (99 unknown -> missing)
  APOE4       NACCNE4S 0/1/2 (9 -> missing)
Not available in NACC: ADAS-Cog 11/13 and the 323 UCSF FreeSurfer-7 measures, so the transported model excludes them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ["NACCID", "VISITDATE", "NACCVNUM", "NACCUDSD", "NACCAGE", "NACCSEX", "EDUC", "HANDED", "NACCMMSE", "CDRGLOB",
            "CDRSUM", "NACCNE4S"]
FAQ_ITEMS = ["BILLS", "TAXES", "SHOPPING", "GAMES", "STOVE", "MEALPREP", "EVENTS", "PAYATTN", "REMDATES", "TRAVEL"]
DX = {1: "CN", 2: "CN", 3: "MCI", 4: "DEM"}
FEATURES = ["MMSCORE", "CDGLOBAL", "CDRSB", "FAQTOTAL", "PTGENDER", "AGE", "PTHAND", "PTEDUCAT", "APOE4"]


def missing_columns(columns) -> list:
    return [c for c in REQUIRED + FAQ_ITEMS if c not in set(columns)]


def map_nacc(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (visits, features, flow). visits: participant_id, visit_month, date, dx (for the label rules);
    features: participant_id, visit_month and the ADNI-named predictors at that visit."""
    miss = missing_columns(df.columns)
    if miss:
        raise ValueError(f"NACC table lacks columns {miss}; check the delivered data dictionary")
    num = lambda c: pd.to_numeric(df[c], errors="coerce")
    flow = {"rows_in": int(len(df)), "participants_in": int(df["NACCID"].nunique())}
    d = pd.DataFrame({"participant_id": df["NACCID"].astype(str), "date": pd.to_datetime(df["VISITDATE"], errors="coerce"),
                      "dx": num("NACCUDSD").map(DX)})
    flow["rows_undated"] = int(d["date"].isna().sum()); flow["rows_unmapped_diagnosis"] = int(d["dx"].isna().sum())
    F = df[FAQ_ITEMS].apply(pd.to_numeric, errors="coerce").where(lambda x: x.isin([0, 1, 2, 3]))
    nvalid = F.notna().sum(axis=1)
    hand = num("HANDED")
    feats = pd.DataFrame({
        "MMSCORE": num("NACCMMSE").where(lambda x: x.between(0, 30)),
        "CDGLOBAL": num("CDRGLOB").where(lambda x: x.isin([0, 0.5, 1, 2, 3])),
        "CDRSB": num("CDRSUM").where(lambda x: x.between(0, 18)),
        "FAQTOTAL": (F.sum(axis=1) * 10 / nvalid).where(nvalid >= 8),
        "PTGENDER": num("NACCSEX").map({1: 0.0, 2: 1.0}),
        "AGE": num("NACCAGE").where(lambda x: x.between(18, 120)),
        "PTHAND": hand.map({2: 0.0, 1: 1.0}),
        "PTEDUCAT": num("EDUC").where(lambda x: x.between(0, 36)),
        "APOE4": num("NACCNE4S").where(lambda x: x.isin([0, 1, 2])),
    })
    flow["handedness_ambidextrous_or_unknown"] = int((hand.notna() & ~hand.isin([1, 2])).sum())
    d = pd.concat([d, feats], axis=1)
    d = d.dropna(subset=["date", "dx"]).sort_values(["participant_id", "date"], kind="mergesort")
    first = d.groupby("participant_id")["date"].transform("min")
    d["visit_month"] = ((d["date"] - first).dt.days / 30.4375).round().astype(int)
    before = len(d)
    d["_miss"] = d[FEATURES].isna().sum(axis=1)
    d = d.sort_values(["participant_id", "visit_month", "_miss"], kind="mergesort")
    clash = d.groupby(["participant_id", "visit_month"])["dx"].transform("nunique") > 1
    flow["rows_conflicting_diagnosis_same_month"] = int(clash.sum())
    d = d[~clash].drop_duplicates(["participant_id", "visit_month"], keep="first").drop(columns="_miss")
    flow["rows_duplicate_month_dropped"] = before - flow["rows_conflicting_diagnosis_same_month"] - len(d)
    for c in FEATURES:
        flow[f"missing_{c}"] = int(d[c].isna().sum())
    flow["visits_used"] = int(len(d)); flow["participants_used"] = int(d["participant_id"].nunique())
    visits = d[["participant_id", "visit_month", "date", "dx"]].reset_index(drop=True)
    features = d[["participant_id", "visit_month"] + FEATURES].reset_index(drop=True)
    return visits, features, flow


def nacc_samples(visits: pd.DataFrame, features: pd.DataFrame, data_cutoff, window_days: int = 180,
                 event_grace_days: int = 0) -> tuple[pd.DataFrame, dict]:
    """Every MCI visit labelled with the calendar three-state rule (+/- window_days; the same rule and window as the ADNI
    development data), joined to its same-visit predictors. Determinate labels only; complete case is applied by the caller."""
    from yoad.data.reference_outcome import EVENT, NONEVENT
    from yoad.data.visit_anchor_labels import anchors, drop_post_dementia_anchors, label_calendar_reference
    a, n_post = drop_post_dementia_anchors(visits, anchors(visits, "all_mci_visits"))
    lab = label_calendar_reference(visits, a, w_days=window_days, u_days=window_days, data_cutoff=data_cutoff,
                                   event_grace_days=event_grace_days)
    from yoad.data.visit_anchor_labels import state_summary
    flow = {"mci_visits": int(len(lab)), "mci_anchors_after_recorded_dementia_dropped": n_post,
            "death_rule": "not applied (NACC death coding unverified)", **state_summary(lab, "labels_")}
    for (s, r), n in lab.groupby(["state", "reason"]).size().items():
        flow[f"label_{s}_{r}"] = int(n)
    lab = lab[lab["state"].isin([EVENT, NONEVENT])].copy()
    lab["y"] = (lab["state"] == EVENT).astype(int)
    out = lab.rename(columns={"anchor_month": "visit_month"}).merge(features, on=["participant_id", "visit_month"], how="left")
    out = out.rename(columns={"visit_month": "anchor_month", "reason": "label_reason"})
    return out[["participant_id", "anchor_month", "y", "label_reason"] + FEATURES].reset_index(drop=True), flow
