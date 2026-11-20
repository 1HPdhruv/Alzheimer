"""Aggregate modality and cohort feasibility (PROTOCOL.md §4), computed inside the researcher's guarded run.

Availability is measured on the frozen analysis population (eligible protocol-baseline MCI anchors), at the primary predictor
window, separately for the development partition, the held-out partition and both. A modality counts as available for a
participant when the attached assessment exists in the window (wide instruments) or the feature is non-missing (long-format
instruments). Counts 1..10 are suppressed; fractions are withheld when their numerator is suppressed. External cohorts are not
read: their status records the documented prerequisite that is missing (decision log 107, 126; header inventory 2026-10-12).
"""
from __future__ import annotations

import pandas as pd

from yoad.multimodal import registry as R
from yoad.multimodal.metrics import sup
from yoad.data.reference_outcome import EVENT, NONEVENT

MODALITIES = {   # name -> (columns that must be non-missing (any), wide-instrument date column or None)
    "demographics (age, sex, education)": (("age", "female", "education_years"), None),
    "MMSE": (("mmse",), None), "ADAS-Cog13": (("adas13",), None), "FAQ": (("faq",), None), "APOE genotype": (("apoe4",), None),
    "ADAS-Cog11": (("adas11",), None), "CDR-SB": (("cdrsb",), None),
    "RAVLT": (("ravlt_immediate", "ravlt_perc_forgetting"), None), "logical memory delayed": (("ldeltotal",), None),
    "Trails B": (("trabscor",), None), "category fluency (animals)": (("catanimsc",), None),
    "structural MRI (FreeSurfer 7)": ((), "mri_date"), "CSF Elecsys": ((), "csf_date"), "plasma assays": ((), "plasma_date"),
    "amyloid PET": ((), "amy_date"),
}
EXTERNAL = [
    {"cohort": "AIBL", "local_status": "not present",
     "missing_prerequisite": "no AIBL file in the governed data folders (header-only inventory 2026-10-12); requires an AIBL data "
                             "access approval and download, then a variable dictionary and FreeSurfer-compatible MRI",
     "evaluable_here": False},
    {"cohort": "NACC", "local_status": "extract present (investigator_nacc75.csv); not read by this benchmark",
     "missing_prerequisite": "decision log 126: validation blocked until the NACC freeze date, data-dictionary mappings, "
                             "transport sign-offs, a NACC event rule and runbook step-5 readiness are documented; AI processing "
                             "of NACC participant data is not permitted (ticket 4307). The earlier aggregate audit (2026-10-15, "
                             "quarantined, unverified) found no ADAS-Cog13, regional MRI or plasma in the extract, so only the "
                             "transportable reduced model FS08b could be validated. Cohorts are not pooled (decision log 107).",
     "evaluable_here": False},
]


def _avail(base: pd.DataFrame, cols, date_col) -> pd.Series:
    if date_col is not None:
        return base[date_col].notna() if date_col in base.columns else pd.Series(False, index=base.index)
    present = [c for c in cols if c in base.columns]
    return base[present].notna().any(axis=1) if present else pd.Series(False, index=base.index)


def modality_availability(rows_primary: pd.DataFrame) -> pd.DataFrame:
    base = rows_primary[rows_primary["is_baseline"]]
    out = []
    for part, b in (("development", base[base["partition"] == "train"]), ("held_out", base[base["partition"] == "held_out"]),
                    ("all", base)):
        lab = b["state_REF"].isin([EVENT, NONEVENT])
        ev = b["state_REF"] == EVENT
        for name, (cols, dcol) in MODALITIES.items():
            a = _avail(b, cols, dcol)
            n_lab_av = int((a & lab).sum())
            out.append({"partition": part, "modality": name, "eligible_participants": sup(len(b)),
                        "available": sup(a.sum()), "ref_labelled": sup(lab.sum()), "ref_labelled_available": sup(n_lab_av),
                        "events_available": sup((a & ev).sum()), "nonevents_available": sup((a & lab & ~ev).sum()),
                        "missing_fraction_labelled": (float(1 - n_lab_av / lab.sum())
                                                      if lab.sum() >= R.K_SUPPRESS and (n_lab_av == 0 or n_lab_av >= R.K_SUPPRESS)
                                                      and (lab.sum() - n_lab_av == 0 or lab.sum() - n_lab_av >= R.K_SUPPRESS)
                                                      else None)})
    return pd.DataFrame(out)


def availability_by_phase(rows_primary: pd.DataFrame) -> pd.DataFrame:
    """REF-labelled development participants with each modality, by ADNI phase of the baseline visit."""
    base = rows_primary[rows_primary["is_baseline"] & (rows_primary["partition"] == "train")
                        & rows_primary["state_REF"].isin([EVENT, NONEVENT])]
    phase = base["phase"].astype(str) if "phase" in base.columns else pd.Series("not recorded", index=base.index)
    out = []
    for name, (cols, dcol) in MODALITIES.items():
        a = _avail(base, cols, dcol)
        for ph in sorted(phase.unique()):
            m = phase == ph
            out.append({"modality": name, "phase": ph, "ref_labelled": sup(m.sum()), "available": sup((a & m).sum()),
                        "events_available": sup((a & m & (base["state_REF"] == EVENT)).sum())})
    return pd.DataFrame(out)


def baseline_characteristics(rows_primary: pd.DataFrame) -> pd.DataFrame:
    """Table-1 summaries of the frozen baseline predictors by partition and label subset (the manuscript's outstanding author
    item 3, final_quality_check.md). Groups below 11 participants report only '<11'; a count or percentage whose numerator or
    complement is 1..10 is withheld."""
    base = rows_primary[rows_primary["is_baseline"]]
    subsets = {"eligible": lambda b: b, "SIM_labelled": lambda b: b[b["state_SIM"].isin([EVENT, NONEVENT])],
               "REF_labelled": lambda b: b[b["state_REF"].isin([EVENT, NONEVENT])],
               "REF_event": lambda b: b[b["state_REF"] == EVENT], "REF_nonevent": lambda b: b[b["state_REF"] == NONEVENT],
               "REF_not_labelled": lambda b: b[~b["state_REF"].isin([EVENT, NONEVENT])]}
    ok = lambda k, n: k == 0 or (k >= R.K_SUPPRESS and (n - k == 0 or n - k >= R.K_SUPPRESS))
    out = []
    for part, b in (("development", base[base["partition"] == "train"]), ("held_out", base[base["partition"] == "held_out"]),
                    ("all", base)):
        for name, f in subsets.items():
            g = f(b); n = len(g)
            row = {"partition": part, "subset": name, "participants": sup(n)}
            if n >= R.K_SUPPRESS:
                for v in ("age", "education_years", "adas13"):
                    x = g[v].dropna()
                    row[f"{v}_mean"], row[f"{v}_sd"] = (round(float(x.mean()), 2), round(float(x.std(ddof=1)), 2)) \
                        if len(x) >= R.K_SUPPRESS else (None, None)
                for v in ("mmse", "faq"):
                    x = g[v].dropna()
                    row[f"{v}_median"], row[f"{v}_q1"], row[f"{v}_q3"] = (
                        (float(x.median()), float(x.quantile(.25)), float(x.quantile(.75))) if len(x) >= R.K_SUPPRESS
                        else (None, None, None))
                for label, k, d in (("female", int((g["female"] == 1).sum()), int(g["female"].notna().sum())),
                                    ("apoe4_carrier", int((g["apoe4"] >= 1).sum()), int(g["apoe4"].notna().sum()))):
                    row[f"{label}_n"] = sup(k)
                    row[f"{label}_pct"] = round(100 * k / d, 1) if d and ok(k, d) else None
                for v in R.FROZEN_FEATURES:
                    row[f"missing_{v}"] = sup(int(g[v].isna().sum()))
            out.append(row)
    return pd.DataFrame(out)


def cohort_matrix(availability: pd.DataFrame) -> pd.DataFrame:
    dev = availability[availability["partition"] == "development"]
    rows = [{"cohort": "ADNI (development partition)", "modality": r["modality"], "ref_labelled_available": r["ref_labelled_available"],
             "events_available": r["events_available"], "missing_fraction_labelled": r["missing_fraction_labelled"],
             "local_status": "present; measured in this run", "evaluable_here": True, "missing_prerequisite": ""}
            for _, r in dev.iterrows()]
    for e in EXTERNAL:
        rows.append({"cohort": e["cohort"], "modality": "all", "ref_labelled_available": None, "events_available": None,
                     "missing_fraction_labelled": None, "local_status": e["local_status"], "evaluable_here": e["evaluable_here"],
                     "missing_prerequisite": e["missing_prerequisite"]})
    return pd.DataFrame(rows)


def assert_aggregate_only(tables: dict) -> None:
    """No identifier, date or per-participant column may leave the run (extends discordance_lib.assert_aggregate_only)."""
    forbidden = {"participant_id", "RID", "PTID", "SUBJID", "anchor_date", "index_time", "EXAMDATE", "VISDATE", "test_ids",
                 "time_days", "p_test", "date"}
    for name, t in tables.items():
        if not isinstance(t, pd.DataFrame):
            continue
        bad = forbidden & set(t.columns) | {c for c in t.columns if str(c).endswith("_date")}
        if bad:
            raise ValueError(f"row-level field(s) {sorted(bad)} in aggregate output {name}")
        for c in ("participants", "events", "nonevents", "n"):
            if c in t.columns:
                v = pd.to_numeric(t[c], errors="coerce")
                if ((v >= 1) & (v < R.K_SUPPRESS)).any():
                    raise ValueError(f"unsuppressed small count in {name}.{c}")
