"""Reference-baseline dataset of Simkhada et al. 2026 (GeroScience, doi 10.1007/s11357-026-02551-x), rebuilt from ADNI source
tables. Pure functions; no file access. Synthetic-tested in tests/test_simkhada_baseline.py.

Provenance tags (see literature/PIPELINE_REPRODUCTION_AND_ADAPTATION_AUDIT.md):
  [A] author-reported in the article   [C] read in the public code at commit c67358d (no licence: re-implemented, not copied)
  [D] our documented deviation         [U] unverifiable from paper or code; flagged in the run metadata

Construction
  samples      every MCI diagnosis visit (DXSUM DIAGNOSIS = 2) [A][C]; visit month from VISCODE2 [D: the code uses a numeric
               VISIT column that our DXSUM export may lack]
  label        src.cohort.visit_anchor_labels.label_visitcode_sim2026 (exact visit codes m+6..m+24, asymmetric ascertainment,
               any CN excluded, change at m+6/m+12 then MCI excluded, DEM at m+18 then MCI at m+24 kept as stable) [C]
  outcome      all-cause dementia (DIAGNOSIS = 3) [C]; the article says "progression to AD" [A] - recorded as a discrepancy
  features     joined at the SAME visit (RID + visit month) [C]:
               COG  MMSCORE, CDGLOBAL, CDRSB, TOTSCORE (ADAS-Cog 11), TOTAL13, FAQTOTAL           [A][C]
               DEM  PTGENDER - 1, AGE = (exam date - PTDOB) / 365.25, PTHAND - 1, PTEDUCAT, APOE4    [A][C]
               MRI  UCSFFSX7 FreeSurfer measures with suffix SV/CV/SA/TA/TS (paper: 48/69/70/68/68 = 323)  [A]; selected
                    by pattern unless the exact Table S1 list is supplied [D]
  duplicates   several rows for one participant-visit in a table -> the row with the fewest missing values is kept [C]
  missing      complete case over ALL features of the analysis, so every feature set uses the same samples [A][C]
  negative codes  ADNI missing codes (< 0) in score columns are set missing before the complete-case step [D]; counted
  advisory     March-2026 ADNI advisory participants (PTID 381_S_xxxxx) excluded by default [D: after the paper's download]
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from yoad.data.reference_outcome import EVENT, NONEVENT
from yoad.data.visit_anchor_labels import (anchors, drop_post_dementia_anchors, label_calendar_reference, label_visitcode_sim2026,
                                            state_summary)

COG = ["MMSCORE", "CDGLOBAL", "CDRSB", "TOTSCORE", "TOTAL13", "FAQTOTAL"]
DEM = ["PTGENDER", "AGE", "PTHAND", "PTEDUCAT", "APOE4"]
CATEGORICAL = ["PTGENDER", "PTHAND"]                       # SMOTENC nominal columns [C]
MRI_PATTERN = re.compile(r"^ST\d+(SV|CV|SA|TA|TS)$")
MRI_EXPECTED = {"SV": 48, "CV": 69, "SA": 70, "TA": 68, "TS": 68}    # paper Table S1 counts (CT = TA, CTS = TS) [A]
FEATURE_SETS = {"COG": ("COG",), "DEM": ("DEM",), "MRI": ("MRI",), "COG+DEM": ("COG", "DEM"), "COG+MRI": ("COG", "MRI"),
                "DEM+MRI": ("DEM", "MRI"), "ALL": ("COG", "DEM", "MRI")}   # 7 = all non-empty modality combinations [A]
SOURCE_COLUMNS = {
    "dxsum": ["PTID", "RID", "VISCODE2", "EXAMDATE", "DIAGNOSIS"],
    "mmse": ["RID", "VISCODE2", "MMSCORE", "VISDATE"],
    "cdr": ["RID", "VISCODE2", "CDGLOBAL", "CDRSB"],
    "adas": ["RID", "VISCODE2", "TOTSCORE", "TOTAL13"],
    "faq": ["RID", "VISCODE2", "FAQTOTAL"],
    "apoeres": ["RID"],                                   # APOE4, or GENOTYPE from which the e4 count is derived
    "ptdemog": ["RID", "PTGENDER", "PTDOB", "PTHAND", "PTEDUCAT"],
    "ucsffsx7": ["RID", "VISCODE2"],
}
# predictor_timing="dated_window" (decision log 121; baseline model only): each instrument's values come from ONE dated
# assessment (VISDATE) in [index - 90 d, index], index = DXSUM exam date of the month-0 MCI anchor; nothing after the index
DATED_INSTRUMENTS = {"mmse": ["MMSCORE"], "cdr": ["CDGLOBAL", "CDRSB"], "adas": ["TOTSCORE", "TOTAL13"], "faq": ["FAQTOTAL"]}
BASELINE_DATED_WINDOW = (-90, 0)


def _check(tables: dict) -> None:
    for name, cols in SOURCE_COLUMNS.items():
        if name not in tables:
            raise ValueError(f"missing source table {name!r}")
        miss = [c for c in cols if c not in tables[name].columns]
        if miss:
            raise ValueError(f"table {name!r} lacks columns {miss}; verify the release before running")
    a = tables["apoeres"]
    if "APOE4" not in a.columns and "GENOTYPE" not in a.columns:
        raise ValueError("APOERES needs APOE4 or GENOTYPE")


def mri_columns(fs: pd.DataFrame, explicit: list | None = None) -> tuple[list, dict]:
    """FreeSurfer feature columns and their counts by type; `explicit` (the Table S1 list) overrides the pattern."""
    cols = list(explicit) if explicit is not None else [c for c in fs.columns if MRI_PATTERN.match(c)]
    missing = [c for c in cols if c not in fs.columns]
    if missing:
        raise ValueError(f"{len(missing)} requested MRI columns are not in UCSFFSX7 (e.g. {missing[:3]})")
    counts = {t: sum(c.endswith(t) for c in cols) for t in MRI_EXPECTED}
    return cols, counts


def _per_visit(df: pd.DataFrame, value_cols: list, flow: dict, name: str, month_col="visit_month") -> pd.DataFrame:
    """One row per RID-visit: the row with the fewest missing values among value_cols (ties: first) [C]."""
    from yoad.experiments.visit_anchor_audit.audit_lib import viscode_month
    d = df.copy()
    d[month_col] = d["VISCODE2"].map(viscode_month)
    flow[f"{name}_rows_non_protocol_visit"] = int(d[month_col].isna().sum())
    d = d.dropna(subset=[month_col]); d[month_col] = d[month_col].astype(int)
    d["RID"] = pd.to_numeric(d["RID"], errors="coerce")
    d["_miss"] = d[value_cols].isna().sum(axis=1)
    before = len(d)
    d = d.sort_values(["RID", month_col, "_miss"], kind="mergesort").drop_duplicates(["RID", month_col], keep="first")
    flow[f"{name}_duplicate_visit_rows_dropped"] = before - len(d)
    return d.drop(columns=["_miss", "VISCODE2"])


def _numeric(df, cols, flow, name):
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce")
        flow[f"{name}_{c}_negative_codes_set_missing"] = int((v < 0).sum())
        df[c] = v.where(v >= 0)
    return df


def dated_baseline_predictors(s: pd.DataFrame, tables: dict, keep, data_cutoff, flow: dict) -> pd.DataFrame:
    """COG columns for month-0 samples by the approved dated rule (decision log 121). s: labelled samples with RID, visit_month
    and EXAMDATE (DXSUM exam date of the anchor visit; undated -> no index -> no match). Returns RID, visit_month and the COG
    columns for month-0 samples; other samples get none. Reuses anchor_predictors (cleaning, window, nearest date, leakage check)."""
    from yoad.experiments.visit_anchor_audit.audit_lib import K
    from yoad.features.anchor_predictors import attach_predictors, clean_measurements, predictor_window_summary
    long = []
    for name, cols in DATED_INSTRUMENTS.items():
        t = keep(tables[name])
        if "VISDATE" not in t.columns:
            raise ValueError(f"table {name!r} lacks VISDATE, needed for predictor_timing='dated_window'")
        for c in cols:
            v = pd.to_numeric(t[c], errors="coerce")
            flow[f"dated_{name}_{c}_negative_codes_set_missing"] = int((v < 0).sum())
            long.append(pd.DataFrame({"participant_id": pd.to_numeric(t["RID"], errors="coerce"), "variable": c,
                                      "date": t["VISDATE"], "value": v.where(v >= 0)}))
    long = pd.concat(long, ignore_index=True).dropna(subset=["participant_id"])
    pool, cflow = clean_measurements(long, data_cutoff)          # undated, missing, post-cut-off, conflicting same-date: dropped
    flow.update({f"dated_{k}": v for k, v in cflow.items()})
    # one assessment = participant + date with EVERY value of the instrument valid (keeps CDR and ADAS pairs together)
    wide = pool.pivot(index=["participant_id", "date"], columns="variable", values="value")
    assess = []
    for name, cols in DATED_INSTRUMENTS.items():
        have = wide.reindex(columns=cols)
        part = have.notna().any(axis=1)
        ok = have.notna().all(axis=1)
        flow[f"dated_{name}_assessments_incomplete_excluded"] = int((part & ~ok).sum())
        flow[f"dated_{name}_assessments_valid"] = int(ok.sum())
        assess.append(have[ok].reset_index().assign(variable=name, value=1.0))
    assess = pd.concat(assess, ignore_index=True)
    m0 = s[s["visit_month"] == 0]
    a = pd.DataFrame({"participant_id": m0["RID"].to_numpy(), "anchor_month": 0,
                      "anchor_date": pd.to_datetime(m0["EXAMDATE"], errors="coerce").to_numpy()})
    flow["dated_m0_anchors"] = len(a)
    flow["dated_m0_anchors_undated_index"] = int(a["anchor_date"].isna().sum())
    flow["dated_non_baseline_samples_not_matched"] = int((s["visit_month"] != 0).sum())
    lo, hi = BASELINE_DATED_WINDOW
    # Candidates too old to be used, the mirror image of attach_predictors' candidates_after_index_time (timing proposal §4).
    # Counted on the same unit as that count -- one anchor x one valid assessment of the instrument, merged on participant_id,
    # so the two are directly comparable; an undated index gives a NaN offset and is counted in neither (it is reported
    # separately as dated_m0_anchors_undated_index).
    cand = assess[["participant_id", "variable", "date"]].merge(a[["participant_id", "anchor_date"]],
                                                                on="participant_id", how="inner")
    cand_offset = (cand["date"] - cand["anchor_date"]).dt.days
    rows, mflow = attach_predictors(a, assess[["participant_id", "variable", "date", "value"]],
                                    windows={n: (lo, hi) for n in DATED_INSTRUMENTS}, post_days=hi, data_cutoff=data_cutoff)
    for n in DATED_INSTRUMENTS:
        flow[f"dated_{n}_candidates_after_index"] = mflow[f"{n}_candidates_after_index_time"]
        flow[f"dated_{n}_candidates_before_window"] = int(((cand["variable"] == n) & (cand_offset < lo)).sum())
        flow[f"dated_{n}_missing_at_anchor"] = mflow[f"{n}_missing_at_anchor"]
    # How far before the index date the used assessment actually sits: the interpretive fact about this timing rule (decision
    # log 121). Offsets are signed day counts, not cell counts, so sup() must not touch them: they go to flow as a descriptive
    # string, and are withheld entirely while fewer than K anchors are matched.
    for r in predictor_window_summary(rows, {n: (lo, hi) for n in DATED_INSTRUMENTS}).itertuples():
        flow[f"dated_{r.variable}_matched"] = int(r.n_available)
        flow[f"dated_{r.variable}_offset_days_min_median_max"] = (
            f"{r.offset_min:.0f} / {r.offset_median:.1f} / {r.offset_max:.0f}" if r.n_available >= K else f"suppressed (n<{K})")
    out = rows[["participant_id", "anchor_month"]].rename(columns={"participant_id": "RID", "anchor_month": "visit_month"})
    for name, cols in DATED_INSTRUMENTS.items():               # values of the chosen assessment, joined by its date
        pick = rows[["participant_id", f"{name}_date"]].rename(columns={f"{name}_date": "date"})
        vals = pick.merge(assess.loc[assess["variable"] == name, ["participant_id", "date", *cols]],
                          on=["participant_id", "date"], how="left")
        for c in cols:
            out[c] = vals[c].to_numpy()
    return out


def build_dataset(tables: dict, mri_explicit: list | None = None, exclude_advisory: bool = True, label_rule: str = "SIM",
                  window_days: int = 90, data_cutoff=None, include_mri: bool = True,
                  complete_case_columns: list | None = None, event_grace_days: int = 0,
                  death_dates: dict | None = None, predictor_timing: str = "same_visit") -> tuple[pd.DataFrame, dict, dict]:
    """label_rule "SIM" (default; the reference rule) or "REF" (calendar three-state rule with +/- window_days around H, used for
    the NACC-transportable model, because NACC's annual visits make the visit-code rule meaningless; needs data_cutoff).
    complete_case_columns: the columns the complete-case step uses (default: all features, as in the reference [A][C]); the
    transportable model passes only its own predictors so that samples are not dropped for ADAS or MRI it never uses.
    death_dates ({RID: date}, e.g. from audit_lib.adsl_death_dates) are used ONLY by the calendar rule, where death on/before H
    without earlier dementia is a competing event (INDETERMINATE, reason competing_death; protocol §3). The visit-code reference
    rule ignores them, as the published code does.
    include_mri=False builds the clinical-only table (no UCSFFSX7 needed). Returns (samples, flow, info). samples: one row per labelled complete-case MCI visit with participant_id,
    anchor_month, exam_date, y (1 = dementia by m+24 under the reference rule), label_reason and all features.
    predictor_timing: "same_visit" (default, the reference join) or "dated_window" (decision log 121, baseline model only): the
    COG columns of month-0 samples come from dated_baseline_predictors and later samples get none. Later samples are therefore
    removed at the complete-case step only when the complete-case columns include a COG column (e.g. COG, COG+DEM); otherwise
    (e.g. DEM only) they remain, and the runner's required --anchoring baseline removes them. Needs include_mri=False and data_cutoff. Anchors, labels, demographics, AGE and the
    complete-case rule are the same in both modes."""
    from yoad.experiments.today_audit.audit_core import advisory_exclusion_mask
    from yoad.experiments.visit_anchor_audit.audit_lib import harmonise_dxsum
    if label_rule not in ("SIM", "REF"):
        raise ValueError("label_rule must be 'SIM' or 'REF'")
    if predictor_timing not in ("same_visit", "dated_window"):
        raise ValueError("predictor_timing must be 'same_visit' or 'dated_window'")
    if predictor_timing == "dated_window" and (include_mri or data_cutoff is None):
        raise ValueError("predictor_timing='dated_window' needs include_mri=False (no MRI timing rule approved) and data_cutoff")
    if label_rule == "REF" and data_cutoff is None:
        raise ValueError("label_rule='REF' needs data_cutoff (administrative censoring)")
    if not include_mri:
        tables = {**tables, "ucsffsx7": pd.DataFrame(columns=SOURCE_COLUMNS["ucsffsx7"])}
    _check(tables)
    # The dated rule needs the assessment date of EVERY instrument; SOURCE_COLUMNS requires VISDATE for MMSE only, so the
    # other three are checked here, all at once, before any cohort work (runbook step 2c, "Extra column for the dated mode").
    if predictor_timing == "dated_window":
        no_date = [n for n in DATED_INSTRUMENTS if "VISDATE" not in tables[n].columns]
        if no_date:
            raise ValueError(f"predictor_timing='dated_window' needs a VISDATE column in {no_date}; SOURCE_COLUMNS requires "
                             "VISDATE for 'mmse' only, so verify these columns in your release before running")
    flow: dict = {}
    dx = tables["dxsum"].copy()
    excl = advisory_exclusion_mask(dx["PTID"]) if exclude_advisory else pd.Series(False, index=dx.index)
    bad = set(pd.to_numeric(dx.loc[excl, "RID"], errors="coerce"))
    flow["advisory_participants_excluded"] = len(bad)
    dx = dx.loc[~excl].copy()
    # The label rule uses visit codes only [C], so undated DXSUM rows still count as outcome visits: they get a placeholder
    # date for harmonisation only. The real exam date (DXSUM, else MMSE VISDATE [C]) is attached below for AGE.
    flow["dxsum_rows_undated_kept_for_labels"] = int(pd.to_datetime(dx["EXAMDATE"], errors="coerce").isna().sum())
    dx["_label_date"] = pd.to_datetime(dx["EXAMDATE"], errors="coerce").fillna(pd.Timestamp("1900-01-01"))
    visits, hflow = harmonise_dxsum(dx.drop(columns=["PTID"]), date_col="_label_date")
    flow.update({f"dxsum_{k}": v for k, v in hflow.items()})
    visits["participant_id"] = pd.to_numeric(visits["participant_id"], errors="coerce")
    if label_rule == "REF":                       # calendar rule needs real dates: undated rows cannot be outcome visits
        undated = visits["date"] == pd.Timestamp("1900-01-01")
        flow["dxsum_rows_undated_dropped_for_REF"] = int(undated.sum())
        visits = visits[~undated]
    a = anchors(visits, "all_mci_visits")
    flow["m0_mci_anchors_before_labelling"] = int((a["anchor_month"] == 0).sum())   # month-0 diagnostics: aggregate only
    if label_rule == "REF":
        a, flow["mci_anchors_after_recorded_dementia_dropped"] = drop_post_dementia_anchors(visits, a)
    lab = (label_visitcode_sim2026(visits, a) if label_rule == "SIM"
           else label_calendar_reference(visits, a, w_days=window_days, u_days=window_days, data_cutoff=data_cutoff,
                                         event_grace_days=event_grace_days, death_dates=death_dates))
    flow["death_dates_supplied"] = int(len(death_dates or {}))
    flow["death_rule"] = ("competing (calendar rule)" if label_rule == "REF" and death_dates else
                          "not applied: visit-code reference rule ignores death" if label_rule == "SIM" else
                          "not applied: no death dates supplied")
    flow.update(state_summary(lab, "labels_"))
    flow["mci_visits"] = len(lab)
    for (s, r), n in lab.groupby(["state", "reason"]).size().items():
        flow[f"label_{s}_{r}"] = int(n)
    m0 = lab[lab["anchor_month"] == 0]
    flow["m0_anchors_labelled"] = len(m0)
    flow.update(state_summary(m0, "m0_labels_"))
    for (s, r), n in m0.groupby(["state", "reason"]).size().items():
        flow[f"m0_label_{s}_{r}"] = int(n)
    lab = lab[lab["state"].isin([EVENT, NONEVENT])].copy()
    lab["y"] = (lab["state"] == EVENT).astype(int)
    s = lab.rename(columns={"participant_id": "RID", "anchor_month": "visit_month", "reason": "label_reason"})
    s = s[["RID", "visit_month", "y", "label_reason"]]
    dd = _per_visit(dx[["RID", "VISCODE2", "EXAMDATE"]], ["EXAMDATE"], flow, "dxsum_dates")
    s = s.merge(dd, on=["RID", "visit_month"], how="left")

    keep = lambda d: d[~pd.to_numeric(d["RID"], errors="coerce").isin(bad)]
    mm = _numeric(_per_visit(keep(tables["mmse"][SOURCE_COLUMNS["mmse"]]), ["MMSCORE"], flow, "mmse"), ["MMSCORE"], flow, "mmse")
    cdr = _numeric(_per_visit(keep(tables["cdr"][SOURCE_COLUMNS["cdr"]]), ["CDGLOBAL", "CDRSB"], flow, "cdr"), ["CDGLOBAL", "CDRSB"], flow, "cdr")
    adas = _numeric(_per_visit(keep(tables["adas"][SOURCE_COLUMNS["adas"]]), ["TOTSCORE", "TOTAL13"], flow, "adas"), ["TOTSCORE", "TOTAL13"], flow, "adas")
    faq = _numeric(_per_visit(keep(tables["faq"][SOURCE_COLUMNS["faq"]]), ["FAQTOTAL"], flow, "faq"), ["FAQTOTAL"], flow, "faq")
    fs_all = keep(tables["ucsffsx7"])
    mcols, mcounts = mri_columns(fs_all, mri_explicit) if include_mri else ([], {t: 0 for t in MRI_EXPECTED})
    fs = _per_visit(fs_all[["RID", "VISCODE2"] + mcols], mcols, flow, "ucsffsx7")
    for c in mcols:
        fs[c] = pd.to_numeric(fs[c], errors="coerce")
    flow["predictor_timing"] = predictor_timing
    if predictor_timing == "dated_window":
        dated = dated_baseline_predictors(s, tables, keep, data_cutoff, flow)
        mm, cdr, adas, faq = mm[["RID", "visit_month", "VISDATE"]], dated, adas[["RID", "visit_month"]], faq[["RID", "visit_month"]]
    for t in (mm, cdr, adas, faq, fs):
        s = s.merge(t, on=["RID", "visit_month"], how="left")
    # exam date: DXSUM date, else the MMSE visit date [C]
    s["exam_date"] = pd.to_datetime(s.pop("EXAMDATE"), errors="coerce").fillna(pd.to_datetime(s.pop("VISDATE"), errors="coerce"))

    ap = keep(tables["apoeres"]).copy(); ap["RID"] = pd.to_numeric(ap["RID"], errors="coerce")
    if "APOE4" in ap.columns:
        ap["APOE4"] = pd.to_numeric(ap["APOE4"], errors="coerce")
    else:                                               # [D] count of e4 alleles from GENOTYPE (e.g. "3/4" -> 1)
        g = ap["GENOTYPE"].astype(str)
        ap["APOE4"] = g.str.count("4").clip(upper=2).astype(float).where(ap["GENOTYPE"].notna())
    s = s.merge(ap.dropna(subset=["RID"]).drop_duplicates("RID")[["RID", "APOE4"]], on="RID", how="left")
    dm = keep(tables["ptdemog"][SOURCE_COLUMNS["ptdemog"]]).copy(); dm["RID"] = pd.to_numeric(dm["RID"], errors="coerce")
    dm = dm.dropna(subset=["RID"]).drop_duplicates("RID")       # first row per participant [C]
    s = s.merge(dm, on="RID", how="left")
    dob = pd.to_datetime(s.pop("PTDOB"), errors="coerce")
    flow["ptdob_unparseable"] = int(dob.isna().sum())
    s["AGE"] = (s["exam_date"] - dob).dt.days / 365.25
    for c in ("PTGENDER", "PTHAND"):
        s[c] = pd.to_numeric(s[c], errors="coerce") - 1                       # 1/2 -> 0/1 [C]
        s.loc[~s[c].isin([0, 1]), c] = np.nan
    s["PTEDUCAT"] = pd.to_numeric(s["PTEDUCAT"], errors="coerce")
    s.loc[s["PTEDUCAT"] < 0, "PTEDUCAT"] = np.nan

    features = COG + DEM + mcols
    flow["labelled_samples_before_complete_case"] = len(s)
    for c in COG + DEM:
        flow[f"missing_{c}"] = int(s[c].isna().sum())
    flow["missing_any_mri"] = int(s[mcols].isna().any(axis=1).sum())
    cc = features if complete_case_columns is None else list(complete_case_columns)
    s0 = s[s["visit_month"] == 0]                          # month-0 determinate labels, before the complete-case step
    flow["m0_determinate_before_complete_case"] = len(s0)
    flow["m0_determinate_events_before_complete_case"] = int(s0["y"].sum())
    for c in [c for c in cc if c not in mcols]:
        flow[f"m0_missing_{c}"] = int(s0[c].isna().sum())
    mri_cc = [c for c in cc if c in mcols]
    if mri_cc:                                             # any = at least one MRI column missing; all = no MRI at that visit
        flow["m0_missing_any_mri"] = int(s0[mri_cc].isna().any(axis=1).sum())
        flow["m0_missing_all_mri"] = int(s0[mri_cc].isna().all(axis=1).sum())
    s = s.dropna(subset=cc).copy()   # [A][C]
    flow["m0_determinate_complete_case"] = int((s["visit_month"] == 0).sum())
    flow["m0_events_complete_case"] = int(s.loc[s["visit_month"] == 0, "y"].sum())
    flow["samples_complete_case"] = len(s)
    flow["participants_complete_case"] = int(s["RID"].nunique())
    flow["events_complete_case"] = int(s["y"].sum())
    s = s.rename(columns={"RID": "participant_id", "visit_month": "anchor_month"}).reset_index(drop=True)
    info = {"features": {"COG": COG, "DEM": DEM, "MRI": mcols}, "categorical": CATEGORICAL, "mri_counts": mcounts,
            "death_rule": flow["death_rule"], "predictor_timing": predictor_timing,
            "label_rule": label_rule, "window_days": window_days if label_rule == "REF" else None,
            "mri_counts_match_paper": mcounts == MRI_EXPECTED, "mri_selection": "explicit list" if mri_explicit else "pattern"}
    return s, flow, info


class EmptyCohortError(ValueError):
    """The selected analysis cohort has no samples; raised before any split or model fit."""


M0_STEPS = ("m0_mci_anchors_before_labelling", "m0_anchors_labelled", "m0_determinate_before_complete_case",
            "m0_determinate_complete_case")


def restrict_to_baseline(samples: pd.DataFrame, flow: dict) -> pd.DataFrame:
    """--anchoring baseline: keep the month-0 MCI visit only (anchor definition unchanged). Fails fast, naming the first month-0
    flow step that reached zero, instead of failing later inside the scikit-learn split."""
    flow["samples_before_baseline_restriction"] = len(samples)
    out = samples[samples["anchor_month"] == 0].reset_index(drop=True)
    flow["samples_baseline_only"] = len(out)
    if out.empty:
        first = next((k for k in M0_STEPS if flow.get(k) == 0), "m0_determinate_complete_case")
        raise EmptyCohortError(f"baseline cohort is empty: no month-0 samples; first zero step in flow.csv: {first}. "
                               "See the m0_* rows (labels by state/reason, m0_missing_<predictor>) to locate the loss.")
    return out


def feature_columns(info: dict, feature_set: str) -> tuple[list, list]:
    """(numeric, categorical) columns of one of the 7 feature sets; categorical columns go last (SMOTENC indices)."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {list(FEATURE_SETS)}")
    cols = [c for m in FEATURE_SETS[feature_set] for c in info["features"][m]]
    cat = [c for c in cols if c in info["categorical"]]
    return [c for c in cols if c not in cat], cat
