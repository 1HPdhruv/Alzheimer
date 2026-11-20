"""Cohort validation and outcome derivation. Missing/insufficient follow-up is never coded as negative."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
DAYS_PER_YEAR = 365.25

# Dataset-specific mapping must be supplied via config; this default is only for harmonised labels.
VALID_LABELS = {"CN", "MCI", "AD", "OTHER_DEM"}


class ExclusionError(RuntimeError):
    pass


def normalize_diagnosis(s: pd.Series, code_map: dict) -> pd.Series:
    """Map raw codes to {CN,MCI,AD,OTHER_DEM}. Unmapped values become NA (counted, warned), never guessed."""
    out = s.map(code_map)
    unmapped = s[out.isna() & s.notna()]
    if len(unmapped):
        log.warning("%d rows with unmapped diagnosis codes: %s", len(unmapped), sorted(unmapped.unique())[:10])
    bad = set(out.dropna().unique()) - VALID_LABELS
    if bad:
        raise ValueError(f"code_map produced invalid labels {bad}")
    return out


def validate_transitions(long_df: pd.DataFrame) -> dict:
    """Diagnosis-sequence checks on canonical long data (participant_id, visit_date, diagnosis[normalised])."""
    df = long_df.copy()
    res = {"n_rows": len(df), "n_participants": df["participant_id"].nunique()}
    res["missing_date_rows"] = int(df["visit_date"].isna().sum())
    res["missing_dx_rows"] = int(df["diagnosis"].isna().sum())
    d = df.dropna(subset=["visit_date"])
    dup = d.groupby(["participant_id", "visit_date"])["diagnosis"].nunique(dropna=True)
    res["conflicting_same_day_dx"] = int((dup > 1).sum())
    res["exact_duplicate_rows"] = int(d.duplicated(["participant_id", "visit_date", "diagnosis"]).sum())
    rev, patterns = [], {}
    for pid, g in d.sort_values("visit_date").groupby("participant_id"):
        seq = [x for x in g["diagnosis"].tolist() if pd.notna(x)]
        collapsed = [x for i, x in enumerate(seq) if i == 0 or x != seq[i - 1]]
        key = ">".join(collapsed)
        patterns[key] = patterns.get(key, 0) + 1
        seen_dem = False
        for x in collapsed:
            if x in ("AD", "OTHER_DEM"):
                seen_dem = True
            elif seen_dem:  # dementia followed by non-dementia
                rev.append(pid)
                break
    res["reversion_participants"] = sorted(set(rev))
    res["n_reversions"] = len(set(rev))
    res["transition_patterns"] = dict(sorted(patterns.items(), key=lambda kv: -kv[1]))
    return res


def define_baseline(long_df: pd.DataFrame, baseline_dx: str = "MCI") -> pd.DataFrame:
    """Baseline = earliest dated visit with the baseline diagnosis. One row per participant; no silent drops."""
    d = long_df.dropna(subset=["visit_date"])
    d = d[d["diagnosis"] == baseline_dx].sort_values(["participant_id", "visit_date"])
    base = d.groupby("participant_id", as_index=False).first()[["participant_id", "visit_date"]]
    return base.rename(columns={"visit_date": "baseline_date"})


def check_baseline_consistency(base: pd.DataFrame, long_df: pd.DataFrame, declared: pd.DataFrame | None = None) -> dict:
    """Baseline has dx==MCI, nothing used before baseline as 'follow-up', and (if given) matches a declared baseline table."""
    m = long_df.merge(base, on="participant_id", how="inner")
    res = {"n_baseline": len(base)}
    first_dx = (m[m["visit_date"] == m["baseline_date"]].groupby("participant_id")["diagnosis"].agg(lambda s: set(s)))
    res["baseline_visit_not_mci"] = int(sum("MCI" not in v for v in first_dx))
    res["participants_with_prebaseline_visits"] = int(m[m["visit_date"] < m["baseline_date"]]["participant_id"].nunique())
    if declared is not None:
        j = base.merge(declared[["participant_id", "baseline_date"]], on="participant_id", how="outer",
                       suffixes=("", "_declared"), indicator=True)
        res["missing_in_declared"] = int((j["_merge"] == "left_only").sum())
        res["missing_in_derived"] = int((j["_merge"] == "right_only").sum())
        both = j[j["_merge"] == "both"]
        res["baseline_date_mismatch"] = int((both["baseline_date"] != both["baseline_date_declared"]).sum())
    return res


def derive_outcomes(long_df: pd.DataFrame, base: pd.DataFrame, horizon_years: float | None = None,
                    min_followup_years: float = 0.0) -> pd.DataFrame:
    """Per-participant outcome table.

    status: AD_CONVERTER | OTHER_DEM_CONVERTER | STABLE_OBSERVED | CENSORED_INSUFFICIENT | NO_FOLLOWUP
    label_ad (fixed horizon): 1 = AD within horizon; 0 = observed without AD for >= horizon (or AD after horizon);
    NaN = censored before horizon, competing non-AD dementia within horizon, or no follow-up.
    Time-to-event columns are always provided so survival analysis can use censored participants.
    """
    d = long_df.dropna(subset=["visit_date", "diagnosis"]).merge(base, on="participant_id")
    d = d[d["visit_date"] > d["baseline_date"]].sort_values(["participant_id", "visit_date"])
    rows = []
    last = d.groupby("participant_id")["visit_date"].max()
    for pid, bdate in base.set_index("participant_id")["baseline_date"].items():
        g = d[d["participant_id"] == pid]
        rec = {"participant_id": pid, "baseline_date": bdate, "n_followups": len(g)}
        if g.empty:
            rec.update(status="NO_FOLLOWUP", event_time_years=np.nan, followup_years=0.0)
        else:
            fu = (last[pid] - bdate).days / DAYS_PER_YEAR
            ev = g[g["diagnosis"].isin(["AD", "OTHER_DEM"])]
            if len(ev):
                first = ev.iloc[0]
                t = (first["visit_date"] - bdate).days / DAYS_PER_YEAR
                rec.update(status="AD_CONVERTER" if first["diagnosis"] == "AD" else "OTHER_DEM_CONVERTER",
                           event_time_years=t, followup_years=fu)
            else:
                rec.update(status="STABLE_OBSERVED" if fu >= min_followup_years else "CENSORED_INSUFFICIENT",
                           event_time_years=np.nan, followup_years=fu)
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["event_ad"] = (out["status"] == "AD_CONVERTER").astype(int)
    out["censored_for_ad"] = (out["status"] != "AD_CONVERTER").astype(int)  # competing events are censored for cause-specific AD
    if horizon_years is not None:
        out[f"label_ad_{horizon_years:g}y"] = derive_outcomes_label(out, horizon_years)
    return out


def exclusion_report(n_before: int, n_after: int, reason: str, max_fraction: float = 0.10,
                     allow_large: bool = False) -> dict:
    """Every exclusion step must be recorded; large silent exclusions raise."""
    frac = (n_before - n_after) / n_before if n_before else 0.0
    rec = {"reason": reason, "n_before": n_before, "n_after": n_after, "excluded": n_before - n_after, "fraction": round(frac, 4)}
    if frac > max_fraction and not allow_large:
        raise ExclusionError(f"{reason}: would exclude {frac:.1%} (> {max_fraction:.0%}); pass allow_large=True to confirm")
    if frac > 0:
        log.warning("exclusion %s", rec)
    return rec


def compare_followup_windows(outcomes: pd.DataFrame, windows=(1, 2, 3, 5)) -> pd.DataFrame:
    """Evaluable N / events / censored-unknown for candidate horizons, to justify the minimum-follow-up choice."""
    rows = []
    for w in windows:
        lab = derive_outcomes_label(outcomes, w)
        rows.append({"horizon_years": w, "n_total": len(outcomes), "n_labelled": int(lab.notna().sum()),
                     "n_events": int((lab == 1).sum()), "n_nonevents": int((lab == 0).sum()),
                     "n_unlabelled": int(lab.isna().sum())})
    return pd.DataFrame(rows)


def derive_outcomes_label(outcomes: pd.DataFrame, h: float) -> pd.Series:
    t, fu, st = outcomes["event_time_years"], outcomes["followup_years"], outcomes["status"]
    lab = pd.Series(np.nan, index=outcomes.index)
    lab[(st == "AD_CONVERTER") & (t <= h)] = 1
    lab[(st == "AD_CONVERTER") & (t > h)] = 0
    lab[st.isin(["STABLE_OBSERVED", "CENSORED_INSUFFICIENT"]) & (fu >= h)] = 0
    return lab


def classify_age_eligibility(df: pd.DataFrame, onset_age_col: str | None, enrol_age_col: str | None,
                             cutoff: float = 65.0) -> pd.DataFrame:
    """Never labels a participant verified YOAD from enrollment age alone."""
    out = df.copy()
    cls = pd.Series("UNKNOWN_AGE", index=out.index, dtype=object)
    if enrol_age_col and enrol_age_col in out:
        a = pd.to_numeric(out[enrol_age_col], errors="coerce")
        cls[a < cutoff] = "APPROX_ENROLLMENT_AGE_LT_CUTOFF"
        cls[a >= cutoff] = "ENROLLMENT_AGE_GE_CUTOFF"
    if onset_age_col and onset_age_col in out:
        o = pd.to_numeric(out[onset_age_col], errors="coerce")
        cls[o < cutoff] = "ONSET_VERIFIED_LT_CUTOFF"
        cls[o >= cutoff] = "ONSET_GE_CUTOFF"
    out["age_class"] = cls
    if onset_age_col and enrol_age_col and onset_age_col in out and enrol_age_col in out:
        o = pd.to_numeric(out[onset_age_col], errors="coerce"); e = pd.to_numeric(out[enrol_age_col], errors="coerce")
        out["onset_after_enrollment_flag"] = (o > e)  # implausible: onset after enrollment
    return out


def resolve_onset_age(visits: pd.DataFrame, onset_col: str, policy: str = "first_visit", tol_years: float = 2.0) -> pd.DataFrame:
    """Collapse a visit-level onset-age field (e.g. NACC DECAGE, which can change across visits because different
    clinicians assess participants) to one value per participant, with an instability flag.

    visits: participant_id, visit_date, <onset_col>. policy: 'first_visit' (first non-missing by date) or 'latest'.
    Returns participant_id, onset_age, onset_n_distinct, onset_range, onset_unstable (range > tol_years), onset_missing.
    Participants with no non-missing value are kept with onset_missing=True (never dropped).
    """
    if policy not in ("first_visit", "latest"):
        raise ValueError(f"unknown policy {policy!r}")
    v = visits.copy()
    v[onset_col] = pd.to_numeric(v[onset_col], errors="coerce")
    v = v.sort_values(["participant_id", "visit_date"])
    rows = []
    for pid, g in v.groupby("participant_id", sort=True):
        x = g[onset_col].dropna()
        if x.empty:
            rows.append({"participant_id": pid, "onset_age": np.nan, "onset_n_distinct": 0, "onset_range": np.nan,
                         "onset_unstable": False, "onset_missing": True})
        else:
            rng = float(x.max() - x.min())
            rows.append({"participant_id": pid, "onset_age": float(x.iloc[0] if policy == "first_visit" else x.iloc[-1]),
                         "onset_n_distinct": int(x.nunique()), "onset_range": rng, "onset_unstable": rng > tol_years,
                         "onset_missing": False})
    return pd.DataFrame(rows)


def onset_vs_age_definition(df: pd.DataFrame, onset_col: str, baseline_age_col: str, cutoff: float = 65.0, band: float = 3.0) -> dict:
    """Synthetic-testable summary for the secondary question (onset-defined vs baseline-age-defined younger group).

    If participants are symptomatic at baseline, onset <= baseline age, so baseline age < cutoff implies onset < cutoff;
    the proxy can therefore MISS onset<cutoff cases (baseline age >= cutoff) and contaminate the comparison group, but not add
    onset>=cutoff cases. Rows with onset > baseline age are impossible and are flagged, not silently fixed. Missing onset is
    reported separately and never replaced by baseline age.
    """
    o = pd.to_numeric(df[onset_col], errors="coerce"); a = pd.to_numeric(df[baseline_age_col], errors="coerce")
    ok = o.notna() & a.notna()
    impossible = ok & (o > a)
    use = ok & ~impossible
    on, ag = o[use] < cutoff, a[use] < cutoff
    return {"n_total": int(len(df)), "n_missing_onset": int(o.isna().sum()), "n_missing_age": int(a.isna().sum()),
            "n_onset_after_baseline_flagged": int(impossible.sum()), "n_analysable": int(use.sum()),
            "onset_lt_cutoff": int(on.sum()), "age_lt_cutoff": int(ag.sum()),
            "onset_lt_but_age_ge (missed by proxy)": int((on & ~ag).sum()),
            "age_lt_but_onset_ge (must be 0 if consistent)": int((~on & ag).sum()),
            "proxy_sensitivity_for_onset_lt": float((on & ag).sum() / on.sum()) if on.sum() else float("nan"),
            "n_within_band_of_cutoff": int(((o[use] - cutoff).abs() <= band).sum())}
