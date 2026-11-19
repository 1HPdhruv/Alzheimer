"""Visit-anchored analysis rows for the 2 x 2 design (ANALYSIS_PROTOCOL.md §2, §5; OUTCOME_DEFINITION_CODEBOOK.md).

Definitions
  analysis population  participants whose protocol-baseline visit (visit_month 0) is an ELIGIBLE MCI anchor; participants without
                       one are excluded entirely (their later visits are not used either)
  anchor               an MCI diagnosis visit of a participant in the analysis population
  anchor date          the diagnosis exam date of that visit
  index time           anchor date + post_days (default 14 d, the protocol's assessment window for the same visit). No predictor
                       measurement dated after the index time is ever used; `check_no_post_index` enforces this at run time.
  horizon H            anchor date + 24 calendar months (pandas DateOffset, as in src/cohort/reference_outcome.py)
  eligible anchor      MCI recorded at the visit; no dementia recorded on or before the anchor date (so a later MCI record after
                       a dementia diagnosis - a reversion - is never an anchor); H + 90 d on or before the data cut-off;
                       anchor date before death when a death date is known
Input tables (pure functions; no file access)
  visits        participant_id, visit_month (int protocol month), date, dx in {CN, MCI, DEM}; one diagnosis per participant-visit
                (experiments/visit_anchor_audit/audit_lib.harmonise_dxsum produces this: identical duplicates collapse, a visit with
                conflicting diagnoses is dropped and counted)
  measurements  long format: participant_id, variable, date, value
  static        participant_id, baseline_age, female, education_years, apoe4 (time-invariant; age is advanced to each anchor
                by the time since the participant's baseline anchor)
Measurement handling (each step counted in the returned flow)
  undated or non-numeric value -> dropped; exact duplicates collapse; one participant/variable/date with different values ->
  that date is dropped (conflict); dated after the data cut-off -> dropped (future-dated); per anchor and variable, the
  measurement inside [anchor date + lo, index time] closest to the anchor date is used, ties -> the earlier one; none -> missing
  (imputed later inside the training pipeline only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON_MONTHS = 24
DEFAULT_WINDOWS = {"mmse": (-90, 14), "adas13": (-90, 14), "faq": (-90, 14)}   # cognition and function (protocol §5)
STATIC_COLUMNS = ("baseline_age", "female", "education_years", "apoe4")


def eligible_anchors(visits: pd.DataFrame, data_cutoff, horizon_months: int = HORIZON_MONTHS, u_days: int = 90,
                     death_dates: dict | None = None, baseline_month: int = 0) -> tuple[pd.DataFrame, dict]:
    """Eligible MCI anchors of the analysis population. Returns (anchors, flow) with columns participant_id, anchor_month,
    anchor_date, horizon_date, is_baseline."""
    need = {"participant_id", "visit_month", "date", "dx"}
    if need - set(visits.columns):
        raise ValueError(f"visits missing columns {sorted(need - set(visits.columns))}")
    v = visits.copy()
    v["date"] = pd.to_datetime(v["date"], errors="coerce")
    if v.duplicated(["participant_id", "visit_month"]).any():
        raise ValueError("visits must have one diagnosis per participant-visit; harmonise first")
    cut = pd.Timestamp(data_cutoff)
    flow = {"participants_in": int(v["participant_id"].nunique()), "mci_visits_in": int((v["dx"] == "MCI").sum())}
    first_dem = v[v["dx"] == "DEM"].groupby("participant_id")["date"].min()
    a = v[(v["dx"] == "MCI") & v["date"].notna()].copy()
    a["first_dem"] = a["participant_id"].map(first_dem)
    after_dem = a["first_dem"].notna() & (a["date"] >= a["first_dem"])
    flow["mci_visits_on_or_after_recorded_dementia"] = int(after_dem.sum())
    a = a[~after_dem]
    a["horizon_date"] = [d + pd.DateOffset(months=horizon_months) for d in a["date"]]
    open_win = a["horizon_date"] + pd.Timedelta(days=u_days) > cut
    flow["mci_visits_window_not_closed"] = int(open_win.sum())
    a = a[~open_win]
    if death_dates:
        dd = pd.to_datetime(a["participant_id"].map(death_dates))
        dead = dd.notna() & (a["date"] >= dd)
        flow["mci_visits_on_or_after_death"] = int(dead.sum())
        a = a[~dead]
    a["is_baseline"] = a["visit_month"] == baseline_month
    pop = set(a.loc[a["is_baseline"], "participant_id"])
    flow["mci_visits_of_participants_without_eligible_baseline"] = int((~a["participant_id"].isin(pop)).sum())
    a = a[a["participant_id"].isin(pop)]
    flow["participants_analysis_population"] = len(pop)
    flow["eligible_anchors"] = int(len(a))
    out = a.rename(columns={"visit_month": "anchor_month", "date": "anchor_date"})
    out = out[["participant_id", "anchor_month", "anchor_date", "horizon_date", "is_baseline"]]
    return out.sort_values(["participant_id", "anchor_month"]).reset_index(drop=True), flow


def clean_measurements(measurements: pd.DataFrame, data_cutoff) -> tuple[pd.DataFrame, dict]:
    need = {"participant_id", "variable", "date", "value"}
    if need - set(measurements.columns):
        raise ValueError(f"measurements missing columns {sorted(need - set(measurements.columns))}")
    m = measurements[list(need)].copy()
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m["value"] = pd.to_numeric(m["value"], errors="coerce")
    flow = {"measurements_in": int(len(m)), "measurements_undated": int(m["date"].isna().sum()),
            "measurements_non_numeric_or_missing": int((m["date"].notna() & m["value"].isna()).sum())}
    m = m.dropna(subset=["date", "value"])
    future = m["date"] > pd.Timestamp(data_cutoff)
    flow["measurements_after_data_cutoff"] = int(future.sum())
    m = m[~future].drop_duplicates()
    clash = m.duplicated(["participant_id", "variable", "date"], keep=False)
    flow["measurements_conflicting_same_date"] = int(clash.sum())
    m = m[~clash]
    flow["measurements_used_pool"] = int(len(m))
    return m.reset_index(drop=True), flow


def attach_predictors(anchors: pd.DataFrame, measurements: pd.DataFrame, static: pd.DataFrame | None = None,
                      windows: dict | None = None, post_days: int = 14, data_cutoff=None) -> tuple[pd.DataFrame, dict]:
    """One row per anchor with each windowed predictor, its measurement date and offset (days from the anchor date), plus
    static predictors and age at the anchor. Raises if a window ends after the index time."""
    windows = dict(DEFAULT_WINDOWS if windows is None else windows)
    for var, (lo, hi) in windows.items():
        if hi > post_days or lo > hi:
            raise ValueError(f"window for {var} must satisfy lo <= hi <= post_days ({post_days}); got ({lo}, {hi})")
    if data_cutoff is None:
        raise ValueError("data_cutoff is required (future-dated measurements are removed against it)")
    m, flow = clean_measurements(measurements, data_cutoff)
    rows = anchors.copy()
    rows["index_time"] = rows["anchor_date"] + pd.Timedelta(days=post_days)
    key = rows[["participant_id", "anchor_month", "anchor_date"]]
    for var, (lo, hi) in windows.items():
        mv = m[m["variable"] == var][["participant_id", "date", "value"]]
        j = key.merge(mv, on="participant_id", how="inner")
        j["off"] = (j["date"] - j["anchor_date"]).dt.days
        flow[f"{var}_candidates_after_index_time"] = int((j["off"] > hi).sum())
        j = j[(j["off"] >= lo) & (j["off"] <= hi)]
        j["absoff"] = j["off"].abs()
        j = j.sort_values(["participant_id", "anchor_month", "absoff", "off"]).groupby(["participant_id", "anchor_month"]).head(1)
        j = j.rename(columns={"value": var, "date": f"{var}_date", "off": f"{var}_offset_days"})
        rows = rows.merge(j[["participant_id", "anchor_month", var, f"{var}_date", f"{var}_offset_days"]],
                          on=["participant_id", "anchor_month"], how="left")
        flow[f"{var}_missing_at_anchor"] = int(rows[var].isna().sum())
    if static is not None:
        miss = set(STATIC_COLUMNS) - set(static.columns)
        if miss:
            raise ValueError(f"static table missing columns {sorted(miss)}")
        if static["participant_id"].duplicated().any():
            raise ValueError("static table must have one row per participant")
        rows = rows.merge(static[["participant_id", *STATIC_COLUMNS]], on="participant_id", how="left")
        base = rows.loc[rows["is_baseline"]].set_index("participant_id")["anchor_date"]
        rows["age"] = rows["baseline_age"] + (rows["anchor_date"] - rows["participant_id"].map(base)).dt.days / 365.25
        rows = rows.drop(columns=["baseline_age"])
    check_no_post_index(rows, windows, post_days)
    return rows, flow


def check_no_post_index(rows: pd.DataFrame, windows: dict, post_days: int) -> None:
    """Run-time leakage check: every used measurement date lies in [anchor date + lo, anchor date + post_days]."""
    for var, (lo, _) in windows.items():
        d = rows[f"{var}_date"]
        used = d.notna()
        late = used & (d > rows["anchor_date"] + pd.Timedelta(days=post_days))
        early = used & (d < rows["anchor_date"] + pd.Timedelta(days=lo))
        if late.any() or early.any():
            raise AssertionError(f"{var}: {int(late.sum())} measurement(s) after the index time, {int(early.sum())} before the window")


def predictor_window_summary(rows: pd.DataFrame, windows: dict) -> pd.DataFrame:
    """Aggregate per-predictor availability and offsets (no participant-level content)."""
    out = []
    for var in windows:
        off = rows[f"{var}_offset_days"]
        out.append({"variable": var, "n_anchors": int(len(rows)), "n_available": int(off.notna().sum()),
                    "offset_min": float(off.min()) if off.notna().any() else np.nan,
                    "offset_median": float(off.median()) if off.notna().any() else np.nan,
                    "offset_max": float(off.max()) if off.notna().any() else np.nan})
    return pd.DataFrame(out)
