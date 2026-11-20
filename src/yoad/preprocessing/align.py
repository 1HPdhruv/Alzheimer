"""Visit alignment, missingness, counts, modality overlap."""
from __future__ import annotations

import pandas as pd


def align_nearest(base: pd.DataFrame, other: pd.DataFrame, window_days: int, date_col: str = "visit_date") -> pd.DataFrame:
    """For each participant, nearest `other` record to baseline_date within +/-window_days (one match per participant).
    Returns all baseline participants with matched flag and signed delta (other - baseline)."""
    o = other[["participant_id", date_col]].dropna().merge(base[["participant_id", "baseline_date"]], on="participant_id")
    o["delta_days"] = (o[date_col] - o["baseline_date"]).dt.days
    o["abs"] = o["delta_days"].abs()
    best = o[o["abs"] <= window_days].sort_values(["participant_id", "abs", date_col]).groupby("participant_id", as_index=False).first()
    res = base[["participant_id"]].merge(best[["participant_id", date_col, "delta_days"]], on="participant_id", how="left")
    res["matched"] = res["delta_days"].notna()
    return res


def missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"n_missing": df.isna().sum(), "pct_missing": (df.isna().mean() * 100).round(2)}).rename_axis("column").reset_index()


def visit_counts(long_df: pd.DataFrame) -> pd.DataFrame:
    c = long_df.groupby("participant_id").size().rename("n_visits")
    return c.describe().rename("value").rename_axis("stat").reset_index()


def modality_overlap(avail: dict[str, set]) -> pd.DataFrame:
    """avail: modality -> set of participant ids. Returns counts per availability pattern."""
    ids = sorted(set().union(*avail.values())) if avail else []
    mods = list(avail)
    rows = [{"pattern": "+".join(m for m in mods if i in avail[m]) or "none"} for i in ids]
    df = pd.DataFrame(rows)
    return (df.value_counts().rename("n").reset_index() if len(df) else pd.DataFrame(columns=["pattern", "n"]))
