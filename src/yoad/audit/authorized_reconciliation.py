"""LOCAL RECONCILIATION CHECKS - REQUIRES REAL DATA - NOT EXECUTED BY THE AI ASSISTANT.

These functions are meant to be run by an authorised researcher, in an approved environment, on the existing ADNI-derived
files. They return AGGREGATE results only (counts, booleans, hashes) - never participant identifiers. In this repository they
are exercised ONLY on synthetic fixtures (tests/test_authorized_reconciliation.py).

Column names follow the project's own documentation of FINAL_MCI_COHORT.csv (RID, baseline_date, conversion_status,
conversion_date, last_followup_date, event_date, censored, time_to_event_days, followup_duration_days, ...). Names are
parameters because the real files have not been inspected by the assistant.
"""
from __future__ import annotations

import hashlib
import sys

import numpy as np
import pandas as pd

CHECK_COHORT_COLUMNS = dict(id="RID", baseline="baseline_date", status="conversion_status", event_date="event_date",
                            last="last_followup_date", censored="censored", tte="time_to_event_days", fu="followup_duration_days")


def file_sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def one_row_per_participant(df: pd.DataFrame, id_col: str = "RID") -> dict:
    n = len(df); u = df[id_col].nunique(dropna=False)
    return {"rows": int(n), "unique_ids": int(u), "duplicate_rows": int(n - u), "null_ids": int(df[id_col].isna().sum()), "pass": bool(n == u and not df[id_col].isna().any())}


def recompute_cohort_counts(c: pd.DataFrame, cols=CHECK_COHORT_COLUMNS) -> dict:
    s = c[cols["status"]].astype(str).str.lower()
    conv = int(s.str.startswith("conv").sum()); n = len(c)
    return {"N": int(n), "converters": conv, "non_converters_or_censored": int(n - conv), "converter_proportion": round(conv / n, 4) if n else None}


def two_year_eligibility(c: pd.DataFrame, cols=CHECK_COHORT_COLUMNS, horizon_days: int = 730) -> dict:
    """PRIOR project rule (documented in FINAL_MULTIMODAL_COHORT_METHODS.txt): event=1 if converter and time_to_conversion<=730 d;
    event=0 if converter after 730 d or non-converter followed >=730 d; excluded if non-converter followed <730 d.
    NOTE: it labels a late converter 0 without requiring a non-dementia observation near the horizon (interval-censoring not checked)."""
    s = c[cols["status"]].astype(str).str.lower().str.startswith("conv")
    t = c[cols["tte"]].astype(float); fu = c[cols["fu"]].astype(float)
    ev = s & (t <= horizon_days); ne = (s & (t > horizon_days)) | (~s & (fu >= horizon_days)); ex = ~s & (fu < horizon_days)
    return {"eligible": int((ev | ne).sum()), "events": int(ev.sum()), "non_events": int(ne.sum()), "excluded_insufficient_followup": int(ex.sum()),
            "partition_ok": bool((ev | ne | ex).all() and not (ev & ne).any())}


def two_year_interval_censoring_exposure(c: pd.DataFrame, last_nondementia_days: str, cols=CHECK_COHORT_COLUMNS, horizon_days: int = 730, grace_days: int = 90) -> dict:
    """How many prior-rule non-events (late converters) lack a non-dementia observation at/after horizon-grace? (Definition A comparison.)
    `last_nondementia_days` = column with days from baseline to the last non-dementia visit (must be derived by the authorised researcher)."""
    s = c[cols["status"]].astype(str).str.lower().str.startswith("conv"); t = c[cols["tte"]].astype(float)
    late = s & (t > horizon_days); lnd = c[last_nondementia_days].astype(float)
    return {"late_converters_labelled_nonevent": int(late.sum()), "of_which_not_observed_nondementia_near_horizon": int((late & (lnd < horizon_days - grace_days)).sum())}


def outcome_censoring_consistency(c: pd.DataFrame, cols=CHECK_COHORT_COLUMNS) -> dict:
    s = c[cols["status"]].astype(str).str.lower().str.startswith("conv"); cen = c[cols["censored"]].astype(int)
    b = pd.to_datetime(c[cols["baseline"]]); ev = pd.to_datetime(c[cols["event_date"]], errors="coerce"); last = pd.to_datetime(c[cols["last"]], errors="coerce")
    tte = c[cols["tte"]].astype(float)
    return {"converter_but_censored": int((s & (cen == 1)).sum()), "nonconverter_not_censored": int((~s & (cen == 0)).sum()),
            "converter_missing_event_date": int((s & ev.isna()).sum()), "event_before_baseline": int((ev < b).sum()),
            "last_followup_before_baseline": int((last < b).sum()),
            "tte_inconsistent_with_dates_gt1d": int((s & ((ev - b).dt.days - tte).abs().gt(1)).sum())}


def feature_row_alignment(features: pd.DataFrame, cohort: pd.DataFrame, id_col: str = "RID") -> dict:
    f, c = set(features[id_col]), set(cohort[id_col])
    return {"feature_rows": len(features), "in_cohort": len(f & c), "not_in_cohort": len(f - c), "cohort_without_features": len(c - f)}


def mri_visit_matching(match: pd.DataFrame, baseline_col="baseline_date", mri_col="mri_date", window_days: int = 90) -> dict:
    d = (pd.to_datetime(match[mri_col]) - pd.to_datetime(match[baseline_col])).dt.days.abs()
    return {"n": int(len(match)), "within_window": int((d <= window_days).sum()), "outside_window": int((d > window_days).sum()), "missing_date": int(d.isna().sum())}


def complete_feature_count(df: pd.DataFrame, features: list[str]) -> dict:
    return {"n": len(df), "complete": int(df[features].notna().all(axis=1).sum())}


def policy_difference(old_flag: pd.Series, new_flag: pd.Series) -> dict:
    """Compare two availability definitions (e.g. label-based 903 vs strict-date 857)."""
    o, n = old_flag.astype(bool), new_flag.astype(bool)
    return {"old": int(o.sum()), "new": int(n.sum()), "in_old_not_new": int((o & ~n).sum()), "in_new_not_old": int((n & ~o).sum())}


def modality_intersections(avail: dict[str, set]) -> dict:
    names = sorted(avail); universe = set().union(*avail.values()) if avail else set(); pat = {}
    for rid in universe:
        key = "+".join(m for m in names if rid in avail[m]); pat[key] = pat.get(key, 0) + 1
    return {"patterns": dict(sorted(pat.items())), "n_any": len(universe)}


def split_integrity(oof: pd.DataFrame, id_col="RID", seed_col="seed", fold_col="fold") -> dict:
    """Each participant appears exactly once per seed (held out in exactly one fold)."""
    per = oof.groupby([seed_col, id_col])[fold_col].nunique()
    cnt = oof.groupby([seed_col, id_col]).size()
    return {"seeds": int(oof[seed_col].nunique()), "ids_with_multiple_folds_in_a_seed": int((per > 1).sum()), "ids_repeated_within_seed": int((cnt > 1).sum()),
            "pass": bool((per <= 1).all() and (cnt == 1).all())}


def oof_agreement(oof: pd.DataFrame, labels: pd.DataFrame, id_col="RID", y_col="event_2y", p_col="pred", reported_auc: float | None = None, tol: float = 0.01) -> dict:
    from scipy.stats import rankdata
    m = oof.merge(labels[[id_col, y_col]], on=id_col, how="left", suffixes=("_saved", ""))
    ysaved = f"{y_col}_saved"
    mism = int((m[ysaved] != m[y_col]).sum()) if ysaved in m else None
    y = m[y_col].to_numpy(float); p = m[p_col].to_numpy(float); ok = ~np.isnan(y)
    r = rankdata(p[ok]); n1 = y[ok].sum(); n0 = ok.sum() - n1
    auc = float((r[y[ok] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    out = {"n": int(ok.sum()), "events": int(n1), "label_mismatches_vs_cohort": mism, "recomputed_auc": round(auc, 4), "ids_without_label": int((~ok).sum())}
    if reported_auc is not None: out["matches_reported_within_tol"] = bool(abs(auc - reported_auc) <= tol)
    return out


def count_matching_ids(df: pd.DataFrame, id_col: str, pattern: str) -> dict:
    """Count (never list) rows whose identifier matches a regex, e.g. the ADNI March-2026 advisory PTIDs r'^381_S_\\d{5}$'."""
    m = df[id_col].astype(str).str.match(pattern)
    return {"rows": int(len(df)), "matching": int(m.sum()), "unique_matching": int(df.loc[m, id_col].nunique())}


def main(argv=None) -> int:  # pragma: no cover - requires real data
    sys.stderr.write("This module is a library for AUTHORISED local use; it is intentionally not wired to real file paths here.\n")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
