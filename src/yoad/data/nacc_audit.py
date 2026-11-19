"""NACC feasibility audit — SPECIFICATION AS CODE. Runs only on data the user is authorised to analyse.

DO NOT run against real NACC participant data in this project phase. Tested on synthetic NACC-shaped fixtures only.
Column names follow the public UDS Researchers Data Dictionary (uds4-rdd.csv, accessed 2026-10-12); they are configurable
because variable names/coding differ across UDS versions (FORMVER 1, 2, 3, 3.2, 4) and must be validated per release.
Only aggregate counts are returned; cell suppression for small counts is available.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MISSING_ONSET_CODES = (-4, 888, 999)  # RDD: -4 not collected, 888 no impairment/N-A, 999 unknown (confirm per release)


@dataclass
class AuditConfig:
    id: str = "NACCID"
    visit_num: str = "NACCVNUM"
    visit_date: str = "VISIT_DATE"          # assembled from VISITYR/VISITMO/VISITDAY by assemble_visit_date()
    cog_status: str = "NACCUDSD"            # 1 normal, 2 impaired-not-MCI, 3 MCI, 4 dementia, 8 behavioural-only
    etiology_ad: str = "NACCALZD"           # 1 = AD presumptive etiology (RDD: 0 no, 1 yes, 8 no impairment)
    age_baseline_col: str = "NACCAGEB"      # age at initial visit
    onset_col: str = "NACCCOGAGE"           # age cognitive impairment began (v1-4 harmonised); DECAGE not found in uds4 RDD
    baseline_rule: str = "initial_visit_mci"  # or "first_mci_visit"
    horizon_years: float = 2.0
    window_years: float = 0.5               # annual visits +/-6 months (NACC UDS page)
    cutoff_age: float = 65.0
    predictor_cols: tuple = ()
    mci_code: int = 3
    dementia_code: int = 4
    min_cell: int = 0                        # >0 masks counts below this (set per DUA/disclosure policy)
    extra: dict = field(default_factory=dict)


def assemble_visit_date(df: pd.DataFrame, y="VISITYR", m="VISITMO", d="VISITDAY") -> pd.Series:
    day = df[d] if d in df else 1
    return pd.to_datetime(dict(year=df[y], month=df[m], day=day), errors="coerce")


def clean_onset(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return x.mask(x.isin(MISSING_ONSET_CODES))


def _mask(n: int, cfg: AuditConfig):
    return n if (cfg.min_cell <= 0 or n == 0 or n >= cfg.min_cell) else f"<{cfg.min_cell}"


def select_baseline(v: pd.DataFrame, cfg: AuditConfig) -> pd.DataFrame:
    v = v.sort_values([cfg.id, cfg.visit_date])
    if cfg.baseline_rule == "initial_visit_mci":
        b = v[(v[cfg.visit_num] == 1) & (v[cfg.cog_status] == cfg.mci_code)]
    elif cfg.baseline_rule == "first_mci_visit":
        b = v[v[cfg.cog_status] == cfg.mci_code].groupby(cfg.id, as_index=False).first()
        return b
    else:
        raise ValueError(cfg.baseline_rule)
    return b.groupby(cfg.id, as_index=False).first()


def derive_outcomes(v: pd.DataFrame, base: pd.DataFrame, cfg: AuditConfig) -> pd.DataFrame:
    rows = []
    H = cfg.horizon_years * 365.25
    tol = cfg.window_years * 365.25
    for _, b in base.iterrows():
        g = v[(v[cfg.id] == b[cfg.id]) & (v[cfg.visit_date] > b[cfg.visit_date])].sort_values(cfg.visit_date)
        days = (g[cfg.visit_date] - b[cfg.visit_date]).dt.days
        any_ev = g[(g[cfg.cog_status] == cfg.dementia_code) & (days <= H)]
        ad_ev = any_ev[any_ev[cfg.etiology_ad] == 1] if cfg.etiology_ad in g else any_ev.iloc[0:0]
        last = days.max() if len(days) else np.nan
        # reversion: dementia followed by a non-dementia status (flag only)
        seq = g[cfg.cog_status].tolist(); rev = False; seen = False
        for s in seq:
            if s == cfg.dementia_code: seen = True
            elif seen and s in (1, 2, 3): rev = True
        rows.append({cfg.id: b[cfg.id], "n_followup_visits": int(len(g)), "last_followup_days": last,
                     "any_dementia_event": bool(len(any_ev)), "ad_dementia_event": bool(len(ad_ev)),
                     "observed_to_horizon": bool(pd.notna(last) and last >= H - tol), "reversion_flag": rev})
    o = pd.DataFrame(rows)
    # labels: event=1; non-event=0 only if observed to horizon; otherwise NaN (never 0 for short follow-up)
    for ep, col in (("any", "any_dementia_event"), ("ad", "ad_dementia_event")):
        lab = pd.Series(np.nan, index=o.index)
        lab[o[col]] = 1
        lab[(~o[col]) & o["observed_to_horizon"]] = 0
        o[f"label_{ep}"] = lab
    return o


def run_audit(visits: pd.DataFrame, cfg: AuditConfig) -> dict:
    """Aggregate-only feasibility audit. Returns nested dict of counts (masked if cfg.min_cell>0)."""
    v = visits.copy()
    need = [cfg.id, cfg.visit_num, cfg.visit_date, cfg.cog_status, cfg.age_baseline_col]
    missing = [c for c in need if c not in v.columns]
    if missing:
        raise KeyError(f"missing columns {missing}; map schema first")
    n_dup = int(v.duplicated([cfg.id, cfg.visit_date]).sum())
    v = v.drop_duplicates([cfg.id, cfg.visit_date])           # exact same-day duplicates dropped AFTER being counted
    base = select_baseline(v, cfg)
    out = {"config": {"baseline_rule": cfg.baseline_rule, "horizon_years": cfg.horizon_years, "cutoff_age": cfg.cutoff_age,
                      "onset_col": cfg.onset_col}, "duplicate_same_day_rows_dropped": n_dup,
           "n_participants_total": int(v[cfg.id].nunique()), "n_mci_baseline": _mask(len(base), cfg)}
    if base.empty:
        return out
    # onset: baseline-visit value vs latest non-missing (pre-specified sensitivity, not a selection of the better result)
    if cfg.onset_col in v.columns:
        v["_on"] = clean_onset(v[cfg.onset_col])
        first_val = v.sort_values(cfg.visit_date).groupby(cfg.id)["_on"].apply(lambda s: s.iloc[0] if len(s) else np.nan)
        latest = v.sort_values(cfg.visit_date).groupby(cfg.id)["_on"].apply(lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan)
        nuniq = v.groupby(cfg.id)["_on"].nunique()
        base = base.merge(first_val.rename("onset_baseline_visit"), left_on=cfg.id, right_index=True, how="left") \
                   .merge(latest.rename("onset_latest_nonmissing"), left_on=cfg.id, right_index=True, how="left") \
                   .merge(nuniq.rename("onset_n_distinct"), left_on=cfg.id, right_index=True, how="left")
    else:
        base["onset_baseline_visit"] = np.nan; base["onset_latest_nonmissing"] = np.nan; base["onset_n_distinct"] = 0
    ba = pd.to_numeric(base[cfg.age_baseline_col], errors="coerce")
    for rule in ("onset_baseline_visit", "onset_latest_nonmissing"):
        o = base[rule]; imposs = o.notna() & ba.notna() & (o > ba)
        out[rule] = {"n_nonmissing": _mask(int(o.notna().sum()), cfg), "n_missing": _mask(int(o.isna().sum()), cfg),
                     "n_onset_lt_cutoff": _mask(int((o < cfg.cutoff_age).sum()), cfg), "n_onset_ge_cutoff": _mask(int((o >= cfg.cutoff_age).sum()), cfg),
                     "n_onset_after_baseline_age_flagged": _mask(int(imposs.sum()), cfg)}
    out["onset_instability"] = {"n_with_more_than_one_distinct_value": _mask(int((base["onset_n_distinct"] > 1).sum()), cfg),
                                "n_baseline_vs_latest_differ": _mask(int((base["onset_baseline_visit"].notna() & base["onset_latest_nonmissing"].notna() & (base["onset_baseline_visit"] != base["onset_latest_nonmissing"])).sum()), cfg)}
    oc = derive_outcomes(v, base, cfg)
    base = base.merge(oc, on=cfg.id)
    out["followup"] = {"n_ge1_followup_visit": _mask(int((base["n_followup_visits"] >= 1).sum()), cfg),
                       "n_observed_to_horizon": _mask(int(base["observed_to_horizon"].sum()), cfg),
                       "n_censored_before_horizon_no_event_any": _mask(int(((~base["observed_to_horizon"]) & (~base["any_dementia_event"])).sum()), cfg),
                       "n_reversion_flagged": _mask(int(base["reversion_flag"].sum()), cfg)}
    strata = {}
    for name, rule in (("by_baseline_visit_onset", "onset_baseline_visit"), ("by_latest_onset", "onset_latest_nonmissing")):
        s = {}
        for lab, cond in (("onset_lt_cutoff", base[rule] < cfg.cutoff_age), ("onset_ge_cutoff", base[rule] >= cfg.cutoff_age), ("onset_missing", base[rule].isna())):
            sub = base[cond]
            s[lab] = {"n": _mask(len(sub), cfg)}
            for ep in ("any", "ad"):
                s[lab][f"{ep}_events"] = _mask(int((sub[f"label_{ep}"] == 1).sum()), cfg)
                s[lab][f"{ep}_nonevents_observed_to_horizon"] = _mask(int((sub[f"label_{ep}"] == 0).sum()), cfg)
                s[lab][f"{ep}_unevaluable_nan"] = _mask(int(sub[f"label_{ep}"].isna().sum()), cfg)
        strata[name] = s
    out["strata"] = strata
    if cfg.predictor_cols:
        pm = {}
        for c in cfg.predictor_cols:
            col = base[c] if c in base.columns else pd.Series(np.nan, index=base.index)
            x = pd.to_numeric(col, errors="coerce")
            pm[c] = {"n_available": _mask(int(x.notna().sum()), cfg), "pct_missing": float(round(100 * x.isna().mean(), 1))}
        cc = base[[c for c in cfg.predictor_cols if c in base.columns]].apply(pd.to_numeric, errors="coerce").notna().all(axis=1) if all(c in base.columns for c in cfg.predictor_cols) else pd.Series(False, index=base.index)
        out["predictors"] = pm
        out["eligible_for_frozen_model_validation"] = {
            "onset_lt_cutoff_any_evaluable_complete_predictors": _mask(int((cc & (base["onset_baseline_visit"] < cfg.cutoff_age) & base["label_any"].notna()).sum()), cfg),
            "onset_ge_cutoff_any_evaluable_complete_predictors": _mask(int((cc & (base["onset_baseline_visit"] >= cfg.cutoff_age) & base["label_any"].notna()).sum()), cfg)}
    return out
