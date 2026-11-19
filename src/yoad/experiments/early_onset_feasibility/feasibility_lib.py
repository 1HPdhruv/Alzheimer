"""Early-onset feasibility audit (ADNI) - pure functions, no file access, aggregate outputs only.

Answers, for the month-0 MCI population of the frozen 2 x 2 protocol (src/cohort/anchor_predictors.eligible_anchors) and a broader
month-0 MCI population without the 24-month window rule, by age-at-baseline band and by retrospective age-at-onset band:
  - how many participants, and how many have an onset year that gives a plausible age at onset (PTDEMOG PTCOGBEG / PTMCIBEG /
    PTADBEG; year-level, retrospective best estimates; never treated as verified onset);
  - 24-month calendar-rule labels (any-cause and AD-type), and dementia events, deaths and known event-free status at 1-4 years;
  - Aalen-Johansen cumulative incidence of dementia with death as a competing event, with participant-bootstrap intervals;
  - dated assessments per participant (>= 1..5) and the distribution of intervals between assessments;
  - landmark feasibility: participants event-free and under observation at 1 and 2 years with >= 2 dated cognitive assessments,
    and their later events.
Age at baseline is NOT age at onset. A dementia diagnosis date is the visit at which it was recorded (interval-censored onset).
Every count goes through audit_lib.sup (1..10 -> "<11"); summary statistics are withheld unless their denominator is >= K.
No threshold for "enough events" is applied here: the counts are reported, and the decision is made in the audit report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from yoad.experiments.visit_anchor_audit.audit_lib import K, sup
from yoad.features.anchor_predictors import eligible_anchors
from yoad.data.reference_outcome import EVENT, NONEVENT
from yoad.data.visit_anchor_labels import label_calendar_reference

AGE_BANDS = ((None, 55, "<55"), (55, 60, "55-59"), (60, 65, "60-64"), (65, 70, "65-69"), (70, 75, "70-74"), (75, None, "75+"))
ONSET_FIELDS = ("PTCOGBEG", "PTMCIBEG", "PTADBEG")     # year of onset of cognitive / MCI / AD symptoms (ADNI dictionary)
ONSET_NOT_APPLICABLE = 9999                           # documented code (cognitively normal / not applicable)
HORIZONS_YEARS = (1, 2, 3, 4)
LANDMARKS_YEARS = (1, 2)
COGNITIVE = ("mmse", "adas13", "faq", "cdrsb")
YEAR = 365.25


def age_band(age) -> str:
    if age is None or pd.isna(age):
        return "missing"
    for lo, hi, name in AGE_BANDS:
        if (lo is None or age >= lo) and (hi is None or age < hi):
            return name
    return "missing"


def birth_years(ptdemog: pd.DataFrame) -> pd.Series:
    """Birth year per RID from PTDOB (a date) or PTDOBYY; rows of one RID that disagree -> NaN."""
    d = ptdemog.copy()
    d["RID"] = pd.to_numeric(d["RID"], errors="coerce")
    y = pd.to_datetime(d["PTDOB"], errors="coerce").dt.year if "PTDOB" in d else pd.Series(np.nan, index=d.index)
    if "PTDOBYY" in d:
        y = y.fillna(pd.to_numeric(d["PTDOBYY"], errors="coerce"))
    d["by"] = pd.to_numeric(y, errors="coerce").where(lambda s: s > 1880)
    g = d.dropna(subset=["RID", "by"]).groupby("RID")["by"]
    return g.agg(lambda s: s.iloc[0] if s.nunique() == 1 else np.nan)


def onset_table(ptdemog: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    """Per participant of `anchors` (participant_id, anchor_date): for each onset field a status in {valid, missing,
    not_applicable, conflicting, implausible, no_birth_year, field_absent} and the derived age at onset in whole years
    (onset year - birth year; year granularity, so +/- 1 year). Valid = birth year + 18 <= onset year <= anchor year.
    'earliest' = the earliest valid onset year among the fields."""
    by = birth_years(ptdemog)
    d = ptdemog.copy(); d["RID"] = pd.to_numeric(d["RID"], errors="coerce")
    out = pd.DataFrame({"participant_id": anchors["participant_id"].to_numpy()})
    ay = pd.to_datetime(anchors["anchor_date"]).dt.year.to_numpy()
    b = out["participant_id"].map(by).to_numpy(dtype=float)
    valid_years = []
    for f in ONSET_FIELDS:
        if f not in d.columns:
            out[f"{f}_status"] = "field_absent"; out[f"{f}_age"] = np.nan
            continue
        v = pd.to_numeric(d[f], errors="coerce")
        v = v.where(v >= 0)                                           # negative ADNI codes -> missing
        rows = pd.DataFrame({"RID": d["RID"], "v": v}).dropna(subset=["RID"])
        agg = rows.dropna(subset=["v"]).groupby("RID")["v"].agg(lambda s: s.iloc[0] if s.nunique() == 1 else -1.0)
        raw = out["participant_id"].map(agg).to_numpy(dtype=float)
        st = np.full(len(out), "missing", dtype=object)
        st[raw == ONSET_NOT_APPLICABLE] = "not_applicable"
        st[raw == -1.0] = "conflicting"
        cand = ~np.isnan(raw) & (raw != ONSET_NOT_APPLICABLE) & (raw != -1.0)
        st[cand & np.isnan(b)] = "no_birth_year"
        ok = cand & ~np.isnan(b) & (raw >= b + 18) & (raw <= ay)
        st[cand & ~np.isnan(b) & ~ok] = "implausible"
        st[ok] = "valid"
        yrs = np.where(ok, raw, np.nan)
        out[f"{f}_status"] = st
        out[f"{f}_age"] = yrs - b
        valid_years.append(yrs)
    ey = pd.DataFrame(np.vstack(valid_years).T).min(axis=1).to_numpy() if valid_years else np.full(len(out), np.nan)
    out["earliest_age"] = ey - b
    out["earliest_status"] = np.where(np.isnan(out["earliest_age"]), "none_valid", "valid")
    return out


def onset_band(age) -> str:
    if age is None or pd.isna(age):
        return "unknown"
    return "<65" if age < 65 else ">=65"


def follow_up(pop: pd.DataFrame, visits: pd.DataFrame, deaths: dict | None, data_cutoff) -> pd.DataFrame:
    """Per participant: years from the anchor to the first recorded dementia (event), to death without earlier dementia,
    and to the last dated diagnosis visit on/before the cut-off (end of observation). status: 1 dementia, 2 death, 0 censored."""
    v = visits.copy(); v["date"] = pd.to_datetime(v["date"], errors="coerce")
    cut = pd.Timestamp(data_cutoff)
    v = v[v["date"].notna() & (v["date"] <= cut)]
    rows = []
    for r in pop.itertuples(index=False):
        g = v[(v["participant_id"] == r.participant_id) & (v["date"] > r.anchor_date)]
        dd = (deaths or {}).get(r.participant_id)
        dd = pd.Timestamp(dd) if dd is not None and not pd.isna(dd) else None
        if dd is not None:                                   # records dated after death are discarded (as in the REF rule)
            g = g[g["date"] <= dd]
        dem = g[g["dx"] == "DEM"]["date"]
        last = g["date"].max() if len(g) else r.anchor_date
        if len(dem) and (dd is None or dem.min() <= dd):
            t, s = (dem.min() - r.anchor_date).days / YEAR, 1
        elif dd is not None and dd <= cut:
            t, s = (dd - r.anchor_date).days / YEAR, 2
        else:
            t, s = (last - r.anchor_date).days / YEAR, 0
        rows.append((r.participant_id, max(t, 0.0), s, (last - r.anchor_date).days / YEAR))
    return pd.DataFrame(rows, columns=["participant_id", "time", "status", "last_visit_years"])


def horizon_counts(fu: pd.DataFrame, years=HORIZONS_YEARS) -> list[dict]:
    """At each horizon t: dementia by t, death without dementia by t, known event-free (observed beyond t, no event), unknown."""
    out = []
    for t in years:
        ev = (fu.status == 1) & (fu.time <= t)
        de = (fu.status == 2) & (fu.time <= t)
        free = ~ev & ~de & (fu.last_visit_years >= t)
        out.append({"horizon_years": t, "n": len(fu), "dementia_by_t": int(ev.sum()), "death_without_dementia_by_t": int(de.sum()),
                    "known_event_free_at_t": int(free.sum()), "status_unknown_at_t": int(len(fu) - ev.sum() - de.sum() - free.sum())})
    return out


def aalen_johansen(time, status, t_grid) -> np.ndarray:
    """Cumulative incidence of cause 1 with cause 2 competing (right-censored data); estimates at each t in t_grid."""
    time = np.asarray(time, float); status = np.asarray(status, int)
    uniq = np.unique(time[status > 0])
    S, F, res, j = 1.0, 0.0, [], 0
    grid = sorted(t_grid)
    vals = {}
    for u in uniq:
        n = (time >= u).sum()
        d1 = ((time == u) & (status == 1)).sum(); dall = ((time == u) & (status > 0)).sum()
        while j < len(grid) and grid[j] < u:
            vals[grid[j]] = F; j += 1
        if n:
            F += S * d1 / n; S *= 1 - dall / n
    while j < len(grid):
        vals[grid[j]] = F; j += 1
    return np.array([vals[t] for t in t_grid])


def aj_with_ci(fu: pd.DataFrame, years=HORIZONS_YEARS, B: int = 200, seed: int = 20261025) -> list[dict]:
    """Aalen-Johansen estimate with a participant-bootstrap 95% percentile interval. Withheld when n < K or when fewer than K
    dementia events occur by the horizon."""
    rng = np.random.default_rng(seed)
    est = aalen_johansen(fu.time, fu.status, years)
    boots = np.array([aalen_johansen(fu.time.to_numpy()[i], fu.status.to_numpy()[i], years)
                      for i in (rng.integers(0, len(fu), len(fu)) for _ in range(B))]) if len(fu) else np.empty((0, len(years)))
    out = []
    for k, t in enumerate(years):
        events = int(((fu.status == 1) & (fu.time <= t)).sum())
        ok = len(fu) >= K and events >= K
        out.append({"horizon_years": t, "cif_dementia": round(float(est[k]), 4) if ok else None,
                    "ci_low": round(float(np.percentile(boots[:, k], 2.5)), 4) if ok else None,
                    "ci_high": round(float(np.percentile(boots[:, k], 97.5)), 4) if ok else None,
                    "withheld": None if ok else f"n or events < {K}", "bootstrap_B": B})
    return out


def measurement_profile(pop: pd.DataFrame, meas: pd.DataFrame, fu: pd.DataFrame, lo_days: int = -90) -> tuple[list, list]:
    """Per instrument: participants with >= k dated assessments in [anchor + lo_days, end of observation]; pooled distribution
    of intervals between consecutive assessments (days). Summaries withheld unless >= K participants and intervals."""
    end = pop[["participant_id", "anchor_date"]].merge(fu[["participant_id", "last_visit_years", "time", "status"]], on="participant_id")
    end["end_date"] = end["anchor_date"] + pd.to_timedelta(np.maximum(end["last_visit_years"], end["time"]) * YEAR, unit="D")
    m = meas.dropna(subset=["date", "value"]).copy(); m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m = m.merge(end[["participant_id", "anchor_date", "end_date"]], on="participant_id", how="inner")
    m = m[(m["date"] >= m["anchor_date"] + pd.Timedelta(days=lo_days)) & (m["date"] <= m["end_date"])]
    m = m.drop_duplicates(["participant_id", "variable", "date"])
    counts, gaps = [], []
    for var in sorted(m["variable"].unique()) if len(m) else []:
        mv = m[m["variable"] == var].sort_values(["participant_id", "date"])
        per = mv.groupby("participant_id").size().reindex(pop["participant_id"], fill_value=0)
        counts.append({"instrument": var, "n_population": len(pop),
                       **{f"n_with_ge_{k}": int((per >= k).sum()) for k in (1, 2, 3, 4, 5)}})
        d = mv.groupby("participant_id")["date"].diff().dt.days.dropna()
        n_part = int(mv.groupby("participant_id").size().ge(2).sum())
        ok = len(d) >= K and n_part >= K
        q = (lambda p: float(np.percentile(d, p))) if ok else (lambda p: None)
        gaps.append({"instrument": var, "n_intervals": len(d), "n_participants_with_interval": n_part,
                     "p10_days": q(10), "p25_days": q(25), "median_days": q(50), "p75_days": q(75), "p90_days": q(90),
                     "withheld": None if ok else f"fewer than {K} intervals or participants"})
    return counts, gaps


def landmark_profile(pop: pd.DataFrame, meas: pd.DataFrame, fu: pd.DataFrame, landmarks=LANDMARKS_YEARS,
                     horizons=(1, 2, 3), min_assessments: int = 2) -> list[dict]:
    """At landmark s: participants event-free, alive and observed at/after s, with >= min_assessments dated cognitive assessments
    (any of COGNITIVE) between anchor - 90 d and anchor + s; their dementia events in (s, s + h]."""
    m = meas[meas["variable"].isin(COGNITIVE)].dropna(subset=["date", "value"]).copy()
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m = m.merge(pop[["participant_id", "anchor_date"]], on="participant_id", how="inner")
    m["t"] = (m["date"] - m["anchor_date"]).dt.days / YEAR
    out = []
    for s in landmarks:
        at_risk = fu[(fu.time > s) & (fu.last_visit_years >= s)]
        hist = m[(m.t >= -90 / YEAR) & (m.t <= s)].drop_duplicates(["participant_id", "date"]).groupby("participant_id").size()
        lm = at_risk[at_risk.participant_id.map(hist).fillna(0) >= min_assessments]
        rec = {"landmark_years": s, "n_at_risk": len(at_risk), f"n_at_risk_with_ge_{min_assessments}_dated_cognitive": len(lm)}
        for h in horizons:
            rec[f"dementia_in_(s,s+{h}]"] = int(((lm.status == 1) & (lm.time <= s + h)).sum())
        out.append(rec)
    return out


def _strata(pop: pd.DataFrame, onset: pd.DataFrame) -> dict:
    """{(stratum_type, stratum): boolean mask over pop rows}. Age at baseline and retrospective age at onset are kept apart."""
    pid = pop["participant_id"].to_numpy()
    on = onset.set_index("participant_id")
    s = {("all", "all"): np.ones(len(pop), bool)}
    bands = pop["baseline_age"].map(age_band).to_numpy()
    for _, _, name in AGE_BANDS:
        s[("age_at_baseline_band", name)] = bands == name
    s[("age_at_baseline_band", "missing")] = bands == "missing"
    s[("age_at_baseline", "<65")] = pop["baseline_age"].to_numpy() < 65
    for f in (*ONSET_FIELDS, "earliest"):
        a = pd.Series(pid).map(on[f"{f}_age"]).map(onset_band).to_numpy()
        for b in ("<65", ">=65", "unknown"):
            s[(f"age_at_onset_{f}", b)] = a == b
    return s


def run_audit(visits: pd.DataFrame, measurements: pd.DataFrame, static: pd.DataFrame, ptdemog: pd.DataFrame,
              data_cutoff, death_dates: dict | None = None, ad_visits: pd.DataFrame | None = None, B: int = 200) -> dict:
    """Returns {"tables": {name: DataFrame, aggregate only}, "metadata": dict}. ad_visits: visits with a dem_type column for
    the AD-type labels (omit to skip them)."""
    cut = pd.Timestamp(data_cutoff)
    v = visits.copy(); v["date"] = pd.to_datetime(v["date"], errors="coerce")
    elig, eflow = eligible_anchors(v, cut, death_dates=death_dates)
    pops = {"month0_mci_window_closed": elig[elig["is_baseline"]][["participant_id", "anchor_date"]]}
    first_dem = v[v["dx"] == "DEM"].groupby("participant_id")["date"].min()
    m0 = v[(v["visit_month"] == 0) & (v["dx"] == "MCI") & v["date"].notna()][["participant_id", "date"]].rename(columns={"date": "anchor_date"})
    m0 = m0[~(m0["participant_id"].map(first_dem) <= m0["anchor_date"])]
    pops["month0_mci_all"] = m0.drop_duplicates("participant_id")
    st = static.set_index("participant_id")
    tables = {k: [] for k in ("population", "onset_availability", "labels_24m", "horizon_counts", "cumulative_incidence",
                              "measurement_counts", "measurement_intervals", "landmark", "modality_baseline")}
    for pname, pop in pops.items():
        pop = pop.reset_index(drop=True).copy()
        pop["baseline_age"] = pop["participant_id"].map(st["baseline_age"]) if "baseline_age" in st else np.nan
        onset = onset_table(ptdemog, pop)
        fu = follow_up(pop, v, death_dates, cut)
        anc = pop.assign(anchor_month=0)[["participant_id", "anchor_month", "anchor_date"]]
        lab_any = label_calendar_reference(v, anc, death_dates=death_dates, data_cutoff=cut)
        lab_ad = (label_calendar_reference(ad_visits.assign(date=lambda d: pd.to_datetime(d["date"], errors="coerce")), anc,
                                           death_dates=death_dates, data_cutoff=cut, endpoint="ad") if ad_visits is not None else None)
        for f in (*ONSET_FIELDS, "earliest"):
            for status, n in onset[f"{f}_status"].value_counts().items():
                tables["onset_availability"].append({"population": pname, "field": f, "status": status, "n": sup(n)})
        for (stype, sname), mask in _strata(pop, onset).items():
            key = {"population": pname, "stratum_type": stype, "stratum": sname}
            sub, fsub = pop[mask], fu[mask]
            tables["population"].append({**key, "n_participants": sup(len(sub))})
            for endpoint, lab in (("any_cause", lab_any), ("ad_type", lab_ad)):
                if lab is None:
                    continue
                s = lab["state"].to_numpy()[mask]
                tables["labels_24m"].append({**key, "endpoint": endpoint, "n_event": sup((s == EVENT).sum()),
                                             "n_nonevent": sup((s == NONEVENT).sum()),
                                             "n_indeterminate": sup((~np.isin(s, [EVENT, NONEVENT])).sum())})
            for h in horizon_counts(fsub):
                tables["horizon_counts"].append({**key, **{k: (sup(x) if k != "horizon_years" else x) for k, x in h.items()}})
            for c in aj_with_ci(fsub.reset_index(drop=True), B=B):
                tables["cumulative_incidence"].append({**key, **c})
            if stype in ("all", "age_at_baseline", "age_at_onset_earliest"):
                cnt, gap = measurement_profile(sub, measurements, fsub)
                tables["measurement_counts"] += [{**key, **{k: (sup(x) if k.startswith("n_") else x) for k, x in c.items()}} for c in cnt]
                tables["measurement_intervals"] += [{**key, **{k: (sup(x) if k.startswith("n_") else x) for k, x in g.items()}} for g in gap]
                tables["landmark"] += [{**key, **{k: (sup(x) if k != "landmark_years" else x) for k, x in r.items()}}
                                       for r in landmark_profile(sub, measurements, fsub)]
                for col in ("baseline_age", "female", "education_years", "apoe4"):
                    present = sub["participant_id"].map(st[col]).notna().sum() if col in st else 0
                    tables["modality_baseline"].append({**key, "variable": col, "n_present": sup(present), "n": sup(len(sub))})
    out = {k: pd.DataFrame(x) for k, x in tables.items()}
    meta = {"data_cutoff": str(cut.date()), "eligibility_flow": {k: sup(x) for k, x in eflow.items()},
            "populations": {k: sup(len(p)) for k, p in pops.items()}, "suppression": f"counts 1..{K - 1} shown as '<{K}'",
            "bootstrap_B": B, "ad_type_labels": ad_visits is not None,
            "notes": ["age at baseline is not age at onset",
                      "onset fields are retrospective, year-level best estimates (+/- 1 year); not verified onset",
                      "dementia dates are visit dates (interval-censored onset)",
                      "differences between totals and suppressed cells may reveal small counts: review before sharing"]}
    return {"tables": out, "metadata": meta}
