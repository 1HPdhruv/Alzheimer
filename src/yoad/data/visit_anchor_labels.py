"""Visit-anchored 24-month labels: the unit of analysis is an MCI VISIT (an "anchor"), not a participant.

Two label rules on the same anchors, so their effect can be isolated:

* `label_visitcode_sim2026` - re-implementation, from the published methods and the publicly visible logic, of the label
  construction described in Simkhada et al. 2026 (GeroScience, doi 10.1007/s11357-026-02551-x). The authors' repository has no
  licence; no code was copied. Rule, per MCI anchor at protocol month m, using diagnoses recorded at the EXACT visit codes
  m+6, m+12, m+18, m+24:
    - any CN at those visits                                   -> EXCLUDED  reverted_to_cn
    - otherwise D24 := diagnosis at m+24; if missing, filled from the first observed CHANGE at m+18, else m+12, else m+6
    - D24 still missing                                        -> EXCLUDED  no_m24_status   (stable people need an m+24 visit)
    - D24 == DEM                                               -> EVENT
    - D24 == MCI but a change was observed at m+6/m+12         -> EXCLUDED  changed_then_mci (e.g. DEM then MCI)
    - D24 == MCI                                               -> NONEVENT  (this includes DEM at m+18 followed by MCI at m+24:
                                                                  the public logic only checks changes at m+6/m+12 here)
  Asymmetry (an analytic observation, not an author claim): a converter is labelled from ANY of the four visits, whereas a
  non-converter needs the m+24 visit, so anchors lost to follow-up are excluded only when they did not convert.

* `label_calendar_reference` - the project's v1.3 three-state reference (src/cohort/reference_outcome.py) applied at each anchor
  date: calendar horizon, first-dementia rule (reversions do not exclude), non-event needs a non-dementia diagnosis in
  [H - w, H + u], missing follow-up is never a non-event.

Input `visits`: one row per assessment with columns participant_id, visit_month (protocol month code, int), date, dx in
{"CN", "MCI", "DEM"} (anything else = uncertain). Pure functions; no file access. Synthetic-tested (tests/test_visit_anchor_labels.py).
"""
from __future__ import annotations

import pandas as pd

from yoad.data.reference_outcome import EVENT, INDET, NONEVENT, reference_state

EXCLUDED = "EXCLUDED"
DX = {"CN", "MCI", "DEM"}


def _check(visits: pd.DataFrame) -> pd.DataFrame:
    need = {"participant_id", "visit_month", "date", "dx"}
    miss = need - set(visits.columns)
    if miss:
        raise ValueError(f"visits missing columns {sorted(miss)}")
    v = visits.copy()
    v["date"] = pd.to_datetime(v["date"], errors="coerce")
    if v["participant_id"].isna().any():
        raise ValueError("missing participant_id")
    dup = v.duplicated(["participant_id", "visit_month"], keep=False)
    conflicting = v[dup].groupby(["participant_id", "visit_month"])["dx"].nunique()
    if (conflicting > 1).any():
        raise ValueError(f"{int((conflicting > 1).sum())} participant-visit(s) carry conflicting diagnoses; resolve upstream")
    return v.drop_duplicates(["participant_id", "visit_month"])


def anchors(visits: pd.DataFrame, policy: str = "all_mci_visits") -> pd.DataFrame:
    """MCI anchors. policy='all_mci_visits' (visit-level samples, as in the reference paper) or 'first_mci_visit'
    (one anchor per participant: the participant-baseline design)."""
    v = _check(visits)
    a = v[v["dx"] == "MCI"].sort_values(["participant_id", "visit_month"])
    if policy == "first_mci_visit":
        a = a.groupby("participant_id", as_index=False).first()
    elif policy != "all_mci_visits":
        raise ValueError("policy must be 'all_mci_visits' or 'first_mci_visit'")
    return a[["participant_id", "visit_month", "date"]].rename(columns={"visit_month": "anchor_month", "date": "anchor_date"}).reset_index(drop=True)


def drop_post_dementia_anchors(visits: pd.DataFrame, anchor_table: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Protocol eligibility for the calendar-label paths: an MCI anchor dated on/after the participant's first recorded dementia
    (a reversion) is not an anchor; otherwise the reference rule, which ignores visits on/before the index, could call it a
    non-event. Returns (kept anchors, number dropped). Not applied to the visit-code reference rule (faithful reproduction)."""
    v = visits.copy(); v["date"] = pd.to_datetime(v["date"], errors="coerce")
    first = v[v["dx"] == "DEM"].groupby("participant_id")["date"].min()
    fd = anchor_table["participant_id"].map(first)
    bad = fd.notna() & (pd.to_datetime(anchor_table["anchor_date"]) >= fd)
    return anchor_table[~bad].reset_index(drop=True), int(bad.sum())


def label_visitcode_sim2026(visits: pd.DataFrame, anchor_table: pd.DataFrame) -> pd.DataFrame:
    v = _check(visits).set_index(["participant_id", "visit_month"])["dx"]
    out = []
    for r in anchor_table.itertuples(index=False):
        d = {k: v.get((r.participant_id, r.anchor_month + k)) for k in (6, 12, 18, 24)}
        if any(x == "CN" for x in d.values()):
            out.append((EXCLUDED, "reverted_to_cn")); continue
        changes = {k: x for k, x in d.items() if k in (6, 12, 18) and x is not None and x != "MCI"}
        d24 = d[24]
        if d24 is None:
            for k in (18, 12, 6):
                if k in changes:
                    d24 = changes[k]; break
        if d24 is None:
            out.append((EXCLUDED, "no_m24_status")); continue
        if d24 == "DEM":
            out.append((EVENT, "dementia_by_m24_visitcode")); continue
        if d24 == "MCI" and any(k in changes for k in (6, 12)):
            out.append((EXCLUDED, "changed_then_mci")); continue
        if d24 == "MCI":
            out.append((NONEVENT, "mci_at_m24")); continue
        out.append((EXCLUDED, "uncertain_m24_diagnosis"))
    res = anchor_table.copy()
    res["state"], res["reason"] = zip(*out) if out else ([], [])
    return res


def label_calendar_reference(visits: pd.DataFrame, anchor_table: pd.DataFrame, w_days: int = 90, u_days: int | None = 90,
                             death_dates: dict | None = None, data_cutoff=None, event_grace_days: int = 0,
                             endpoint: str = "any") -> pd.DataFrame:
    """endpoint "any" (any-cause recorded dementia; the primary) or "ad" (AD-type recorded dementia; a sensitivity analysis,
    which needs a dem_type column in {AD, OTHER}: a missing or mixed etiology is never read as AD)."""
    if endpoint == "ad" and "dem_type" not in visits.columns:
        raise ValueError("endpoint='ad' needs a dem_type column (AD / OTHER); dementia is never assumed to be AD-type")
    v = _check(visits)
    v["dx3"] = v["dx"].map({"DEM": "DEM", "CN": "NONDEM", "MCI": "NONDEM"})
    cols = ["date", "dx3"] + (["dem_type"] if endpoint == "ad" else [])
    groups = {pid: g[cols].rename(columns={"dx3": "dx"}) for pid, g in v.groupby("participant_id")}
    out = []
    for r in anchor_table.itertuples(index=False):
        s, why, _ = reference_state(r.anchor_date, groups[r.participant_id], w_days=w_days, u_days=u_days, endpoint=endpoint,
                                    death_date=(death_dates or {}).get(r.participant_id), data_cutoff=data_cutoff,
                                    event_grace_days=event_grace_days)
        out.append((s, why))
    res = anchor_table.copy()
    res["state"], res["reason"] = zip(*out) if out else ([], [])
    return res


def state_summary(labelled: pd.DataFrame, prefix: str = "") -> dict:
    """Event, non-event, competing-death, other-indeterminate and excluded counts, reported separately."""
    s, r = labelled["state"], labelled["reason"]
    return {f"{prefix}n_event": int((s == EVENT).sum()), f"{prefix}n_nonevent": int((s == NONEVENT).sum()),
            f"{prefix}n_competing_death": int(((s == INDET) & (r == "competing_death")).sum()),
            f"{prefix}n_indeterminate_other": int(((s == INDET) & (r != "competing_death")).sum()),
            f"{prefix}n_excluded_visitcode_rule": int((s == EXCLUDED).sum())}


def summarise(labelled: pd.DataFrame, k_suppress: int = 11) -> pd.DataFrame:
    """Aggregate counts by state and reason (anchors and participants); counts 1..k-1 suppressed to '<k'."""
    g = labelled.groupby(["state", "reason"]).agg(n_anchors=("participant_id", "size"),
                                                   n_participants=("participant_id", "nunique")).reset_index()
    sup = lambda n: n if n == 0 or n >= k_suppress else f"<{k_suppress}"
    for c in ("n_anchors", "n_participants"):
        g[c] = g[c].map(sup)
    return g
