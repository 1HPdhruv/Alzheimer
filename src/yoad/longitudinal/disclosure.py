"""Disclosure control for every exported file of temporal multimodal v1 (PROTOCOL.md §11). Pure functions; no file access.

Rules (person and anchor counts; K = 11):
  * a count of 1..10 is shown as the neutral label "Suppressed"; 0 and counts >= 11 are shown. The label never says whether
    a cell is above or below 10, so primary and secondary suppression are indistinguishable;
  * partition tables (cells that sum to a published total): if exactly one non-zero cell is suppressed, the smallest other
    non-zero cell is also suppressed ("Suppressed");
  * a count whose complement against a published total is 1..10 is suppressed as well;
  * an evaluation subset whose anchors, event anchors or non-event anchors, or whose complement within all anchors, is 1..10
    loses its counts, and a row with 1..10 events or non-events loses its threshold metrics (sensitivity, specificity, PPV,
    NPV, accuracy, balanced accuracy) and log O:E, which with the denominator would reconstruct the event count;
  * exact prevalence (events / n) is never exported (row-A metrics and bootstrap tables included).
`audit` checks CSV tables, JSON objects and report text together before anything is written.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

K = 11
LABEL = "Suppressed"
THRESHOLD_FIELDS = ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "ppv", "npv", "log_oe", "mcc", "f1_macro")
COUNT_COLUMNS = ("anchors", "event_anchors", "participants", "event_participants", "nonevent_participants",
                 "nonevent_only_participants", "events", "n")


def is_small(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, bool) and 0 < int(v) < K


def sup(v):
    return LABEL if is_small(v) else v


def partition(values: list) -> list:
    """Primary + secondary suppression of cells that add up to a published total."""
    out = [sup(v) for v in values]
    supp = [i for i, v in enumerate(values) if is_small(v)]
    if len(supp) == 1:
        rest = [i for i, v in enumerate(values) if isinstance(v, (int, np.integer)) and v > 0 and i not in supp]
        if rest:
            out[min(rest, key=lambda i: values[i])] = LABEL
    return out


def partition_df(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.Series(partition([int(v) for v in df[col]]), index=df.index, dtype=object)
    return out


def complement(v: int, total: int):
    return LABEL if is_small(v) or is_small(int(total) - int(v)) else v


def _drop_threshold(row: dict) -> dict:
    for f in THRESHOLD_FIELDS:
        for k in list(row):
            if k == f or k.startswith(f + "_"):
                row[k] = np.nan
    return row


def eval_set_rows(rows: list, key=("candidate", "output", "weighting")) -> list:
    """rows: result dicts with raw integer anchors / event_anchors per eval_set. Applies the subset and threshold rules."""
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(tuple(r.get(k) for k in key), {})[r["eval_set"]] = i
    out = [dict(r) for r in rows]
    for g in groups.values():
        n = {es: (rows[i]["anchors"], rows[i]["event_anchors"]) for es, i in g.items()}
        bad_subsets = False
        if "all_anchors" in n:
            a, e = n["all_anchors"]
            sa = sum(n[es][0] for es in n if es != "all_anchors"); se = sum(n[es][1] for es in n if es != "all_anchors")
            bad_subsets = any(is_small(x) for x in (a - sa, e - se, (a - e) - (sa - se)))
        for es, i in g.items():
            a, e = n[es]
            small_row = any(is_small(x) for x in (a, e, a - e))
            if small_row or (bad_subsets and es != "all_anchors"):
                out[i]["anchors"], out[i]["event_anchors"] = LABEL, LABEL
            if small_row:
                _drop_threshold(out[i])
    return out


def export_row_a(simk: dict) -> tuple[dict, pd.DataFrame | None]:
    """Row-A (Simkhada published configuration) export: counts with partition / complement suppression, no prevalence, and
    threshold metrics withheld when the test events or non-events are 1..10."""
    s = json.loads(json.dumps({k: v for k, v in simk.items() if k not in ("flow", "test_bootstrap")}, default=str))
    boot = simk.get("test_bootstrap")
    c = s.get("counts")
    if c:
        pm, sm = partition([c["pMCI"], c["sMCI"]])
        s["counts"] = {"samples": sup(c["samples"]), "participants": sup(c["participants"]), "pMCI": pm, "sMCI": sm}
    f = s.get("final", {})
    small_test = False
    for part in ("train", "test"):
        n, e = f.get(f"n_{part}_samples"), f.get(f"{part}_events")
        if n is None or e is None:
            continue
        bad = any(is_small(x) for x in (n, e, n - e))
        small_test |= bad and part == "test"
        f[f"n_{part}_samples"], f[f"{part}_events"] = (LABEL, LABEL) if bad else (n, e)
        f[f"n_{part}_participants"] = sup(f.get(f"n_{part}_participants"))
    if f.get("train_events") == LABEL or f.get("test_events") == LABEL:
        f["train_events"] = f["test_events"] = LABEL                   # their sum is pMCI
    tm = dict(s.get("test_metrics") or {})
    tm.pop("prevalence", None)
    if small_test:
        tm = _drop_threshold(tm)
    s["test_metrics"] = tm
    if boot is not None:
        boot = boot[boot["metric"] != "prevalence"].copy()
        if small_test:
            boot = boot[~boot["metric"].isin(THRESHOLD_FIELDS)]
    return s, boot


# ------------------------------------------------------------------------------------------------ audit
PARTITIONS = {"history_length": ("anchors", "event_anchors"),
              "modality_overlap": ("anchors", "participants", "event_anchors", "event_participants")}


def _walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _walk(v, f"{path}.{k}")
            yield (f"{path}.{k}", k)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _walk(v, f"{path}[{i}]")
    else:
        yield (path, o)


def audit(tables: dict, jsons: dict, texts: dict) -> list:
    """Problems across all outputs ([] = clean). Never returns a suppressed value."""
    problems = []
    for name, t in tables.items():
        if not isinstance(t, pd.DataFrame):
            continue
        if any("prevalence" in str(c).lower() for c in t.columns) or (
                "metric" in t.columns and t["metric"].astype(str).str.contains("prevalence").any()):
            problems.append(f"{name}: prevalence exported")
        for c in COUNT_COLUMNS:
            if c in t.columns:
                v = pd.to_numeric(t[c], errors="coerce")
                if ((v >= 1) & (v < K)).any():
                    problems.append(f"{name}.{c}: unsuppressed small count")
        txt = t.astype(str)
        if txt.apply(lambda col: col.str.contains("<11|<\\s*11", regex=True)).any().any():
            problems.append(f"{name}: inequality label present")
        part = next((cols for key, cols in PARTITIONS.items() if name.endswith(key)), ())
        for c in part:
            if c in t.columns and int((t[c].astype(str) == LABEL).sum()) == 1:
                problems.append(f"{name}.{c}: a single suppressed cell in a partition")
    for name, o in jsons.items():
        for path, v in _walk(o):
            if "prevalence" in str(path).lower() or (isinstance(v, str) and "prevalence" in v.lower()):
                problems.append(f"{name}: prevalence exported")
            if isinstance(v, str) and ("<11" in v or "< 11" in v):
                problems.append(f"{name}: inequality label present")
            last = str(path).rsplit(".", 1)[-1]
            if last in COUNT_COLUMNS + ("pMCI", "sMCI", "samples", "train_events", "test_events", "n_train_samples",
                                        "n_test_samples", "event_participants", "nonevent_participants") and is_small(v):
                problems.append(f"{name}{path}: unsuppressed small count")
    for name, s in texts.items():
        if "<11" in s or "prevalence" in s.lower():
            problems.append(f"{name}: inequality label or prevalence in text")
    return sorted(set(problems))


def finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)
