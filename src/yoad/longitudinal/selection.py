"""Prespecified selection rule (PROTOCOL.md §8). Pure function of the aggregate candidate and contrast tables; no data access.

Step 1  eligibility of each neural candidate (B, C, D, E): completed every outer fold; calibration slope in [0.8, 1.2] and
        |log O:E| <= ln 1.10 (frozen benchmark constants) on the RECALIBRATED output (inner-OOF Platt, applied identically
        to every candidate); seed SD of balanced accuracy <= 0.02; outer-fold SD of balanced accuracy <= 0.06. Balanced
        accuracy is anchor-weighted and uses the common inner-OOF Youden threshold.
Step 2  improvement over the matched baseline A_SIMKHADA on the primary evaluation set (all anchors), identical anchors and
        folds: paired 95% cluster-bootstrap CI of the balanced-accuracy difference wholly above 0 AND lower CI of the AUROC
        difference above -0.02.
Step 3  among improving eligible candidates: highest balanced accuracy; candidates within one paired SE of it -> fewest
        trainable parameters. -> Outcome N1 (a neural model improves on the Simkhada-compatible baseline).
        No improving candidate but >= 1 eligible -> Outcome N2: the best eligible neural candidate (same tie rule) is reported
        as the final NEURAL candidate, explicitly not superior to the baseline.
        No eligible candidate -> Outcome N3 (no neural candidate is supported).
Step 4  overall best supported model: among all calibration-adequate (recalibrated output) candidates (baselines included; E_SNAP and AVAIL
        excluded), the SIMPLEST in COMPLEXITY_ORDER whose balanced accuracy is within one paired SE of the highest. If the
        highest is a neural candidate that improved (N1), it is the overall model only if no simpler candidate lies within one
        SE of it.
A higher point estimate alone never selects a model, and architectural complexity is never a reason to prefer one.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from yoad.longitudinal import config as CF

SEL = CF.SELECTION


def _ok_cal(row) -> bool:
    s, oe = row.get("calibration_slope"), row.get("log_oe")
    return (CF.finite(s) and SEL["cal_slope_range"][0] <= s <= SEL["cal_slope_range"][1]
            and CF.finite(oe) and abs(oe) <= SEL["abs_log_oe_max"])


def _contrast(contrasts: pd.DataFrame, a: str, b: str, metric: str) -> dict | None:
    sel = lambda x, z: contrasts[(contrasts["candidate"] == x) & (contrasts["comparator"] == z)
                                 & (contrasts["metric"] == metric) & (contrasts["eval_set"] == CF.PRIMARY_EVAL)]
    c = sel(a, b)
    if not c.empty:
        return c.iloc[0].to_dict()
    c = sel(b, a)                                       # the reverse contrast: negate and swap the interval
    if c.empty:
        return None
    d = c.iloc[0].to_dict()
    return d | {"candidate": a, "comparator": b, "difference": -d["difference"], "ci_low": -d["ci_high"],
                "ci_high": -d["ci_low"]}


def _best_with_tie(ids, results: dict, contrasts: pd.DataFrame, order_key) -> str:
    best = max(ids, key=lambda c: results[c]["balanced_accuracy"])
    tied = []
    for c in ids:
        if c == best:
            tied.append(c); continue
        ct = _contrast(contrasts, best, c, "balanced_accuracy")
        se = ct["se"] if ct else np.nan
        if CF.finite(se) and results[best]["balanced_accuracy"] - results[c]["balanced_accuracy"] <= se:
            tied.append(c)
    return min(tied, key=order_key)


def select(results: dict, contrasts: pd.DataFrame, n_params: dict, seed_sd: dict, fold_sd: dict, complete: dict) -> dict:
    """results: {cid: metric dict on the primary evaluation set}; contrasts: paired table (candidate, comparator, metric,
    eval_set, difference, ci_low, ci_high, se)."""
    rows, eligible, improving = [], [], []
    for c in CF.SELECTABLE_NEURAL:
        r = results.get(c)
        reasons = []
        if r is None or not complete.get(c, False):
            reasons.append("did not complete every outer fold")
        else:
            if not _ok_cal(r):
                reasons.append("calibration outside the frozen range")
            if not (CF.finite(seed_sd.get(c)) and seed_sd[c] <= SEL["max_seed_sd_balanced_accuracy"]):
                reasons.append("seed SD of balanced accuracy above 0.02")
            if not (CF.finite(fold_sd.get(c)) and fold_sd[c] <= SEL["max_fold_sd_balanced_accuracy"]):
                reasons.append("fold SD of balanced accuracy above 0.06")
        ba = _contrast(contrasts, c, CF.MATCHED_BASELINE, "balanced_accuracy") if r else None
        au = _contrast(contrasts, c, CF.MATCHED_BASELINE, "roc_auc") if r else None
        improves = bool(ba and au and CF.finite(ba["ci_low"]) and ba["ci_low"] > 0
                        and CF.finite(au["ci_low"]) and au["ci_low"] > -SEL["auroc_noninferiority_margin"])
        if not reasons:
            eligible.append(c)
            if improves:
                improving.append(c)
        rows.append({"candidate": c, "eligible": not reasons, "ineligible_reasons": "; ".join(reasons),
                     "improves_on_matched_baseline": improves, "n_params": n_params.get(c),
                     "balanced_accuracy": r.get("balanced_accuracy") if r else None,
                     "delta_ba_vs_baseline": ba["difference"] if ba else None,
                     "delta_ba_ci": [ba["ci_low"], ba["ci_high"]] if ba else None,
                     "delta_auroc_vs_baseline": au["difference"] if au else None,
                     "delta_auroc_ci": [au["ci_low"], au["ci_high"]] if au else None})
    by_params = lambda c: (n_params.get(c) or math.inf, CF.COMPLEXITY_ORDER.index(c))
    if improving:
        outcome, neural = "N1", _best_with_tie(improving, results, contrasts, by_params)
        text = "a neural candidate improves balanced accuracy over the Simkhada-compatible baseline on identical anchors"
    elif eligible:
        outcome, neural = "N2", _best_with_tie(eligible, results, contrasts, by_params)
        text = "no neural candidate improves on the Simkhada-compatible baseline; best eligible neural candidate reported"
    else:
        outcome, neural = "N3", None
        text = "no neural candidate meets the eligibility rules"
    pool = [c for c in CF.COMPLEXITY_ORDER if c in results and complete.get(c, False) and _ok_cal(results[c])
            and (c not in CF.SELECTABLE_NEURAL or c in eligible)]
    overall = _best_with_tie(pool, results, contrasts, CF.COMPLEXITY_ORDER.index) if pool else None
    return {"outcome": outcome, "outcome_text": text, "final_neural_candidate": neural, "best_supported_model": overall,
            "candidates": rows, "overall_pool": pool, "rule": SEL}
