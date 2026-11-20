"""Prespecified winner-selection rule (PROTOCOL.md §8), applied to development-partition nested-CV results only.

Pool: completed candidates of one population (P_ALL primary, P_MRI secondary) from stages A and B with median imputation and
primary timing; every outer fold must have succeeded. Steps:
  1. b = the candidate with the lowest mean out-of-fold Brier score (repeat-averaged).
  2. c is COMPARABLE when Brier_c - Brier_b <= 1.0 x SE(Brier_c - Brier_b) (paired participant bootstrap; the
     one-standard-error rule applied to the paired difference) AND AUROC_c - AUROC_b >= -0.02 (the frozen AUROC materiality
     margin, protocol §15 item 1), so a simpler model cannot be chosen at a material loss of discrimination.
  3. c has ADEQUATE calibration when its calibration slope lies in [0.8, 1.2] and |log O:E| <= ln(1.10).
  4. Winner = the least complex comparable and adequately calibrated candidate, complexity ordered by (number of non-
     demographic instruments required, model rank in registry.MODELS, number of features, id). Fallbacks: the lowest-Brier
     adequately calibrated candidate; otherwise b, flagged 'calibration inadequate'.
The rule never looks at the held-out partition. Its output is frozen (hash recorded) before the held-out stage runs.
"""
from __future__ import annotations

import math

from yoad.multimodal import registry as R


def data_burden(features) -> int:
    inst = set()
    for f in features:
        inst.add("structural MRI (FreeSurfer 7)" if f.startswith("fs_") else R.INSTRUMENTS.get(f, f))
    return len(inst - {"demographic"})


def order_key(rec: dict) -> tuple:
    return (data_burden(rec["features"]), R.MODELS[rec["model"]]["rank"], len(rec["features"]), rec["id"])


def adequate(rec: dict) -> bool:
    lo, hi = R.SELECTION["cal_slope_range"]
    s, oe = rec.get("calibration_slope"), rec.get("log_oe")
    return (s is not None and math.isfinite(s) and lo <= s <= hi and oe is not None and math.isfinite(oe)
            and abs(oe) <= R.SELECTION["abs_log_oe_max"])


def select(pool: list[dict], vs_best) -> dict:
    """pool: [{id, model, features, brier, roc_auc, calibration_slope, log_oe, complete}];
    vs_best(best_id, id) -> (delta_brier, se_delta_brier, delta_auroc), all paired on common participants.
    Returns {'winner', 'best_brier', 'reason', 'table': [...]} (the table records every decision input)."""
    pool = [p for p in pool if p.get("complete") and p.get("brier") is not None and math.isfinite(p["brier"])]
    if not pool:
        return {"winner": None, "best_brier": None, "reason": "no completed candidate", "table": []}
    best = min(pool, key=lambda p: (p["brier"], order_key(p)))
    table = []
    for p in pool:
        delta, se, d_auc = (0.0, 0.0, 0.0) if p["id"] == best["id"] else vs_best(best["id"], p["id"])
        comparable = p["id"] == best["id"] or (math.isfinite(delta) and math.isfinite(se) and math.isfinite(d_auc)
                                                and delta <= R.SELECTION["one_se_multiplier"] * se
                                                and d_auc >= -R.SELECTION["max_auroc_loss"])
        table.append({"id": p["id"], "brier": p["brier"], "delta_brier_vs_best": delta, "se_delta": se,
                      "delta_auroc_vs_best": d_auc, "roc_auc": p.get("roc_auc"),
                      "comparable": bool(comparable), "calibration_adequate": adequate(p),
                      "calibration_slope": p.get("calibration_slope"), "log_oe": p.get("log_oe"),
                      "data_burden": data_burden(p["features"]), "model_rank": R.MODELS[p["model"]]["rank"],
                      "n_features": len(p["features"]), "order_key": str(order_key(p))})
    recs = {p["id"]: p for p in pool}
    ok = [t for t in table if t["comparable"] and t["calibration_adequate"]]
    if ok:
        win = min(ok, key=lambda t: order_key(recs[t["id"]]))
        reason = "least complex candidate within one paired SE of the lowest Brier score, with adequate calibration"
    else:
        adeq = [t for t in table if t["calibration_adequate"]]
        if adeq:
            win = min(adeq, key=lambda t: (t["brier"], order_key(recs[t["id"]])))
            reason = "no comparable candidate was adequately calibrated; lowest-Brier adequately calibrated candidate"
        else:
            win = next(t for t in table if t["id"] == best["id"])
            reason = "no candidate met the calibration-adequacy condition; lowest-Brier candidate, flagged calibration inadequate"
    for t in table:
        t["selected"] = t["id"] == win["id"]
    return {"winner": win["id"], "best_brier": best["id"], "reason": reason, "table": sorted(table, key=lambda t: t["brier"])}
