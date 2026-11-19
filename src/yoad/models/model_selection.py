"""Prespecified, source-only comparison of Model A (logistic) and Model B (XGBoost).

RECOMMENDED RULE (paired conservative replacement; to be locked before any target outcome is examined):
  d_i = Brier_A,i - Brier_B,i   (participant-level; positive = B better), averaged over R repeated grouped-CV runs
  select B  iff  mean(d) > c * SE(d)  AND  B is better in at least `stability_min` of the R repeats;
  otherwise keep A (the prespecified simpler default). If |mean(d)| <= c*SE(d) the result is reported as
  NO_STABLE_WINNER: A is retained by default, NOT because it was shown to be better.
The conventional one-standard-error rule is implemented separately (`one_se_rule`) and is NOT the same rule.
SE caveat: participant-level SEs of cross-validated losses ignore the dependence induced by overlapping training sets
and are therefore optimistic (no unbiased variance estimator of K-fold CV exists); c = 1 is used on top of that.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from yoad.evaluation.safeguards import group_kfold
from yoad.models.frozen_pipeline import ModelSpec, fit_source, TargetLeakageError


@dataclass(frozen=True)
class SelectionRule:
    name: str = "paired_conservative_replacement"
    metric: str = "brier"
    c: float = 1.0
    repeats: int = 10
    outer_folds: int = 5
    stability_min: float = 0.8
    default_model: str = "A_logistic"
    challenger: str = "B_xgboost"
    seed: int = 2026


def oof_probabilities(spec: ModelSpec, X: pd.DataFrame, y: np.ndarray, ids: pd.Series, rule: SelectionRule) -> tuple[np.ndarray, pd.Index]:
    """(repeats, n_participants) out-of-fold probabilities, one row per participant label (first row of a participant's rows is
    averaged). Tuning/preprocessing/calibration are re-done inside each outer training set by fit_source (nested)."""
    ids = pd.Series(np.asarray(ids)).reset_index(drop=True); y = np.asarray(y); X = X.reset_index(drop=True)
    participants = pd.Index(sorted(ids.unique()))
    out = np.full((rule.repeats, len(participants)), np.nan)
    for r in range(rule.repeats):
        p = np.full(len(y), np.nan)
        for tr, te in group_kfold(ids, pd.Series(y), k=rule.outer_folds, seed=rule.seed + r):
            m = fit_source(spec, X.iloc[tr], y[tr], ids.iloc[tr], version=f"sel-{r}", cohort_role="source")
            p[te] = m.predict_proba(X.iloc[te])
        out[r] = pd.Series(p).groupby(ids.values).mean().reindex(participants).to_numpy()
    return out, participants


def oof_brier_by_participant(spec: ModelSpec, X: pd.DataFrame, y: np.ndarray, ids: pd.Series, rule: SelectionRule) -> np.ndarray:
    """(repeats, n_participants) out-of-fold Brier per participant (participant-level pairing across models)."""
    P, participants = oof_probabilities(spec, X, y, ids, rule)
    ids = pd.Series(np.asarray(ids)).reset_index(drop=True)
    ylab = pd.Series(np.asarray(y)).groupby(ids.values).max().reindex(participants).to_numpy()
    return (P - ylab[None, :]) ** 2


def paired_replacement_rule(loss_a: np.ndarray, loss_b: np.ndarray, rule: SelectionRule) -> dict:
    """loss_*: (repeats, participants). Pure function of source-only out-of-fold losses."""
    d_rep = loss_a - loss_b                       # positive: B better
    d_i = d_rep.mean(axis=0)                      # per participant, averaged over repeats
    n = d_i.size; mean_d = float(d_i.mean()); se = float(d_i.std(ddof=1) / np.sqrt(n))
    stab_b = float((d_rep.mean(axis=1) > 0).mean()); stab_a = float((d_rep.mean(axis=1) < 0).mean())
    ci = (mean_d - 1.96 * se, mean_d + 1.96 * se)
    if mean_d > rule.c * se and stab_b >= rule.stability_min:
        sel, status = rule.challenger, "CHALLENGER_SELECTED"
    elif mean_d < -rule.c * se and stab_a >= rule.stability_min:
        sel, status = rule.default_model, "DEFAULT_SUPPORTED"
    else:
        sel, status = rule.default_model, "NO_STABLE_WINNER_DEFAULT_RETAINED"
    return {"rule": rule.name, "selected": sel, "status": status, "mean_brier_A": float(loss_a.mean()), "mean_brier_B": float(loss_b.mean()),
            "mean_diff_A_minus_B": mean_d, "se_diff": se, "ci95_diff": ci, "z": mean_d / se if se > 0 else float("inf"),
            "stability_B_better": stab_b, "n_participants": int(n), "repeats": int(loss_a.shape[0]),
            "se_caveat": "participant-level SE ignores CV training-set overlap; optimistic", "uncertainty_reported": True}


def one_se_rule(scores: dict, ses: dict, complexity_order: list[str]) -> dict:
    """Conventional one-standard-error principle (lower score = better): among candidates whose score is within one SE of the
    BEST score, choose the simplest. Differs from the paired replacement rule: it uses the best model's own SE, not the SE of a paired difference."""
    best = min(scores, key=scores.get); thresh = scores[best] + ses[best]
    ok = [m for m in complexity_order if scores[m] <= thresh]
    return {"rule": "one_se", "best": best, "threshold": float(thresh), "selected": ok[0]}


def compare_models(spec_a: ModelSpec, spec_b: ModelSpec, X_src: pd.DataFrame, y_src, ids_src, rule: SelectionRule | None = None,
                   cohort_role: str = "source"):
    """Run the prespecified comparison and also return the out-of-fold probabilities (so metrics need not be recomputed)."""
    if cohort_role != "source":
        raise TargetLeakageError("model selection must use source data only")
    rule = rule or SelectionRule()
    if spec_a.features != spec_b.features:
        raise ValueError("candidates must use the identical prespecified feature list")
    Pa, participants = oof_probabilities(spec_a, X_src, y_src, ids_src, rule)
    Pb, _ = oof_probabilities(spec_b, X_src, y_src, ids_src, rule)
    ids = pd.Series(np.asarray(ids_src)).reset_index(drop=True)
    ylab = pd.Series(np.asarray(y_src)).groupby(ids.values).max().reindex(participants).to_numpy()
    la, lb = (Pa - ylab[None, :]) ** 2, (Pb - ylab[None, :]) ** 2
    res = paired_replacement_rule(la, lb, rule); n = la.shape[1]
    res["one_se_comparison"] = one_se_rule(
        {"A_logistic": float(la.mean()), "B_xgboost": float(lb.mean())},
        {"A_logistic": float(la.mean(axis=0).std(ddof=1) / np.sqrt(n)), "B_xgboost": float(lb.mean(axis=0).std(ddof=1) / np.sqrt(n))},
        ["A_logistic", "B_xgboost"])
    return res, {"A_logistic": Pa, "B_xgboost": Pb}, participants, ylab


def select_model(spec_a: ModelSpec, spec_b: ModelSpec, X_src: pd.DataFrame, y_src, ids_src, rule: SelectionRule | None = None,
                 cohort_role: str = "source") -> dict:
    return compare_models(spec_a, spec_b, X_src, y_src, ids_src, rule, cohort_role)[0]
