"""ADNI-developed, NACC-transportable model and its independent validation in NACC. Pure functions; no file access.
Synthetic-tested in tests/test_nacc_validation.py; the guarded real-data entry point is run_nacc_validation.py (researcher only).

Design (research/source_target_variable_mapping.md, 2026-10-19; owner sign-off pending on the items marked [S])
  development  ADNI only: every MCI visit, calendar three-state labels with +/- window_days [S], all-cause dementia, same-visit
               predictors, complete case; the reference training protocol of experiments/simkhada_baseline/baseline_lib
               (randomized search, out-of-fold Youden threshold, grouped Platt calibration) fitted on ALL ADNI samples
  predictors   NACC_T = MMSE, CDR global, CDR-SB, FAQ, sex, age, handedness, education, APOE e4 (no ADAS, no MRI: absent in NACC);
               NACC_T_noMMSE drops MMSE (keeps MoCA-era NACC visits) [S]
  validation   the frozen ADNI model (preprocessing, calibration and threshold unchanged) applied once to NACC; NACC rows are
               never used for fitting (checked); primary unit = every eligible MCI visit with participant-cluster bootstrap CIs;
               secondary = each participant's first eligible visit; no recalibration in the primary results
  metrics      ROC-AUC, PR-AUC (with prevalence), Brier, calibration intercept and slope, log O:E, and at the transferred
               threshold balanced accuracy, sensitivity, specificity, PPV, NPV; 95% percentile intervals
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from yoad.experiments.simkhada_baseline import baseline_lib as bl

TRANSPORT_SETS = {
    "NACC_T": ["MMSCORE", "CDGLOBAL", "CDRSB", "FAQTOTAL", "AGE", "PTEDUCAT", "APOE4", "PTGENDER", "PTHAND"],
    "NACC_T_noMMSE": ["CDGLOBAL", "CDRSB", "FAQTOTAL", "AGE", "PTEDUCAT", "APOE4", "PTGENDER", "PTHAND"],
}
CATEGORICAL = ["PTGENDER", "PTHAND"]


def columns(feature_set: str) -> tuple[list, list]:
    cols = TRANSPORT_SETS[feature_set]
    return [c for c in cols if c not in CATEGORICAL], [c for c in cols if c in CATEGORICAL]


def complete_case(samples: pd.DataFrame, cols: list, flow: dict, name: str) -> pd.DataFrame:
    flow[f"{name}_labelled_samples"] = int(len(samples))
    out = samples.dropna(subset=cols).copy()
    flow[f"{name}_complete_case_samples"] = int(len(out))
    flow[f"{name}_complete_case_participants"] = int(out["participant_id"].nunique())
    flow[f"{name}_complete_case_events"] = int(out["y"].sum())
    return out


def develop(adni: pd.DataFrame, num, cat, algorithm="xgboost", sampling="smotenc", seed=bl.REFERENCE["seed"],
            n_iter=bl.REFERENCE["final_n_iter"], outer=(bl.REFERENCE["outer_splits"], bl.REFERENCE["outer_repeats"]), n_jobs=1):
    """Fit the transportable model on ALL ADNI samples with the reference protocol. Returns (calibrated model, threshold, info)."""
    X = adni[list(num) + list(cat)]; y = adni["y"].to_numpy().astype(int); g = adni["participant_id"].to_numpy()
    pipe = bl.make_pipeline(algorithm, sampling, num, cat, seed)
    search, failed = bl.run_search(pipe, bl.search_space(algorithm, sampling), X, y, g, n_iter,
                                   bl.repeated_sgkf(y, g, outer[0], outer[1], seed), seed, n_jobs)
    thr, oof = bl.oof_threshold(search.best_estimator_, X, y, g, seed)
    cal = bl.grouped_calibrated(search.best_estimator_, X, y, g, seed)
    i = search.best_index_
    info = {"algorithm": algorithm, "sampling": sampling, "threshold": float(thr), "search_draws_failed": failed,
            "best_params": {k: (v if np.isscalar(v) or v is None else str(v)) for k, v in search.best_params_.items()},
            "search_cv_best": {k: float(search.cv_results_[f"mean_test_{k}"][i]) for k in bl.SCORING},
            "adni_internal_oof": evaluate(y, oof, thr),          # out-of-fold calibrated, grouped: internal reference point
            "fitted_rows": bl.fitted_rows(cal) | set(search.best_estimator_.named_steps["fit_guard"].fitted_rows_)}
    return cal, thr, info


def evaluate(y, p, thr) -> dict:
    m = bl.point_metrics(y, p, thr)
    y = np.asarray(y, float); p = np.asarray(p, float)
    m["log_oe"] = float(np.log(y.sum() / p.sum())) if y.sum() > 0 and p.sum() > 0 else np.nan
    m["mean_predicted_risk"] = float(p.mean())
    return m


def validate(cal, thr, nacc: pd.DataFrame, num, cat, B=bl.REFERENCE["bootstrap_B"], seed=bl.REFERENCE["seed"],
             fitted_rows: set | None = None) -> dict:
    """Apply the frozen model to NACC. Primary: every eligible MCI visit; secondary: each participant's first eligible visit."""
    if fitted_rows is not None and fitted_rows & set(nacc.index):
        raise AssertionError("NACC rows were used for fitting")
    X = nacc[list(num) + list(cat)]; y = nacc["y"].to_numpy().astype(int); g = nacc["participant_id"].to_numpy()
    p = cal.predict_proba(X)[:, 1]
    first = nacc.groupby("participant_id")["anchor_month"].transform("min").eq(nacc["anchor_month"]).to_numpy()
    out = {"visit_level": evaluate(y, p, thr), "first_visit": evaluate(y[first], p[first], thr) if len(np.unique(y[first])) == 2 else None}
    ci = bl.bootstrap(y, p, thr, g, "participant", B, seed)
    oe = []
    rng = np.random.default_rng(seed); ids = np.unique(g); idx_of = {k: np.where(g == k)[0] for k in ids}
    for _ in range(B):
        idx = np.concatenate([idx_of[k] for k in rng.choice(ids, len(ids))])
        if y[idx].sum() > 0:
            oe.append(np.log(y[idx].sum() / p[idx].sum()))
    ci = pd.concat([ci, pd.DataFrame([{"unit": "participant", "metric": "log_oe", "ci_low": float(np.quantile(oe, .025)),
                                       "ci_high": float(np.quantile(oe, .975)), "bootstrap_mean": float(np.mean(oe)),
                                       "B_used": len(oe), "B_skipped_single_class": B - len(oe)}])], ignore_index=True)
    out["visit_level_ci"] = ci
    out["n_samples"] = int(len(y)); out["n_participants"] = int(len(ids)); out["n_events"] = int(y.sum())
    out["n_first_visit_samples"] = int(first.sum()); out["n_first_visit_events"] = int(y[first].sum())
    return out


def case_mix(adni: pd.DataFrame, nacc: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Aggregate side-by-side description (means, SDs, missing share, prevalence); no row-level content."""
    rows = []
    for name, d in (("ADNI", adni), ("NACC", nacc)):
        r = {"cohort": name, "samples": len(d), "participants": d["participant_id"].nunique(), "prevalence": float(d["y"].mean())}
        for c in cols:
            r[f"{c}_mean"] = float(d[c].mean()); r[f"{c}_sd"] = float(d[c].std()); r[f"{c}_missing_share"] = float(d[c].isna().mean())
        rows.append(r)
    return pd.DataFrame(rows)
