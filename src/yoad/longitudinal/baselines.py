"""Non-neural candidates (PROTOCOL.md §5.1). The estimators are frozen, reused ones (REUSE_MAP.md §1); this file feeds
anchor-level tables to them with PARTICIPANT-grouped inner splits, applies the common threshold / recalibration rule and checks
the fitted rows. No file or data access.

  LR_REF            multimodal_benchmark.models.build("lr_l2_ref", C = 1) on CORE7 (anchor snapshot, strict window)
  A0_XGB_INCUMBENT  build("xgb_ref", 200/2/0.05) on FS16_CORE7_COGX (anchor snapshot, strict window)
  A1_XGB_MATCHED    build("xgb_ref") on sequences.snapshot_frame (per-block last observations, staleness, availability)
  AVAIL             build("lr_l2_ref") on per-block availability and assessment counts only (shortcut control)
  A_SIMKHADA        adapted Simkhada-compatible pipeline: experiments.simkhada_baseline.baseline_lib preprocessing
                    (MinObservedFilter -> scaled KNN -> variance and correlation filters; most-frequent categorical
                    imputation) -> SMOTENC -> SelectKBest -> XGBoost; randomized search refit on balanced accuracy over
                    grouped 5-fold CV; grouped Platt calibration. The selector restores the pinned reference (commit c67358d,
                    search_config.py / final/method.py): step "select" AFTER resampling, before the classifier, on all
                    transformed columns; k in {20, ..., 150}, score f_classif or mutual_info_classif(random_state = 11).
Feature availability is decided INSIDE every fit: for the frozen estimators by `select_columns` on each inner-training split
and then on the full outer-training data; for A_SIMKHADA by the MinObservedFilter pipeline step, which every search split,
calibration split and threshold split refits. Inner-validation and outer-test anchors never decide which columns are active.
Common rule (every candidate, neural ones in evaluate.py): threshold = Youden on out-of-fold predictions over the SAME three
grouped inner splits; recalibrated output = Platt (a picklable `Platt`) fitted on those inner out-of-fold predictions.
Differences from Simkhada's own threshold procedure: they used Youden on grouped 5-fold out-of-fold calibrated predictions with
seed 11; here the 3 common inner splits are used so that every candidate is thresholded identically.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import StratifiedGroupKFold

from yoad.multimodal import models as MM
from yoad.multimodal import registry as R
from yoad.multimodal.metrics import youden
from yoad.evaluation.safeguards import LeakageError
from yoad.longitudinal import config as CF

FIXED = {"lr_l2_ref": {"C": 1.0}, "xgb_ref": {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.05}}
CORE7 = tuple(R.resolve("FS08_CORE7"))
FS16 = tuple(R.resolve("FS16_CORE7_COGX"))
FEATURE_BLOCK = {f: b for b, fs in CF.TV_BLOCKS.items() for f in fs}


def inner_splits(y, groups, k: int, seed: int) -> list:
    """StratifiedGroupKFold over anchors grouped by participant (no participant in both parts)."""
    sp = list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))
    for tr, va in sp:
        if set(np.asarray(groups)[tr]) & set(np.asarray(groups)[va]):
            raise LeakageError("participant in both inner training and validation splits")
    return sp


# ------------------------------------------------------------------------------------------------ calibration (persistable)
def _logit(p):
    q = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)                # = src.benchmark.metrics._logit
    return np.log(q / (1 - q))


class Platt:
    """Platt recalibration p -> expit(a * logit(p) + b), fitted like src.benchmark.metrics.fit_calibrator('sigmoid') but
    picklable, so a saved model can reproduce its recalibrated probabilities exactly."""

    def fit(self, y, p, seed: int) -> "Platt":
        from sklearn.linear_model import LogisticRegression
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2:
            raise ValueError("calibration requires both classes in the (training) out-of-fold predictions")
        lr = LogisticRegression(C=1e6, max_iter=5000, random_state=seed).fit(_logit(p).reshape(-1, 1), y)
        self.a_, self.b_ = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def __call__(self, p):
        p = np.asarray(p, float)
        return expit(self.a_ * _logit(p) + self.b_) if len(p) else np.zeros(0)

    def params(self) -> dict:
        return {"a": self.a_, "b": self.b_}


def common_rule(y_tr, oof, p_test, seed) -> dict:
    """Youden threshold and inner-OOF Platt recalibration (training anchors only)."""
    cal = Platt().fit(y_tr, oof, seed)
    return {"threshold": float(youden(y_tr, oof)), "p_test_recal": cal(p_test), "threshold_recal": float(youden(y_tr, cal(oof))),
            "calibration": cal.params(), "_calibrator": cal}


def _check(est, allowed, forbidden):
    rows = MM.fitted_rows(est)
    if rows & frozenset(forbidden):
        raise LeakageError("baseline fitted on outer-test anchors")
    if rows != frozenset(allowed):
        raise LeakageError("baseline not fitted on exactly the intended anchors")


# ------------------------------------------------------------------------------------------------ in-fit feature availability
def column_block(col: str) -> str | None:
    """Block owning a snapshot / availability column (None for non-block columns such as CORE7)."""
    if col in FEATURE_BLOCK:
        return FEATURE_BLOCK[col]
    if col.endswith("_staleness_years"):
        return FEATURE_BLOCK.get(col[: -len("_staleness_years")])
    for prefix in ("has_", "n_"):
        if col.startswith(prefix):
            b = col[len(prefix):].replace("_at_anchor", "")
            if b in CF.TV_BLOCKS:
                return b
    return None


def select_columns(X: pd.DataFrame, groups) -> list:
    """Columns permitted when fitting on exactly these rows: a value column needs >= MIN_FEATURE_TRAIN_OBS non-missing values;
    a time-varying block keeps its staleness / availability / count columns only if it is ENABLED (>= MIN_BRANCH_TRAIN_
    PARTICIPANTS participants with a permitted value, or with an assessment when the table has no value column of the block);
    a disabled block loses every one of its columns. Same rule as sequences.Preprocessor."""
    groups = np.asarray(groups)
    nn = X.notna().sum()
    blocks = {}
    for c in X.columns:
        blocks.setdefault(column_block(c), []).append(c)
    out = []
    for b, cols in blocks.items():
        value = [c for c in cols if b is None or c in CF.TV_BLOCKS[b]]
        kept = [c for c in value if nn[c] >= CF.MIN_FEATURE_TRAIN_OBS]
        if b is None:
            out += kept; continue
        aux = [c for c in cols if c not in value]
        if value:
            seen = X[kept].notna().any(axis=1).to_numpy() if kept else np.zeros(len(X), bool)
        else:
            has = f"has_{b}"
            seen = (X[has].fillna(0).to_numpy() > 0) if has in X.columns else np.zeros(len(X), bool)
        if (kept or not value) and len(set(groups[seen])) >= CF.MIN_BRANCH_TRAIN_PARTICIPANTS:
            out += kept + aux
    return [c for c in X.columns if c in set(out)]


def frozen_task(model: str, X: pd.DataFrame, y, groups, Xte: pd.DataFrame, seed: int, inner_seed: int) -> dict:
    """model in {lr_l2_ref, xgb_ref}; X / Xte indexed by unique anchor ids. Column availability is decided on each
    inner-training split (applied to its inner-validation rows), then on the full outer-training data for the final model."""
    groups = np.asarray(groups)
    oof = np.full(len(y), np.nan)
    inner_cols = []
    for itr, iva in inner_splits(y, groups, CF.INNER_SPLITS, inner_seed):
        cols = select_columns(X.iloc[itr], groups[itr])
        est = MM.build(model, FIXED[model], cols, seed).fit(X.iloc[itr][cols], y[itr])
        _check(est, X.index[itr], X.index[iva])
        oof[iva] = est.predict_proba(X.iloc[iva][cols])[:, 1]
        inner_cols.append(cols)
    cols = select_columns(X, groups)
    final = MM.build(model, FIXED[model], cols, seed).fit(X[cols], y)
    _check(final, X.index, Xte.index)
    p = final.predict_proba(Xte[cols])[:, 1] if len(Xte) else np.zeros(0)
    return {"p_test": p, **common_rule(y, oof, p, seed), "params": FIXED[model] | {"features_used": len(cols)},
            "n_params": len(cols) + 1 if model == "lr_l2_ref" else None, "columns": cols, "inner_columns": inner_cols,
            "_model": final}


# ------------------------------------------------------------------------------------------------ Simkhada-compatible
class MinObservedFilter(BaseEstimator, TransformerMixin):
    """First step of the numeric branch: keeps the columns with >= min_obs non-missing values IN THE ROWS IT IS FITTED ON.
    Refitted by every search, calibration and threshold split, so held-out rows never decide which columns are active."""

    def __init__(self, min_obs: int = CF.MIN_FEATURE_TRAIN_OBS):
        self.min_obs = min_obs

    def fit(self, X, y=None):
        A = np.asarray(X, float)
        self.keep_ = np.isfinite(A).sum(axis=0) >= self.min_obs
        if not self.keep_.any():
            raise ValueError("no numeric column has enough observed training values")
        return self

    def transform(self, X):
        return np.asarray(X, float)[:, self.keep_]


def simkhada_columns(frame: pd.DataFrame) -> tuple[list, list]:
    num = [c for c in CF.SIMKHADA_FEATURES if c != "female" and c in frame.columns]
    num += sorted(c for c in frame.columns if c.startswith("fs_"))
    return num, ["female"]


def simkhada_pipeline(num, cat, seed):
    """baseline_lib pipeline with (i) MinObservedFilter as the first numeric step and (ii) the classifier replaced by
    [SelectKBest -> XGBoost]; the resampling wrapper applies SMOTENC first, so the order is preprocessing -> SMOTENC ->
    select -> classifier, as in the pinned reference."""
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.pipeline import Pipeline
    from yoad.experiments.simkhada_baseline import baseline_lib as bl
    pipe = bl.make_pipeline("xgboost", "smotenc", num, cat, seed)
    prep = pipe.named_steps["prep"]
    name, numeric, cols = prep.transformers[0]
    numeric = Pipeline([("observed", MinObservedFilter())] + list(numeric.steps))
    prep.set_params(transformers=[(name, numeric, cols)] + list(prep.transformers[1:]))
    inner = Pipeline([("select", SelectKBest(score_func=f_classif, k=CF.SIMKHADA_SELECTOR["default_k"])),
                      ("clf", bl.make_estimator("xgboost", seed))])
    pipe.set_params(model__estimator=inner)
    return pipe


def simkhada_space() -> dict:
    from sklearn.feature_selection import f_classif, mutual_info_classif
    from yoad.experiments.simkhada_baseline import baseline_lib as bl
    sp = {}
    for k, v in bl.search_space("xgboost", "smotenc").items():
        sp[k.replace("model__estimator__", "model__estimator__clf__", 1) if k.startswith("model__estimator__") else k] = v
    sp["model__estimator__select__k"] = list(CF.SIMKHADA_SELECTOR["k"])
    sp["model__estimator__select__score_func"] = [f_classif, partial(mutual_info_classif,
                                                                     random_state=CF.SIMKHADA_SELECTOR["mi_random_state"])]
    return sp


def simkhada_task(X: pd.DataFrame, y, groups, Xte: pd.DataFrame, seed: int, inner_seed: int) -> dict:
    import warnings
    from yoad.experiments.simkhada_baseline import baseline_lib as bl
    num, cat = simkhada_columns(X)
    X, Xte = X[num + cat], Xte[num + cat]
    groups = np.asarray(groups)
    cv = bl.sgkf(y, groups, CF.SIMKHADA_SEARCH["inner_splits"], seed)
    with warnings.catch_warnings():                       # k above the number of filtered columns -> all columns (sklearn)
        warnings.filterwarnings("ignore", message=".*k=.*greater than n_features.*")
        s, failed = bl.run_search(simkhada_pipeline(num, cat, seed), simkhada_space(), X, y, groups,
                                  CF.SIMKHADA_SEARCH["n_iter"], cv, seed)
        chosen = clone(s.best_estimator_)
        oof = np.full(len(y), np.nan)
        for itr, iva in inner_splits(y, groups, CF.INNER_SPLITS, inner_seed):
            c = bl.grouped_calibrated(chosen, X.iloc[itr], y[itr], groups[itr], seed)
            if frozenset(bl.fitted_rows(c)) & frozenset(X.index[iva]):
                raise LeakageError("inner validation anchors used for fitting")
            oof[iva] = c.predict_proba(X.iloc[iva])[:, 1]
        cal = bl.grouped_calibrated(chosen, X, y, groups, seed)
    rows = frozenset(bl.fitted_rows(cal))
    if rows & frozenset(Xte.index) or not rows <= frozenset(X.index):
        raise LeakageError("Simkhada-compatible pipeline touched outer-test anchors")
    p = cal.predict_proba(Xte)[:, 1] if len(Xte) else np.zeros(0)
    return {"p_test": p, **common_rule(y, oof, p, seed), "search_failed": int(failed),
            "params": {k: (v.func.__name__ if hasattr(v, "func") else getattr(v, "__name__", repr(v)))
                       for k, v in s.best_params_.items()}, "n_params": None, "_model": cal}
