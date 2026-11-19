"""Leakage / split safeguards."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

DEFAULT_FORBIDDEN = [r"conver", r"event", r"time_to", r"followup", r"censor", r"label", r"outcome", r"last_", r"future", r"_m(12|24|36)\b", r"(^|_)dx_?(m|mo|month)?_?\d+", r"diagnos\w*_?(m|mo|month)?_?\d+", r"months?_to_"]


class LeakageError(RuntimeError):
    pass


def check_feature_names(features: list[str], outcome_cols: list[str], forbidden=DEFAULT_FORBIDDEN) -> list[str]:
    """Return suspicious feature names (outcome columns present, or names matching post-baseline patterns)."""
    hits = [f for f in features if f in outcome_cols]
    hits += [f for f in features if f not in hits and any(re.search(p, f, re.I) for p in forbidden)]
    return hits


def check_feature_dates(feature_dates: pd.DataFrame, base: pd.DataFrame, date_col: str = "visit_date", tol_days: int = 0) -> pd.DataFrame:
    """Rows whose measurement date is after baseline_date + tol_days (post-baseline info used as predictor)."""
    m = feature_dates.merge(base, on="participant_id")
    return m[(m[date_col] - m["baseline_date"]).dt.days > tol_days]


def assert_disjoint(train_ids, test_ids) -> None:
    both = set(train_ids) & set(test_ids)
    if both:
        raise LeakageError(f"{len(both)} participants in both train and test")


def group_kfold(ids: pd.Series, y: pd.Series | None = None, k: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Participant-level folds; rows with the same id always share a fold. Stratified on participant label if y is given."""
    df = pd.DataFrame({"id": ids.values, "y": (y.values if y is not None else 0)})
    per = df.groupby("id")["y"].max().reset_index()
    rng = np.random.default_rng(seed)
    fold_of = {}
    for _, grp in per.groupby("y"):
        order = grp["id"].to_numpy().copy(); rng.shuffle(order)
        for i, pid in enumerate(order):
            fold_of[pid] = i % k
    f = df["id"].map(fold_of).to_numpy()
    folds = []
    for j in range(k):
        te = np.where(f == j)[0]; tr = np.where(f != j)[0]
        assert_disjoint(df["id"].iloc[tr], df["id"].iloc[te])
        folds.append((tr, te))
    return folds


def fit_on_train_only(transform_fit, X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fit preprocessing on train; apply to both. transform_fit(X_train) -> callable."""
    f = transform_fit(X_train)
    return f(X_train), f(X_test)


def flag_suspicious_predictors(X: pd.DataFrame, y, followup=None, auc_thresh: float = 0.95, corr_thresh: float = 0.98) -> dict:
    """Value-based leakage screen for synthetic/authorised use: single predictors that (almost) determine the outcome, or that are
    (almost) a copy of follow-up duration. Names alone are not enough (e.g. a renamed follow-up column)."""
    from scipy.stats import rankdata
    y = np.asarray(y, float); ok = ~np.isnan(y); out = {"outcome_proxy": [], "followup_proxy": []}
    for c in X.columns:
        v = X[c].astype(float).to_numpy(); m = ok & ~np.isnan(v)
        if m.sum() > 10 and 0 < y[m].sum() < m.sum():
            r = rankdata(v[m]); n1 = y[m].sum(); n0 = m.sum() - n1
            auc = (r[y[m] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
            if max(auc, 1 - auc) >= auc_thresh: out["outcome_proxy"].append(c)
        if followup is not None:
            f = np.asarray(followup, float); mm = ~np.isnan(v) & ~np.isnan(f)
            if mm.sum() > 10 and np.std(v[mm]) > 0 and abs(np.corrcoef(v[mm], f[mm])[0, 1]) >= corr_thresh: out["followup_proxy"].append(c)
    return out
