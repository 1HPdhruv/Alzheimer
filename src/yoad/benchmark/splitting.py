"""Participant/group-aware splits. Failures raise SplitError; there is never a fallback to ordinary row splits.

Stratification is on the participant-level label (1 if any of the participant's observations is positive).
Reuses `assert_disjoint` / `LeakageError` from src.evaluation.safeguards.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from yoad.evaluation.safeguards import LeakageError, assert_disjoint


class SplitError(ValueError):
    pass


def _validate(y, groups):
    y = np.asarray(y); g = pd.Series(groups)
    if len(y) != len(g):
        raise SplitError(f"labels ({len(y)}) and group ids ({len(g)}) have different lengths")
    if len(y) == 0:
        raise SplitError("no observations")
    if g.isna().any() or (g.astype(str).str.strip() == "").any():
        raise SplitError("missing group (participant) identifiers")
    if not set(np.unique(y)).issubset({0, 1}):
        raise SplitError(f"labels must be binary 0/1; found {sorted(set(np.unique(y).tolist()))[:5]}")
    per = pd.DataFrame({"g": g.values, "y": y}).groupby("g")["y"].max()
    return y, g.to_numpy(), per


def _assign(per: pd.Series, fractions: list[float], rng, stratify: bool) -> dict:
    """Assign each group to a partition index, per class if stratified, by largest-remainder rounding."""
    out = {}
    strata = per.groupby(per) if stratify else [(0, per)]
    for _, s in strata:
        ids = s.index.to_numpy().copy(); rng.shuffle(ids)
        n = len(ids)
        counts = np.floor(np.array(fractions) * n).astype(int)
        for j in np.argsort(-(np.array(fractions) * n - counts))[: n - counts.sum()]:
            counts[j] += 1
        k = 0
        for part, c in enumerate(counts):
            for gid in ids[k:k + c]:
                out[gid] = part
            k += c
    return out


def group_partition(y, groups, fractions: dict, seed: int, stratify: bool = True, require_both_classes=()) -> dict:
    """Split groups into named partitions (e.g. {'train': .6, 'validation': .2, 'test': .2}).

    Returns {name: row-index array}. Raises SplitError if a partition would have no groups, or if a partition named in
    `require_both_classes` would lack a class.
    """
    y, g, per = _validate(y, groups)
    names = list(fractions); fr = [float(fractions[n]) for n in names]
    if any(f <= 0 for f in fr) or abs(sum(fr) - 1) > 1e-9:
        raise SplitError(f"partition fractions must be positive and sum to 1; got {fractions}")
    if stratify:
        for cls, cnt in per.value_counts().items():
            if cnt < len(names):
                raise SplitError(f"class {cls} has {cnt} participant(s); at least {len(names)} needed for a stratified split into {names}")
    elif per.size < len(names):
        raise SplitError("fewer participants than partitions")
    assign = _assign(per, fr, np.random.default_rng(seed), stratify)
    part = pd.Series(g).map(assign).to_numpy()
    out = {n: np.where(part == i)[0] for i, n in enumerate(names)}
    for n, idx in out.items():
        if idx.size == 0:
            raise SplitError(f"partition {n!r} is empty")
        if n in require_both_classes and len(np.unique(y[idx])) < 2:
            raise SplitError(f"partition {n!r} contains a single class")
    check_disjoint(out, g)
    return out


def group_kfold_splits(y, groups, n_splits: int, seed: int, stratify: bool = True) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped (optionally stratified) k-fold; every group is in exactly one test fold; every test fold has both classes."""
    y, g, per = _validate(y, groups)
    if not isinstance(n_splits, int) or n_splits < 2:
        raise SplitError("n_splits must be an integer >= 2")
    if stratify:
        for cls, cnt in per.value_counts().items():
            if cnt < n_splits:
                raise SplitError(f"class {cls} has {cnt} participant(s); stratified {n_splits}-fold needs at least {n_splits}")
    elif per.size < n_splits:
        raise SplitError("fewer participants than folds")
    assign = _assign(per, [1.0 / n_splits] * n_splits, np.random.default_rng(seed), stratify)
    fold = pd.Series(g).map(assign).to_numpy()
    out = []
    for k in range(n_splits):
        te = np.where(fold == k)[0]; tr = np.where(fold != k)[0]
        if te.size == 0 or len(np.unique(y[te])) < 2:
            raise SplitError(f"fold {k} test partition is empty or single-class")
        check_disjoint({"train": tr, "test": te}, g)
        out.append((tr, te))
    return out


def check_disjoint(parts: dict, groups) -> None:
    """Raise LeakageError if any participant appears in two partitions."""
    g = np.asarray(groups)
    names = list(parts)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            try:
                assert_disjoint(g[parts[names[i]]], g[parts[names[j]]])
            except LeakageError as e:
                raise LeakageError(f"{names[i]} vs {names[j]}: {e}") from None
