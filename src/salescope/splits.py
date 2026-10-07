"""Cross-validation schemes: random rows, new outlets and new items.

``new_outlet`` and ``new_item`` put each outlet (or item) in exactly one test fold, so the model
never sees the test store (or product) during training. This answers the business question
"how good is the model for a store or product that it has not seen?".
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

from salescope.schema import ID_COL, OUTLET_COL

SCHEMES = {"random": None, "new_outlet": OUTLET_COL, "new_item": ID_COL}


class GroupIsolationError(AssertionError):
    """A group appears in both the training rows and the test rows of a fold."""


def group_column(scheme: str) -> str | None:
    if scheme not in SCHEMES:
        raise KeyError(f"unknown CV scheme {scheme!r}. Known: {list(SCHEMES)}")
    return SCHEMES[scheme]


def effective_splits(X: pd.DataFrame, scheme: str, n_splits: int) -> int:
    """Clamp the fold count to the number of groups (BigMart has only 10 outlets)."""
    col = group_column(scheme)
    n_units = len(X) if col is None else X[col].nunique()
    if n_units < 2:
        raise ValueError(f"scheme {scheme!r} needs at least 2 groups, found {n_units}")
    return max(2, min(n_splits, n_units))


def iter_folds(X: pd.DataFrame, scheme: str, n_splits: int = 5, seed: int = 42) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_positions, test_positions)``. Each row is in exactly one test fold.

    Groups are shuffled with ``seed`` and dealt into folds so that the folds have similar row counts.
    """
    col = group_column(scheme)
    k = effective_splits(X, scheme, n_splits)
    rng = np.random.default_rng(seed)
    if col is None:
        order = rng.permutation(len(X))
        fold_of = np.empty(len(X), dtype=int)
        fold_of[order] = np.arange(len(X)) % k
    else:
        keys = X[col].astype(str).to_numpy()
        groups, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        # biggest groups first (shuffled among equals), each to the fold with the fewest rows
        shuffled = rng.permutation(len(groups))
        order = shuffled[np.argsort(-counts[shuffled], kind="stable")]
        load = np.zeros(k, dtype=int)
        group_fold = np.empty(len(groups), dtype=int)
        for g in order:
            f = int(np.argmin(load))
            group_fold[g] = f
            load[f] += counts[g]
        fold_of = group_fold[inverse]
    for f in range(k):
        test = np.flatnonzero(fold_of == f)
        train = np.flatnonzero(fold_of != f)
        if col is not None:
            assert_group_isolation(X, train, test, col)
        yield train, test


def holdout_split(X: pd.DataFrame, scheme: str, fraction: float = 0.2, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(dev_positions, holdout_positions)`` that respect the scheme's groups."""
    if not 0.05 <= fraction <= 0.5:
        raise ValueError("holdout fraction must be between 0.05 and 0.5")
    col = group_column(scheme)
    rng = np.random.default_rng(seed)
    if col is None:
        perm = rng.permutation(len(X))
        n_hold = max(1, int(round(fraction * len(X))))
        hold = np.sort(perm[:n_hold])
    else:
        groups = np.unique(X[col].astype(str).to_numpy())  # sorted: independent of row order
        rng.shuffle(groups)
        n_hold = max(1, int(round(fraction * len(groups))))
        if n_hold >= len(groups):
            raise ValueError(f"scheme {scheme!r} has too few groups for a holdout")
        hold = np.flatnonzero(X[col].astype(str).isin(set(groups[:n_hold])).to_numpy())
    dev = np.setdiff1d(np.arange(len(X)), hold)
    if col is not None:
        assert_group_isolation(X, dev, hold, col)
    return dev, hold


def assert_group_isolation(X: pd.DataFrame, train: np.ndarray, test: np.ndarray, column: str) -> None:
    shared = set(X[column].iloc[train]) & set(X[column].iloc[test])
    if shared:
        raise GroupIsolationError(f"{len(shared)} {column} values are in train and test, first {sorted(shared)[0]}")
