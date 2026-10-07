"""Regression metrics. salescope reports only RMSE, RMSLE, MAE and R2 - never a median-split accuracy."""

from __future__ import annotations

from typing import Callable

import numpy as np

METRIC_NAMES = ("rmse", "rmsle", "mae", "r2")


def _arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    if yt.size == 0:
        raise ValueError("empty input")
    if not (np.isfinite(yt).all() and np.isfinite(yp).all()):
        raise ValueError("non-finite values")
    return yt, yp


def rmse(y_true, y_pred) -> float:
    yt, yp = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def rmsle(y_true, y_pred) -> float:
    """Root mean squared log error. Negative predictions are clipped to 0 (sales cannot be negative)."""
    yt, yp = _arrays(y_true, y_pred)
    if (yt < 0).any():
        raise ValueError("rmsle needs non-negative targets")
    return float(np.sqrt(np.mean((np.log1p(yt) - np.log1p(np.clip(yp, 0, None))) ** 2)))


def mae(y_true, y_pred) -> float:
    yt, yp = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def r2(y_true, y_pred) -> float:
    yt, yp = _arrays(y_true, y_pred)
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot == 0:
        return 0.0
    return 1.0 - float(np.sum((yt - yp) ** 2)) / ss_tot


METRICS: dict[str, Callable] = {"rmse": rmse, "rmsle": rmsle, "mae": mae, "r2": r2}


def regression_report(y_true, y_pred) -> dict[str, float]:
    return {name: fn(y_true, y_pred) for name, fn in METRICS.items()}


def bootstrap_ci(y_true, y_pred, metric: str = "rmse", n_boot: int = 300, level: float = 0.95,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of one metric over rows."""
    yt, yp = _arrays(y_true, y_pred)
    fn = METRICS[metric]
    rng = np.random.default_rng(seed)
    n = len(yt)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = fn(yt[idx], yp[idx])
    alpha = (1 - level) / 2
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha))
