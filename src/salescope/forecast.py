"""Track B: real forecasting on data with real timestamps, with a rolling-origin backtest.

BigMart has no dates. This module never invents a time axis: ``require_real_dates`` refuses a
frame without a date column, and refuses the BigMart columns in all cases. Use a dated retail data
set (for example M5 or the Walmart store sales) in long format: ``series_id, date, sales``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from salescope.schema import ID_COL, TARGET

SERIES, DATE, VALUE = "series_id", "date", "sales"


class NoTimeAxisError(ValueError):
    """The data has no real timestamps, so a forecast is not possible."""


def require_real_dates(df: pd.DataFrame, date_col: str = DATE, min_periods: int = 24) -> pd.DataFrame:
    """Validate a long-format series frame and return it sorted with a parsed date column."""
    if ID_COL in df.columns and TARGET in df.columns and date_col not in df.columns:
        raise NoTimeAxisError(
            "BigMart is cross-sectional: it has no dates. salescope does not make dates from the row order. "
            "Use the tabular commands for BigMart and a dated data set for forecasting."
        )
    missing = [c for c in (SERIES, date_col, VALUE) if c not in df.columns]
    if missing:
        raise NoTimeAxisError(f"missing columns {missing}. Forecasting needs {SERIES}, {date_col}, {VALUE}.")
    out = df[[SERIES, date_col, VALUE]].rename(columns={date_col: DATE}).copy()
    out[DATE] = pd.to_datetime(out[DATE], errors="coerce")
    if out[DATE].isna().any():
        raise NoTimeAxisError(f"{int(out[DATE].isna().sum())} values in {date_col!r} are not dates")
    if out.duplicated([SERIES, DATE]).any():
        raise ValueError("a series has the same date twice")
    if out[VALUE].isna().any() or (pd.to_numeric(out[VALUE], errors="coerce").isna()).any():
        raise ValueError(f"{VALUE} has missing or non-numeric values")
    out[VALUE] = out[VALUE].astype(float)
    out = out.sort_values([SERIES, DATE]).reset_index(drop=True)
    for sid, part in out.groupby(SERIES):
        if len(part) < min_periods:
            raise ValueError(f"series {sid!r} has {len(part)} periods, needs at least {min_periods}")
        expected = pd.date_range(part[DATE].iloc[0], periods=len(part), freq="MS")
        if not part[DATE].reset_index(drop=True).equals(pd.Series(expected, name=DATE)):
            raise ValueError(f"series {sid!r} is not a gap-free monthly series (month-start dates)")
    return out


def make_series(n_series: int = 6, n_months: int = 72, seed: int = 0) -> pd.DataFrame:
    """Synthetic monthly store sales: level x trend x yearly season x noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_months, freq="MS")
    t = np.arange(n_months)
    rows = []
    for s in range(n_series):
        level = rng.lognormal(10, 0.4)
        trend = 1 + rng.uniform(-0.002, 0.008) * t
        season = 1 + rng.uniform(0.1, 0.3) * np.sin(2 * np.pi * (t + rng.integers(12)) / 12)
        noise = rng.normal(1, 0.04, n_months)
        rows.append(pd.DataFrame({SERIES: f"S{s:02d}", DATE: dates, VALUE: np.round(level * trend * season * noise, 2)}))
    return pd.concat(rows, ignore_index=True)


class Forecaster(Protocol):
    name: str

    def fit(self, history: pd.DataFrame) -> "Forecaster": ...

    def predict(self, horizon: int) -> pd.DataFrame: ...


def _future_dates(last: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(last + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")


class NaiveForecaster:
    name = "naive"

    def fit(self, history: pd.DataFrame):
        self.history_ = history
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        rows = []
        for sid, part in self.history_.groupby(SERIES):
            for d in _future_dates(part[DATE].iloc[-1], horizon):
                rows.append({SERIES: sid, DATE: d, "forecast": float(part[VALUE].iloc[-1])})
        return pd.DataFrame(rows)


class SeasonalNaiveForecaster:
    """Forecast = the value of the same month one season ago."""

    name = "seasonal_naive"

    def __init__(self, season: int = 12):
        self.season = season

    def fit(self, history: pd.DataFrame):
        self.history_ = history
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        rows = []
        for sid, part in self.history_.groupby(SERIES):
            values = part[VALUE].to_numpy()
            if len(values) < self.season:
                raise ValueError(f"series {sid!r} is shorter than one season")
            last_season = values[-self.season:]
            for h, d in enumerate(_future_dates(part[DATE].iloc[-1], horizon)):
                rows.append({SERIES: sid, DATE: d, "forecast": float(last_season[h % self.season])})
        return pd.DataFrame(rows)


class LagGBMForecaster:
    """One global gradient-boosting model on scaled lags, with recursive multi-step forecasts.

    Each series is divided by its own historical mean, so one model serves series of different size.
    """

    name = "lag_gbm"

    def __init__(self, lags: int = 12, seed: int = 42):
        self.lags = lags
        self.seed = seed

    def _rows(self, values: np.ndarray, months: np.ndarray):
        X, y = [], []
        for i in range(self.lags, len(values)):
            window = values[i - self.lags:i]
            X.append(np.concatenate([window[::-1], [window[-3:].mean(), months[i]]]))
            y.append(values[i])
        return X, y

    def fit(self, history: pd.DataFrame):
        self.history_ = history
        self.scale_ = history.groupby(SERIES)[VALUE].mean().to_dict()
        X, y = [], []
        for sid, part in history.groupby(SERIES):
            vals = part[VALUE].to_numpy() / self.scale_[sid]
            xs, ys = self._rows(vals, part[DATE].dt.month.to_numpy())
            X += xs
            y += ys
        if not X:
            raise ValueError("history is too short for the lag model")
        self.model_ = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=self.seed)
        self.model_.fit(np.asarray(X), np.asarray(y))
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        rows = []
        for sid, part in self.history_.groupby(SERIES):
            vals = list(part[VALUE].to_numpy() / self.scale_[sid])
            for d in _future_dates(part[DATE].iloc[-1], horizon):
                window = np.asarray(vals[-self.lags:])
                x = np.concatenate([window[::-1], [window[-3:].mean(), d.month]])
                nxt = float(self.model_.predict(x.reshape(1, -1))[0])
                vals.append(nxt)
                rows.append({SERIES: sid, DATE: d, "forecast": nxt * self.scale_[sid]})
        return pd.DataFrame(rows)


class SarimaForecaster:
    """Seasonal ARIMA for each series. Needs ``pip install salescope[forecast]``."""

    name = "sarima"

    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)):
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(self, history: pd.DataFrame):
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError as exc:
            raise ImportError("statsmodels is not installed. Run: pip install 'salescope[forecast]'") from exc
        self.history_ = history
        self.fits_ = {}
        for sid, part in history.groupby(SERIES):
            model = SARIMAX(part[VALUE].to_numpy(), order=self.order, seasonal_order=self.seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False)
            self.fits_[sid] = model.fit(disp=False)
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        rows = []
        for sid, part in self.history_.groupby(SERIES):
            fc = self.fits_[sid].forecast(horizon)
            for d, v in zip(_future_dates(part[DATE].iloc[-1], horizon), fc):
                rows.append({SERIES: sid, DATE: d, "forecast": float(v)})
        return pd.DataFrame(rows)


FORECASTERS: dict[str, Callable[[], Forecaster]] = {
    "naive": NaiveForecaster,
    "seasonal_naive": SeasonalNaiveForecaster,
    "lag_gbm": LagGBMForecaster,
    "sarima": SarimaForecaster,
}


def _mase_scale(history: pd.DataFrame, season: int) -> dict[str, float]:
    out = {}
    for sid, part in history.groupby(SERIES):
        v = part[VALUE].to_numpy()
        diffs = np.abs(v[season:] - v[:-season])
        out[sid] = float(diffs.mean()) if len(diffs) and diffs.mean() > 0 else float("nan")
    return out


@dataclass
class BacktestResult:
    folds: pd.DataFrame  # one row for each (origin, model)

    def summary(self) -> pd.DataFrame:
        return (self.folds.groupby("model")[["mae", "rmse", "smape", "mase"]].mean()
                .sort_values("mase").reset_index())


def rolling_origin_backtest(
    df: pd.DataFrame,
    models: list[str] | None = None,
    horizon: int = 6,
    n_origins: int = 3,
    season: int = 12,
) -> BacktestResult:
    """Fit each model on the history before each origin and score the next ``horizon`` months."""
    data = require_real_dates(df, min_periods=2 * season + horizon)
    names = models or ["naive", "seasonal_naive", "lag_gbm"]
    unknown = [m for m in names if m not in FORECASTERS]
    if unknown:
        raise KeyError(f"unknown forecaster(s) {unknown}. Known: {sorted(FORECASTERS)}")
    dates = np.sort(data[DATE].unique())
    if len(dates) < 2 * season + horizon + n_origins - 1:
        raise ValueError("not enough months for this many origins")
    rows = []
    for k in range(n_origins):
        cut = len(dates) - horizon - (n_origins - 1 - k) * horizon
        origin = pd.Timestamp(dates[cut])
        history = data[data[DATE] < origin]
        future = data[(data[DATE] >= origin) & (data[DATE] < origin + pd.DateOffset(months=horizon))]
        scale = _mase_scale(history, season)
        for name in names:
            fc = FORECASTERS[name]().fit(history).predict(horizon)
            merged = future.merge(fc, on=[SERIES, DATE], how="inner")
            if len(merged) != len(future):
                raise RuntimeError(f"{name} did not forecast every future month")
            err = merged["forecast"] - merged[VALUE]
            denom = (merged["forecast"].abs() + merged[VALUE].abs()).replace(0, np.nan)
            mase_parts = [np.abs(g["forecast"] - g[VALUE]).mean() / scale[sid] for sid, g in merged.groupby(SERIES)]
            rows.append({
                "origin": origin.date().isoformat(), "model": name, "rows": len(merged),
                "mae": float(err.abs().mean()), "rmse": float(np.sqrt((err ** 2).mean())),
                "smape": float((2 * err.abs() / denom).mean() * 100), "mase": float(np.nanmean(mase_parts)),
            })
    return BacktestResult(pd.DataFrame(rows))
