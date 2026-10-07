import numpy as np
import pandas as pd
import pytest

import salescope.forecast as fc
from salescope.forecast import (DATE, SERIES, VALUE, LagGBMForecaster, NoTimeAxisError, SeasonalNaiveForecaster,
                                make_series, require_real_dates, rolling_origin_backtest)


def test_bigmart_frame_is_refused(train_df):
    with pytest.raises(NoTimeAxisError, match="no dates"):
        require_real_dates(train_df)


def test_unparseable_dates_are_refused():
    df = make_series(n_series=1, n_months=30)
    df[DATE] = df[DATE].astype(str)
    df.loc[3, DATE] = "row 3"
    with pytest.raises(NoTimeAxisError):
        require_real_dates(df)


def test_gaps_are_refused():
    df = make_series(n_series=1, n_months=30).drop(index=10)
    with pytest.raises(ValueError, match="gap-free"):
        require_real_dates(df)


def test_seasonal_naive_repeats_last_season():
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    values = np.tile(np.arange(12, dtype=float), 2)
    hist = pd.DataFrame({SERIES: "A", DATE: dates, VALUE: values})
    out = SeasonalNaiveForecaster().fit(hist).predict(14)
    assert out["forecast"].tolist() == list(np.arange(12.0)) + [0.0, 1.0]
    assert out[DATE].iloc[0] == pd.Timestamp("2022-01-01")


def test_backtest_never_shows_future_to_models(monkeypatch):
    df = make_series(n_series=2, n_months=48, seed=1)
    seen = []

    class Spy(SeasonalNaiveForecaster):
        name = "spy"

        def fit(self, history):
            seen.append(history[DATE].max())
            return super().fit(history)

    monkeypatch.setitem(fc.FORECASTERS, "spy", Spy)
    res = rolling_origin_backtest(df, ["spy"], horizon=6, n_origins=3)
    origins = pd.to_datetime(res.folds["origin"]).tolist()
    assert len(seen) == 3
    for last_seen, origin in zip(seen, origins):
        assert last_seen < origin


def test_backtest_on_synthetic_series():
    res = rolling_origin_backtest(make_series(seed=0), ["naive", "seasonal_naive", "lag_gbm"], horizon=6, n_origins=2)
    summary = res.summary().set_index("model")
    assert len(res.folds) == 6
    assert summary.loc["seasonal_naive", "mase"] < summary.loc["naive", "mase"]


def test_lag_model_forecasts_each_series():
    hist = make_series(n_series=3, n_months=40, seed=2)
    out = LagGBMForecaster().fit(hist).predict(4)
    assert len(out) == 12 and np.isfinite(out["forecast"]).all()


def test_sarima_optional():
    pytest.importorskip("statsmodels")
    res = rolling_origin_backtest(make_series(n_series=2, seed=0), ["sarima"], horizon=3, n_origins=1)
    assert np.isfinite(res.folds["mae"]).all()
