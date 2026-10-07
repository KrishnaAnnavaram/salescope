import numpy as np
import pytest

from salescope.data import split_xy
from salescope.evaluate import compare, cross_validate, format_table
from salescope.features import LeakageError
from salescope.models import CORE_MODELS, build_model, parse_model_list
from salescope.schema import TARGET


@pytest.mark.parametrize("name", CORE_MODELS)
def test_each_core_model_fits_and_predicts(train_df, test_df, name):
    X, y = split_xy(train_df)
    model = build_model(name, seed=0).fit(X, y)
    pred = model.predict(test_df)
    assert pred.shape == (len(test_df),)
    assert np.isfinite(pred).all()


def test_unknown_test_categories_do_not_crash(train_df):
    X, y = split_xy(train_df)
    model = build_model("hist_gbm", seed=0).fit(X, y)
    odd = X.head(3).copy()
    odd["Item_Type"] = "Space Food"
    assert np.isfinite(model.predict(odd)).all()


def test_log_target_predictions_are_positive(train_df):
    X, y = split_xy(train_df)
    pred = build_model("ridge", log_target=True).fit(X, y).predict(X)
    assert (pred > 0).all()


def test_parse_model_list():
    assert parse_model_list("ridge, hist_gbm") == ["ridge", "hist_gbm"]
    with pytest.raises(KeyError):
        parse_model_list("ridge,lstm")


def test_cv_oof_covers_every_row_and_keeps_train_metrics_apart(train_df):
    res = cross_validate(train_df, "ridge", "new_outlet", n_splits=4, seed=0)
    assert res.oof.index.equals(train_df.index)
    assert not res.oof.isna().any()
    assert len(res.fold_test) == len(res.fold_train) == 4
    row = res.summary_row()
    assert "train_rmse_mean" in row and "accuracy_mean" not in row


def test_cv_is_reproducible(train_df):
    a = cross_validate(train_df, "random_forest", "random", 3, seed=5)
    b = cross_validate(train_df, "random_forest", "random", 3, seed=5)
    assert np.allclose(a.oof, b.oof)


def test_models_beat_mean_baseline_on_synthetic_data(train_df):
    table = compare(train_df, ["mean", "hist_gbm"], ["random"], n_splits=3, seed=0).set_index("model")
    assert table.loc["hist_gbm", "rmse_mean"] < 0.8 * table.loc["mean", "rmse_mean"]
    assert "hist_gbm" in format_table(table.reset_index())


def test_tree_model_overfit_is_visible(train_df):
    row = cross_validate(train_df, "hist_gbm", "random", 3, seed=0).summary_row()
    assert row["train_rmse_mean"] < row["rmse_mean"]


def test_target_never_reaches_the_pipeline(train_df):
    with pytest.raises(LeakageError):
        build_model("ridge").fit(train_df, train_df[TARGET])


@pytest.mark.parametrize("name", ["lightgbm", "catboost"])
def test_optional_boosters(train_df, name):
    pytest.importorskip(name)
    X, y = split_xy(train_df)
    pred = build_model(name, seed=0).fit(X, y).predict(X)
    assert np.isfinite(pred).all()


def test_predictions_are_never_negative(train_df):
    X, y = split_xy(train_df)
    pred = build_model("ridge", seed=0).fit(X, y).predict(X)
    assert (pred >= 0).all()
