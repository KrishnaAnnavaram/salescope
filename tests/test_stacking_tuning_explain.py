import pandas as pd
import pytest

from salescope.data import split_xy
from salescope.explain import EXPLAINED_COLUMNS, error_by_group, explain, shap_summary
from salescope.models import build_model
from salescope.stacking import AlignmentError, align_predictions, stack_evaluate
from salescope.tuning import SEARCH_SPACES, tune


def test_meta_model_never_sees_holdout_rows(train_df):
    res = stack_evaluate(train_df, ["mrp_baseline", "ridge"], "new_item", n_splits=3, seed=0)
    assert not set(res.meta_train_index) & set(res.holdout_index)
    assert res.n_dev + res.n_holdout == len(train_df)
    assert set(res.holdout_metrics) == {"mrp_baseline", "ridge", "stack"}
    assert all(w >= 0 for w in res.weights.values())


def test_stack_holdout_does_not_depend_on_row_order(train_df):
    shuffled = train_df.sample(frac=1.0, random_state=3)
    a = stack_evaluate(train_df, ["mrp_baseline", "ridge"], "new_outlet", 3, seed=0)
    b = stack_evaluate(shuffled, ["mrp_baseline", "ridge"], "new_outlet", 3, seed=0)
    assert set(a.holdout_index) == set(b.holdout_index)


def test_align_predictions_matches_rows_by_index():
    a = pd.Series([1.0, 2.0, 3.0], index=[10, 11, 12])
    b = pd.Series([30.0, 10.0, 20.0], index=[12, 10, 11])
    out = align_predictions({"a": a, "b": b})
    assert out.loc[12, "b"] == 30.0 and out.loc[10, "a"] == 1.0


def test_align_predictions_refuses_other_rows():
    with pytest.raises(AlignmentError):
        align_predictions({"a": pd.Series([1.0], index=[1]), "b": pd.Series([1.0], index=[2])})


def test_stack_needs_two_models(train_df):
    with pytest.raises(ValueError):
        stack_evaluate(train_df, ["ridge"])


def test_tuning_picks_params_from_space_and_scores_holdout(train_df):
    res = tune(train_df, "ridge", "new_outlet", n_iter=3, n_splits=3, seed=0)
    assert res.best_params["alpha"] in SEARCH_SPACES["ridge"]["alpha"]
    assert len(res.trials) == 3
    assert res.best_cv_rmse == min(t["cv_rmse"] for t in res.trials)
    assert set(res.holdout_metrics) == {"rmse", "rmsle", "mae", "r2"}


def test_tuning_unknown_backend(train_df):
    with pytest.raises(ValueError):
        tune(train_df, "ridge", n_iter=1, n_splits=2, backend="grid")


def test_optuna_backend_optional(train_df):
    pytest.importorskip("optuna")
    res = tune(train_df, "ridge", "new_outlet", n_iter=2, n_splits=2, seed=0, backend="optuna")
    assert len(res.trials) == 2


def test_explain_reports_importance_and_group_errors(train_df):
    res = explain(train_df, "random_forest", "new_outlet", seed=0)
    assert set(res.importance["column"]) <= set(EXPLAINED_COLUMNS)
    assert res.importance.iloc[0]["column"] == "Item_MRP"  # price drives synthetic sales
    assert {"group", "rows", "rmse", "mae", "bias"} <= set(res.errors_by_outlet_type.columns)


def test_error_by_group_values():
    out = error_by_group(pd.Series([1.0, 3.0, 10.0]), [2.0, 3.0, 10.0], pd.Series(["a", "a", "b"]))
    row = out.set_index("group").loc["a"]
    assert row["rows"] == 2 and row["bias"] == pytest.approx(0.5)


def test_shap_summary_optional(train_df):
    pytest.importorskip("shap")
    X, y = split_xy(train_df)
    model = build_model("random_forest", seed=0).fit(X, y)
    assert len(shap_summary(model, X, max_rows=50)) > 5
