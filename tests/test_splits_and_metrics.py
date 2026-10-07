import numpy as np
import pandas as pd
import pytest

from salescope.metrics import METRIC_NAMES, bootstrap_ci, mae, r2, regression_report, rmse, rmsle
from salescope.schema import ID_COL, OUTLET_COL
from salescope.splits import (GroupIsolationError, assert_group_isolation, effective_splits, holdout_split,
                              iter_folds)


@pytest.mark.parametrize("scheme,col", [("new_outlet", OUTLET_COL), ("new_item", ID_COL)])
def test_grouped_folds_never_share_a_group(train_df, scheme, col):
    for train, test in iter_folds(train_df, scheme, 5, seed=1):
        assert not set(train_df[col].iloc[train]) & set(train_df[col].iloc[test])


@pytest.mark.parametrize("scheme", ["random", "new_outlet", "new_item"])
def test_each_row_is_tested_exactly_once(train_df, scheme):
    seen = np.concatenate([test for _, test in iter_folds(train_df, scheme, 5, seed=1)])
    assert sorted(seen.tolist()) == list(range(len(train_df)))


def test_folds_are_deterministic_for_a_seed(train_df):
    a = [t.tolist() for _, t in iter_folds(train_df, "new_item", 4, seed=3)]
    b = [t.tolist() for _, t in iter_folds(train_df, "new_item", 4, seed=3)]
    c = [t.tolist() for _, t in iter_folds(train_df, "new_item", 4, seed=4)]
    assert a == b and a != c


def test_fold_count_is_clamped_to_number_of_outlets(train_df):
    assert effective_splits(train_df, "new_outlet", 50) == train_df[OUTLET_COL].nunique()


def test_unknown_scheme_raises(train_df):
    with pytest.raises(KeyError):
        list(iter_folds(train_df, "by_month", 5))


def test_isolation_check_detects_a_shared_group():
    frame = pd.DataFrame({OUTLET_COL: ["OUT001", "OUT001", "OUT002"]})
    with pytest.raises(GroupIsolationError):
        assert_group_isolation(frame, np.array([0]), np.array([1, 2]), OUTLET_COL)


def test_holdout_respects_groups(train_df):
    dev, hold = holdout_split(train_df, "new_outlet", 0.25, seed=0)
    assert not set(train_df[OUTLET_COL].iloc[dev]) & set(train_df[OUTLET_COL].iloc[hold])
    assert len(dev) + len(hold) == len(train_df)


def test_metric_values_on_known_inputs():
    y, p = [1.0, 2.0, 3.0], [1.0, 2.0, 5.0]
    assert rmse(y, p) == pytest.approx(np.sqrt(4 / 3))
    assert mae(y, p) == pytest.approx(2 / 3)
    assert r2(y, y) == 1.0
    assert rmsle([0.0], [-5.0]) == 0.0  # a negative prediction is clipped to 0


def test_report_has_only_regression_metrics():
    rep = regression_report([1.0, 2.0, 4.0], [1.5, 2.0, 3.0])
    assert tuple(rep) == METRIC_NAMES
    assert "accuracy" not in rep and "roc_auc" not in rep


def test_bootstrap_interval_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.normal(100, 10, 400)
    p = y + rng.normal(0, 5, 400)
    lo, hi = bootstrap_ci(y, p, "rmse", n_boot=200)
    assert lo < rmse(y, p) < hi


def test_metrics_refuse_bad_input():
    with pytest.raises(ValueError):
        rmse([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        rmse([np.nan], [1.0])
