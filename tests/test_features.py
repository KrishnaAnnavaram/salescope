import numpy as np
import pandas as pd
import pytest

from salescope.data import split_xy
from salescope.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, BigMartFeatures, LeakageError
from salescope.schema import ID_COL, OUTLET_COL


def _row(**kw):
    base = {ID_COL: "FDA01", "Item_Weight": 10.0, "Item_Fat_Content": "Low Fat", "Item_Visibility": 0.05,
            "Item_Type": "Dairy", "Item_MRP": 100.0, OUTLET_COL: "OUT010", "Outlet_Establishment_Year": 2000,
            "Outlet_Size": "Small", "Outlet_Location_Type": "Tier 1", "Outlet_Type": "Grocery Store"}
    base.update(kw)
    return base


def test_target_in_features_raises(train_df):
    with pytest.raises(LeakageError):
        BigMartFeatures().fit(train_df)


def test_output_columns_exclude_identifiers(train_df):
    X, _ = split_xy(train_df)
    out = BigMartFeatures().fit(X).transform(X)
    assert list(out.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert ID_COL not in out.columns and OUTLET_COL not in out.columns
    assert not out.isna().any().any()


def test_weight_is_recovered_from_same_item_in_training_rows():
    train = pd.DataFrame([_row(Item_Weight=12.5), _row(**{OUTLET_COL: "OUT011", "Item_Weight": np.nan})])
    out = BigMartFeatures().fit(train).transform(train)
    assert out["Item_Weight"].tolist() == [12.5, 12.5]
    assert out["Weight_Was_Missing"].tolist() == [0.0, 1.0]


def test_statistics_come_from_training_rows_only():
    train = pd.DataFrame([_row(Item_Weight=10.0), _row(**{ID_COL: "FDB02", "Item_Weight": 20.0})])
    test = pd.DataFrame([_row(**{ID_COL: "FDC03", "Item_Weight": np.nan}),
                         _row(**{ID_COL: "FDC03", OUTLET_COL: "OUT011", "Item_Weight": 99.0})])
    feats = BigMartFeatures().fit(train)
    out = feats.transform(test)
    # the test item's own weight (99) must not fill its missing row: the fill is the train median
    assert out["Item_Weight"].iloc[0] == pytest.approx(15.0)
    assert "FDC03" not in feats.item_weight_


def test_zero_visibility_becomes_item_mean_and_is_flagged():
    train = pd.DataFrame([_row(Item_Visibility=0.04), _row(**{OUTLET_COL: "OUT011", "Item_Visibility": 0.0}),
                          _row(**{OUTLET_COL: "OUT012", "Item_Visibility": 0.06})])
    out = BigMartFeatures().fit(train).transform(train)
    assert out["Item_Visibility"].iloc[1] == pytest.approx(0.05)
    assert out["Visibility_Was_Zero"].tolist() == [0.0, 1.0, 0.0]
    assert out["Visibility_Ratio"].iloc[1] == pytest.approx(1.0)


def test_outlet_size_filled_from_outlet_type_and_age_derived():
    train = pd.DataFrame([_row(), _row(**{ID_COL: "FDB02", OUTLET_COL: "OUT011", "Outlet_Size": np.nan,
                                          "Outlet_Establishment_Year": 1990})])
    out = BigMartFeatures(reference_year=2013).fit(train).transform(train)
    assert out["Outlet_Size"].tolist() == ["Small", "Small"]
    assert out["Outlet_Size_Was_Missing"].tolist() == [0.0, 1.0]
    assert out["Outlet_Age"].tolist() == [13.0, 23.0]


def test_item_category_from_prefix_and_non_edible_fat():
    train = pd.DataFrame([_row(), _row(**{ID_COL: "NCA01", "Item_Type": "Household"}),
                          _row(**{ID_COL: "DRA01", "Item_Type": "Soft Drinks"})])
    out = BigMartFeatures().fit(train).transform(train)
    assert out["Item_Category"].tolist() == ["Food", "Non-Consumable", "Drinks"]
    assert out["Item_Fat_Content"].tolist() == ["Low Fat", "Non-Edible", "Low Fat"]


def test_transform_before_fit_raises(train_df):
    X, _ = split_xy(train_df)
    with pytest.raises(RuntimeError):
        BigMartFeatures().transform(X)


def test_outlet_id_is_optional_feature(train_df):
    X, _ = split_xy(train_df)
    out = BigMartFeatures(include_outlet_id=True).fit(X).transform(X)
    assert OUTLET_COL in out.columns
