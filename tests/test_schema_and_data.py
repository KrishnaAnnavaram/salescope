import numpy as np
import pandas as pd
import pytest

from salescope.data import load_frame, normalise_labels, split_xy
from salescope.schema import ID_COL, OUTLET_COL, TARGET, SchemaError, validate


def test_synthetic_train_passes_schema(train_df):
    assert len(train_df) > 300
    assert set(train_df["Item_Fat_Content"]) <= {"Low Fat", "Regular"}


def test_test_file_without_target_passes_only_when_allowed(raw_frames):
    test = normalise_labels(raw_frames[1])
    validate(test, require_target=False)
    with pytest.raises(SchemaError, match="missing columns"):
        validate(test, require_target=True)


def test_fat_content_variants_are_normalised():
    frame = pd.DataFrame({"Item_Fat_Content": ["LF", "low fat", "Low Fat", "reg", "Regular", " LF "]})
    out = normalise_labels(frame)
    assert list(out["Item_Fat_Content"]) == ["Low Fat", "Low Fat", "Low Fat", "Regular", "Regular", "Low Fat"]


def test_validator_reports_every_problem(raw_frames):
    bad = normalise_labels(raw_frames[0]).copy()
    bad.loc[0, ID_COL] = "bad-id"
    bad.loc[1, "Item_MRP"] = -3
    bad.loc[2, "Outlet_Type"] = "Kiosk"
    bad.loc[3, TARGET] = -1
    with pytest.raises(SchemaError) as info:
        validate(bad)
    text = " ".join(info.value.problems)
    assert len(info.value.problems) == 4
    for needle in (ID_COL, "Item_MRP", "Kiosk", TARGET):
        assert needle in text


def test_duplicate_item_outlet_pairs_are_refused(raw_frames):
    frame = normalise_labels(raw_frames[0])
    dup = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(SchemaError, match="duplicate"):
        validate(dup)


def test_missing_weight_and_size_are_allowed_but_missing_mrp_is_not(raw_frames):
    frame = normalise_labels(raw_frames[0])
    assert frame["Item_Weight"].isna().any() and frame["Outlet_Size"].isna().any()
    frame = frame.copy()
    frame.loc[5, "Item_MRP"] = np.nan
    with pytest.raises(SchemaError, match="Item_MRP: 1 missing"):
        validate(frame)


def test_load_frame_missing_file_points_to_help(tmp_path):
    with pytest.raises(FileNotFoundError, match="salescope synth"):
        load_frame(tmp_path / "nope.csv")


def test_load_frame_reads_relative_paths(train_csv):
    df = load_frame(train_csv)
    assert {ID_COL, OUTLET_COL, TARGET} <= set(df.columns)


def test_split_xy_removes_target(train_df):
    X, y = split_xy(train_df)
    assert TARGET not in X.columns
    assert len(X) == len(y)
