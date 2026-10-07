import pandas as pd
import pytest

from salescope.data import normalise_labels
from salescope.schema import validate
from salescope.synthetic import make_dataset


@pytest.fixture(scope="session")
def raw_frames():
    return make_dataset(n_items=220, n_outlets=8, seed=7)


@pytest.fixture(scope="session")
def train_df(raw_frames) -> pd.DataFrame:
    return validate(normalise_labels(raw_frames[0]))


@pytest.fixture(scope="session")
def test_df(raw_frames) -> pd.DataFrame:
    return validate(normalise_labels(raw_frames[1]), require_target=False)


@pytest.fixture()
def train_csv(tmp_path, raw_frames):
    path = tmp_path / "train.csv"
    raw_frames[0].to_csv(path, index=False)
    return path
