import json

import pandas as pd
import pytest

from salescope.cli import main
from salescope.config import ConfigError, Settings, load_dotenv


def test_settings_defaults_and_overrides():
    s = Settings.from_env({})
    assert (s.seed, s.n_splits, s.reference_year, str(s.data_dir)) == (42, 5, 2013, "data")
    s = Settings.from_env({"SALESCOPE_SEED": "7", "SALESCOPE_N_SPLITS": "3", "SALESCOPE_DATA_DIR": "x"})
    assert (s.seed, s.n_splits, s.train_file.name) == (7, 3, "retail_mart_train.csv")


@pytest.mark.parametrize("env", [{"SALESCOPE_SEED": "abc"}, {"SALESCOPE_N_SPLITS": "1"}])
def test_bad_settings_raise(env):
    with pytest.raises(ConfigError):
        Settings.from_env(env)


def test_dotenv_does_not_replace_set_variables(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("SALESCOPE_SEED=9\nSALESCOPE_N_SPLITS=4\n# comment\n", encoding="utf-8")
    monkeypatch.setenv("SALESCOPE_SEED", "1")
    monkeypatch.delenv("SALESCOPE_N_SPLITS", raising=False)
    assert load_dotenv(env_file) == 1
    s = Settings.from_env()
    assert (s.seed, s.n_splits) == (1, 4)


def test_cli_round_trip(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SALESCOPE_N_SPLITS", "3")
    data = tmp_path / "d"
    assert main(["synth", "--out", str(data), "--items", "150", "--outlets", "6"]) == 0
    train, test = data / "retail_mart_train.csv", data / "retail_mart_test.csv"
    assert main(["validate", str(train)]) == 0
    assert main(["validate", str(test), "--no-target"]) == 0
    capsys.readouterr()
    assert main(["evaluate", "--train", str(train), "--models", "mean,ridge", "--schemes", "new_outlet", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {r["model"] for r in rows} == {"mean", "ridge"}
    model_dir = tmp_path / "m"
    assert main(["train", "--train", str(train), "--model", "ridge", "--out", str(model_dir)]) == 0
    meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["model"] == "ridge" and len(meta["train_sha256"]) == 64
    out = tmp_path / "pred.csv"
    assert main(["predict", "--model-dir", str(model_dir), "--input", str(test), "--out", str(out)]) == 0
    pred = pd.read_csv(out)
    assert len(pred) == len(pd.read_csv(test)) and (pred["Item_Outlet_Sales"] >= 0).all()


def test_cli_stack_explain_and_backtest(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SALESCOPE_N_SPLITS", "3")
    data = tmp_path / "d"
    main(["synth", "--out", str(data), "--items", "150", "--outlets", "6"])
    train = str(data / "retail_mart_train.csv")
    capsys.readouterr()
    assert main(["stack", "--train", train, "--models", "mrp_baseline,ridge", "--json"]) == 0
    assert "weights" in json.loads(capsys.readouterr().out)
    assert main(["explain", "--train", train, "--model", "ridge"]) == 0
    assert main(["backtest", "--models", "naive,seasonal_naive", "--horizon", "3", "--origins", "2"]) == 0
    assert "seasonal_naive" in capsys.readouterr().out


def test_cli_errors_are_reported_not_raised(tmp_path, capsys):
    assert main(["validate", str(tmp_path / "missing.csv")]) == 1
    assert "error:" in capsys.readouterr().err
    assert main(["evaluate", "--train", str(tmp_path / "x.csv"), "--schemes", "by_month"]) == 1


def test_backtest_refuses_bigmart_file(tmp_path, capsys):
    data = tmp_path / "d"
    main(["synth", "--out", str(data), "--items", "60", "--outlets", "4"])
    capsys.readouterr()
    assert main(["backtest", "--series", str(data / "retail_mart_train.csv")]) == 1
    assert "no dates" in capsys.readouterr().err
