"""Command line interface: ``salescope <command> [options]``."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from salescope.config import ConfigError, Settings, load_dotenv
from salescope.data import load_frame, split_xy
from salescope.evaluate import compare, format_table
from salescope.explain import explain
from salescope.forecast import FORECASTERS, NoTimeAxisError, make_series, rolling_origin_backtest
from salescope.models import CORE_MODELS, build_model, parse_model_list
from salescope.persistence import frame_fingerprint, load_model, save_model
from salescope.schema import ID_COL, OUTLET_COL, TARGET, SchemaError
from salescope.splits import SCHEMES
from salescope.stacking import stack_evaluate
from salescope.synthetic import make_dataset
from salescope.tuning import SEARCH_SPACES, tune


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (pd.Index, pd.Series)):
        return list(o)
    return str(o)


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, default=_json_default))


def _schemes(text: str) -> list[str]:
    names = [s.strip() for s in text.split(",") if s.strip()]
    bad = [s for s in names if s not in SCHEMES]
    if bad or not names:
        raise ValueError(f"unknown CV scheme(s) {bad}. Known: {list(SCHEMES)}")
    return names


def _train_path(args, settings: Settings) -> Path:
    return Path(args.train) if args.train else settings.train_file


# -- commands -----------------------------------------------------------------------------------
def cmd_synth(args, settings: Settings) -> int:
    train, test = make_dataset(n_items=args.items, n_outlets=args.outlets, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "retail_mart_train.csv", index=False)
    test.to_csv(out / "retail_mart_test.csv", index=False)
    print(f"wrote {len(train)} train rows and {len(test)} test rows (SYNTHETIC) to {out}")
    return 0


def cmd_validate(args, settings: Settings) -> int:
    df = load_frame(args.path, require_target=not args.no_target)
    print(f"OK: {len(df)} rows, {df[ID_COL].nunique()} items, {df[OUTLET_COL].nunique()} outlets")
    print(f"zero visibility: {int((df['Item_Visibility'] == 0).sum())} rows, "
          f"missing Item_Weight: {int(df['Item_Weight'].isna().sum())}, "
          f"missing Outlet_Size: {int(df['Outlet_Size'].isna().sum())}")
    return 0


def cmd_evaluate(args, settings: Settings) -> int:
    df = load_frame(_train_path(args, settings))
    table = compare(df, parse_model_list(args.models), _schemes(args.schemes), args.folds or settings.n_splits,
                    settings.seed, args.log_target, settings.reference_year)
    if args.json:
        _print_json(table.to_dict(orient="records"))
    else:
        print(format_table(table))
        print("\nAll test metrics are out-of-fold. train_rmse_mean is the fit on the training folds.")
    return 0


def cmd_tune(args, settings: Settings) -> int:
    df = load_frame(_train_path(args, settings))
    res = tune(df, args.model, args.scheme, args.iter, args.folds or settings.n_splits, settings.seed,
               backend=args.backend, reference_year=settings.reference_year)
    _print_json({"model": res.model, "scheme": res.scheme, "best_params": res.best_params,
                 "best_cv_rmse": res.best_cv_rmse, "holdout_tuned": res.holdout_metrics,
                 "holdout_default": res.default_holdout_metrics, "trials": len(res.trials)})
    return 0


def cmd_stack(args, settings: Settings) -> int:
    df = load_frame(_train_path(args, settings))
    res = stack_evaluate(df, parse_model_list(args.models), args.scheme, args.folds or settings.n_splits,
                         settings.seed, reference_year=settings.reference_year)
    if args.json:
        _print_json({"scheme": res.scheme, "weights": res.weights, "intercept": res.intercept,
                     "holdout": res.holdout_metrics, "n_dev": res.n_dev, "n_holdout": res.n_holdout})
    else:
        print(f"scheme {res.scheme}: meta-model fitted on {res.n_dev} OOF rows, scored on {res.n_holdout} holdout rows")
        print("weights: " + ", ".join(f"{k}={v:.3f}" for k, v in res.weights.items()) + f", intercept={res.intercept:.1f}")
        print(res.table().round(3).to_string())
    return 0


def cmd_train(args, settings: Settings) -> int:
    df = load_frame(_train_path(args, settings))
    X, y = split_xy(df)
    model = build_model(args.model, seed=settings.seed, reference_year=settings.reference_year,
                        log_target=args.log_target).fit(X, y)
    out = Path(args.out) if args.out else settings.artifact_dir / args.model
    save_model(model, out, {"model": args.model, "log_target": args.log_target, "seed": settings.seed,
                            "reference_year": settings.reference_year, "train_rows": len(df),
                            "train_sha256": frame_fingerprint(df)})
    print(f"saved {args.model} trained on {len(df)} rows to {out}")
    return 0


def cmd_predict(args, settings: Settings) -> int:
    model, meta = load_model(args.model_dir)
    df = load_frame(args.input, require_target=False)
    X = df.drop(columns=[TARGET]) if TARGET in df else df
    pred = np.clip(model.predict(X), 0, None)
    out = pd.DataFrame({ID_COL: df[ID_COL], OUTLET_COL: df[OUTLET_COL], TARGET: np.round(pred, 4)})
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} predictions from {meta.get('model', '?')} to {args.out}")
    return 0


def cmd_explain(args, settings: Settings) -> int:
    df = load_frame(_train_path(args, settings))
    res = explain(df, args.model, args.scheme, settings.seed, reference_year=settings.reference_year)
    print(f"permutation importance ({res.model}, {res.scheme} holdout): RMSE increase when a column is shuffled")
    print(res.importance.round(2).to_string(index=False))
    print("\nerrors by Outlet_Type:")
    print(res.errors_by_outlet_type.round(1).to_string(index=False))
    print("\nerrors by item category:")
    print(res.errors_by_item_category.round(1).to_string(index=False))
    return 0


def cmd_backtest(args, settings: Settings) -> int:
    if args.series:
        df = pd.read_csv(args.series)
        source = args.series
    else:
        df = make_series(seed=settings.seed)
        source = "SYNTHETIC series (make_series)"
    models = [m.strip() for m in args.models.split(",")] if args.models else None
    res = rolling_origin_backtest(df, models, horizon=args.horizon, n_origins=args.origins)
    print(f"rolling-origin backtest on {source}: horizon {args.horizon} months, {args.origins} origins")
    print(res.summary().round(3).to_string(index=False))
    return 0


def cmd_demo(args, settings: Settings) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        train, _ = make_dataset(seed=settings.seed)
        path = Path(tmp) / "train.csv"
        train.to_csv(path, index=False)
        df = load_frame(path)
        print(f"SYNTHETIC data: {len(df)} rows, {df[ID_COL].nunique()} items, {df[OUTLET_COL].nunique()} outlets\n")
        table = compare(df, list(CORE_MODELS), list(SCHEMES), settings.n_splits, settings.seed,
                        reference_year=settings.reference_year)
        print(format_table(table))
        res = stack_evaluate(df, ["ridge", "random_forest", "hist_gbm"], "new_item", settings.n_splits, settings.seed)
        print("\nOOF stack (new_item holdout):")
        print(res.table().round(3).to_string())
        bt = rolling_origin_backtest(make_series(seed=settings.seed))
        print("\nforecast track, SYNTHETIC dated series:")
        print(bt.summary().round(3).to_string(index=False))
    return 0


# -- parser -------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="salescope", description="Honest BigMart sales prediction and a separate forecast track.")
    p.add_argument("--env-file", default=".env", help="read settings from this file first (default .env)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synth", help="write a SYNTHETIC BigMart-like train and test file")
    s.add_argument("--out", default="data/synthetic")
    s.add_argument("--items", type=int, default=400)
    s.add_argument("--outlets", type=int, default=10)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("validate", help="check a CSV against the schema")
    s.add_argument("path")
    s.add_argument("--no-target", action="store_true", help="the file has no target (the test file)")
    s.set_defaults(func=cmd_validate)

    def common(sp, models_default: str | None = None):
        sp.add_argument("--train", help="train CSV (default: $SALESCOPE_DATA_DIR/retail_mart_train.csv)")
        sp.add_argument("--folds", type=int, default=None, help="CV folds (default: $SALESCOPE_N_SPLITS)")
        if models_default is not None:
            sp.add_argument("--models", default=models_default)

    s = sub.add_parser("evaluate", help="cross-validate models under CV schemes")
    common(s, ",".join(CORE_MODELS))
    s.add_argument("--schemes", default=",".join(SCHEMES))
    s.add_argument("--log-target", action="store_true", help="fit on log1p(sales)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("tune", help="search hyperparameters inside grouped CV, score once on a holdout")
    common(s)
    s.add_argument("--model", default="hist_gbm", choices=sorted(SEARCH_SPACES))
    s.add_argument("--scheme", default="new_outlet", choices=list(SCHEMES))
    s.add_argument("--iter", type=int, default=10)
    s.add_argument("--backend", default="random", choices=["random", "optuna"])
    s.set_defaults(func=cmd_tune)

    s = sub.add_parser("stack", help="OOF stacking, scored on a grouped holdout")
    common(s, "ridge,random_forest,hist_gbm")
    s.add_argument("--scheme", default="new_item", choices=list(SCHEMES))
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_stack)

    s = sub.add_parser("train", help="fit one model on all rows and save it")
    s.add_argument("--train")
    s.add_argument("--model", default="hist_gbm")
    s.add_argument("--log-target", action="store_true")
    s.add_argument("--out", help="output folder (default: $SALESCOPE_ARTIFACT_DIR/<model>)")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("predict", help="score a CSV without target with a saved model")
    s.add_argument("--model-dir", required=True)
    s.add_argument("--input", required=True)
    s.add_argument("--out", default="predictions.csv")
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("explain", help="permutation importance and error analysis on a grouped holdout")
    s.add_argument("--train")
    s.add_argument("--model", default="hist_gbm")
    s.add_argument("--scheme", default="new_outlet", choices=list(SCHEMES))
    s.set_defaults(func=cmd_explain)

    s = sub.add_parser("backtest", help="forecast track: rolling-origin backtest on dated series")
    s.add_argument("--series", help="long CSV with series_id,date,sales (default: SYNTHETIC series)")
    s.add_argument("--models", help=f"comma list from {sorted(FORECASTERS)}")
    s.add_argument("--horizon", type=int, default=6)
    s.add_argument("--origins", type=int, default=3)
    s.set_defaults(func=cmd_backtest)

    s = sub.add_parser("demo", help="offline demo on SYNTHETIC data: evaluate, stack, backtest")
    s.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(args.env_file)
    try:
        settings = Settings.from_env()
        return int(args.func(args, settings))
    except (ConfigError, SchemaError, NoTimeAxisError, FileNotFoundError, KeyError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
