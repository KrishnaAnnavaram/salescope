"""Load BigMart-style CSV files, normalise known label variants and validate the schema."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from salescope.schema import TARGET, validate

# The raw file spells the same fat content in five ways.
_FAT_ALIASES = {"low fat": "Low Fat", "lf": "Low Fat", "regular": "Regular", "reg": "Regular"}


def normalise_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map fat-content spelling variants to ``Low Fat`` / ``Regular`` and strip text columns."""
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].where(out[col].isna(), out[col].astype(str).str.strip())
    if "Item_Fat_Content" in out:
        fat = out["Item_Fat_Content"]
        out["Item_Fat_Content"] = fat.map(lambda v: _FAT_ALIASES.get(str(v).lower(), v) if pd.notna(v) else v)
    return out


def load_frame(path: str | Path, require_target: bool = True) -> pd.DataFrame:
    """Read a CSV, normalise labels and validate it. Raises FileNotFoundError or SchemaError."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"{p} does not exist. See data/README.md to download the data, or run `salescope synth`."
        )
    raw = pd.read_csv(p)
    return validate(normalise_labels(raw), require_target=require_target)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the features from the target. The returned X never contains the target."""
    if TARGET not in df:
        raise KeyError(f"{TARGET} is not in the frame")
    return df.drop(columns=[TARGET]), df[TARGET].astype(float)
