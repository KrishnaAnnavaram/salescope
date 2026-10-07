"""Column contract for BigMart-style files and a validator that reports every problem at once."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ID_COL = "Item_Identifier"
OUTLET_COL = "Outlet_Identifier"
TARGET = "Item_Outlet_Sales"

FAT_CONTENT = ("Low Fat", "Regular")
OUTLET_SIZES = ("Small", "Medium", "High")
LOCATION_TYPES = ("Tier 1", "Tier 2", "Tier 3")
OUTLET_TYPES = ("Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3")


@dataclass(frozen=True)
class ColumnSpec:
    kind: str  # "str", "float" or "int"
    nullable: bool = False
    pattern: str | None = None
    allowed: tuple[str, ...] | None = None
    low: float | None = None
    high: float | None = None
    low_inclusive: bool = True


FEATURE_COLUMNS: dict[str, ColumnSpec] = {
    ID_COL: ColumnSpec("str", pattern=r"^[A-Z]{3}\d{2}$"),
    "Item_Weight": ColumnSpec("float", nullable=True, low=0.0, low_inclusive=False, high=100.0),
    "Item_Fat_Content": ColumnSpec("str", allowed=FAT_CONTENT),
    "Item_Visibility": ColumnSpec("float", low=0.0, high=1.0),
    "Item_Type": ColumnSpec("str"),
    "Item_MRP": ColumnSpec("float", low=0.0, low_inclusive=False, high=10_000.0),
    OUTLET_COL: ColumnSpec("str", pattern=r"^OUT\d{3}$"),
    "Outlet_Establishment_Year": ColumnSpec("int", low=1900, high=2100),
    "Outlet_Size": ColumnSpec("str", nullable=True, allowed=OUTLET_SIZES),
    "Outlet_Location_Type": ColumnSpec("str", allowed=LOCATION_TYPES),
    "Outlet_Type": ColumnSpec("str", allowed=OUTLET_TYPES),
}
TARGET_SPEC = ColumnSpec("float", low=0.0)


class SchemaError(ValueError):
    """The frame does not match the column contract. ``problems`` lists each failed rule."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("schema validation failed:\n  - " + "\n  - ".join(problems))


@dataclass
class _Checker:
    problems: list[str] = field(default_factory=list)

    def add(self, msg: str) -> None:
        self.problems.append(msg)


def _check_column(df: pd.DataFrame, name: str, spec: ColumnSpec, chk: _Checker) -> pd.Series:
    col = df[name]
    missing = col.isna()
    if not spec.nullable and missing.any():
        chk.add(f"{name}: {int(missing.sum())} missing values (not allowed)")
    present = col[~missing]
    if spec.kind in ("float", "int"):
        num = pd.to_numeric(present, errors="coerce")
        bad = num.isna()
        if bad.any():
            chk.add(f"{name}: {int(bad.sum())} values are not numeric, first {present[bad].iloc[0]!r}")
        num = num[~bad]
        if spec.kind == "int" and len(num) and not np.all(np.isclose(num, np.round(num))):
            chk.add(f"{name}: values must be whole numbers")
        if spec.low is not None:
            below = num < spec.low if spec.low_inclusive else num <= spec.low
            if below.any():
                op = ">=" if spec.low_inclusive else ">"
                chk.add(f"{name}: {int(below.sum())} values break the rule {op} {spec.low}")
        if spec.high is not None and (num > spec.high).any():
            chk.add(f"{name}: {int((num > spec.high).sum())} values break the rule <= {spec.high}")
        out = pd.to_numeric(col, errors="coerce")
        return out.astype("Int64") if spec.kind == "int" and not out.isna().any() else out.astype(float)
    text = present.astype(str)
    if spec.pattern is not None:
        bad = ~text.str.fullmatch(spec.pattern)
        if bad.any():
            chk.add(f"{name}: {int(bad.sum())} values do not match {spec.pattern}, first {text[bad].iloc[0]!r}")
    if spec.allowed is not None:
        bad = ~text.isin(spec.allowed)
        if bad.any():
            values = sorted(set(text[bad]))[:5]
            chk.add(f"{name}: values {values} are not in {list(spec.allowed)}")
    return col.where(missing, col.astype(str))


def validate(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Return a typed copy of ``df`` or raise :class:`SchemaError` with every problem found.

    Rules: all feature columns exist, types and ranges hold, identifiers match their patterns,
    categorical values are known, and each (item, outlet) pair appears only once.
    """
    chk = _Checker()
    specs = dict(FEATURE_COLUMNS)
    if require_target:
        specs[TARGET] = TARGET_SPEC
    absent = [c for c in specs if c not in df.columns]
    if absent:
        raise SchemaError([f"missing columns: {absent}"])
    if df.empty:
        raise SchemaError(["the file has no rows"])
    out = df.copy()
    for name, spec in specs.items():
        out[name] = _check_column(df, name, spec, chk)
    dup = df.duplicated(subset=[ID_COL, OUTLET_COL])
    if dup.any():
        chk.add(f"{int(dup.sum())} duplicate ({ID_COL}, {OUTLET_COL}) pairs")
    if chk.problems:
        raise SchemaError(chk.problems)
    out["Outlet_Establishment_Year"] = out["Outlet_Establishment_Year"].astype(int)
    return out
