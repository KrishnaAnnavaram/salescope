"""Synthetic BigMart-like data with a known structure, for the offline demo and the tests.

The generator copies the column contract and the known quirks of the public file (zero visibility,
missing weights that are recoverable from other rows of the same item, outlets with no size and
five spellings of fat content). It never reads the real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from salescope.schema import ID_COL, OUTLET_COL, TARGET

_TYPES_BY_PREFIX = {
    "FD": ["Dairy", "Meat", "Fruits and Vegetables", "Snack Foods", "Frozen Foods", "Breads", "Canned",
           "Baking Goods", "Breakfast", "Seafood", "Starchy Foods"],
    "DR": ["Soft Drinks", "Hard Drinks", "Dairy"],
    "NC": ["Household", "Health and Hygiene", "Others"],
}
_PREFIX_WEIGHTS = {"FD": 0.72, "DR": 0.10, "NC": 0.18}
# demand multiplier of each item type (the model can learn it from Item_Type)
_TYPE_EFFECT = {
    "Dairy": 1.1, "Meat": 0.9, "Fruits and Vegetables": 1.25, "Snack Foods": 1.2, "Frozen Foods": 0.95,
    "Breads": 1.05, "Canned": 0.85, "Baking Goods": 0.8, "Breakfast": 0.75, "Seafood": 0.7,
    "Starchy Foods": 0.9, "Soft Drinks": 1.15, "Hard Drinks": 0.8, "Household": 1.0,
    "Health and Hygiene": 0.9, "Others": 0.7,
}

# outlet type -> (relative demand, typical sizes)
_OUTLET_PROFILES = {
    "Grocery Store": (0.18, ["Small"]),
    "Supermarket Type1": (1.0, ["Small", "Medium", "High"]),
    "Supermarket Type2": (0.95, ["Medium"]),
    "Supermarket Type3": (1.75, ["Medium"]),
}
_FAT_SPELLINGS = {"Low Fat": ["Low Fat", "Low Fat", "Low Fat", "LF", "low fat"], "Regular": ["Regular", "Regular", "reg"]}


def _items(rng: np.random.Generator, n_items: int) -> pd.DataFrame:
    prefixes = rng.choice(list(_PREFIX_WEIGHTS), size=n_items, p=list(_PREFIX_WEIGHTS.values()))
    ids: list[str] = []
    seen: set[str] = set()
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for prefix in prefixes:
        while True:
            candidate = f"{prefix}{letters[rng.integers(26)]}{rng.integers(0, 100):02d}"
            if candidate not in seen:
                seen.add(candidate)
                ids.append(candidate)
                break
    types = [rng.choice(_TYPES_BY_PREFIX[p]) for p in prefixes]
    fat = np.where(rng.random(n_items) < 0.64, "Low Fat", "Regular")
    return pd.DataFrame({
        ID_COL: ids,
        "prefix": prefixes,
        "Item_Type": types,
        "fat": fat,
        "Item_Weight": np.round(rng.uniform(4.5, 21.5, n_items), 3),
        "Item_MRP": np.round(rng.uniform(31.0, 267.0, n_items), 4),
        "base_visibility": rng.uniform(0.01, 0.16, n_items),
        "popularity": rng.lognormal(mean=0.0, sigma=0.2, size=n_items),
    })


def _outlets(rng: np.random.Generator, n_outlets: int) -> pd.DataFrame:
    types = list(_OUTLET_PROFILES)
    # guarantee each outlet type once when there is room, then sample the rest
    chosen = types[: min(n_outlets, len(types))]
    chosen += list(rng.choice(types, size=max(0, n_outlets - len(types)), p=[0.2, 0.6, 0.1, 0.1]))
    codes = rng.choice(np.arange(10, 100), size=n_outlets, replace=False)
    rows = []
    for code, otype in zip(codes, chosen):
        demand, sizes = _OUTLET_PROFILES[otype]
        rows.append({
            OUTLET_COL: f"OUT{int(code):03d}",
            "Outlet_Type": otype,
            "Outlet_Size": rng.choice(sizes),
            "Outlet_Location_Type": rng.choice(["Tier 1", "Tier 2", "Tier 3"]),
            "Outlet_Establishment_Year": int(rng.integers(1985, 2010)),
            "demand": demand * rng.uniform(0.9, 1.1),
        })
    out = pd.DataFrame(rows)
    # like the real file: some outlets never report a size
    n_unknown = max(1, n_outlets // 4)
    out.loc[rng.choice(n_outlets, size=n_unknown, replace=False), "Outlet_Size"] = np.nan
    return out


def make_dataset(
    n_items: int = 400,
    n_outlets: int = 10,
    coverage: float = 0.55,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(train, test)`` frames. ``test`` has other item-outlet pairs and no target.

    True sales = MRP x units. Units depend on outlet type demand, item popularity, outlet age and
    visibility, with multiplicative gamma noise. Zero visibility and missing weights are added after
    the target is computed, so they hide real values (as in the public file).
    """
    if not 0.05 <= coverage <= 0.95:
        raise ValueError("coverage must be between 0.05 and 0.95")
    rng = np.random.default_rng(seed)
    items = _items(rng, n_items)
    outlets = _outlets(rng, n_outlets)
    pairs = items.merge(outlets, how="cross")
    pairs = pairs[rng.random(len(pairs)) < coverage].reset_index(drop=True)

    age = 2013 - pairs["Outlet_Establishment_Year"]
    visibility = np.clip(pairs["base_visibility"] * rng.uniform(0.7, 1.3, len(pairs)), 0.003, 0.33)
    type_effect = pairs["Item_Type"].map(_TYPE_EFFECT).fillna(1.0)
    tier_effect = pairs["Outlet_Location_Type"].map({"Tier 1": 0.9, "Tier 2": 1.0, "Tier 3": 1.15})
    # an interaction that a linear model cannot express: big stores sell expensive items better
    premium = np.where((pairs["Outlet_Type"] == "Supermarket Type3") & (pairs["Item_MRP"] > 180), 1.25, 1.0)
    units = (
        14.0
        * pairs["demand"]
        * type_effect
        * tier_effect
        * premium
        * pairs["popularity"]
        * (1.0 + 0.012 * np.clip(age, 0, 30))
        * (1.0 - 1.5 * visibility)
        * np.where(pairs["prefix"] == "NC", 0.85, 1.0)
    )
    noise = rng.gamma(shape=12.0, scale=1 / 12.0, size=len(pairs))
    sales = np.round(pairs["Item_MRP"] * units * noise, 4)

    fat = [rng.choice(_FAT_SPELLINGS[f]) for f in pairs["fat"]]
    frame = pd.DataFrame({
        ID_COL: pairs[ID_COL],
        "Item_Weight": pairs["Item_Weight"],
        "Item_Fat_Content": fat,
        "Item_Visibility": np.round(visibility, 9),
        "Item_Type": pairs["Item_Type"],
        "Item_MRP": pairs["Item_MRP"],
        OUTLET_COL: pairs[OUTLET_COL],
        "Outlet_Establishment_Year": pairs["Outlet_Establishment_Year"],
        "Outlet_Size": pairs["Outlet_Size"],
        "Outlet_Location_Type": pairs["Outlet_Location_Type"],
        "Outlet_Type": pairs["Outlet_Type"],
        TARGET: sales,
    })
    frame.loc[rng.random(len(frame)) < 0.06, "Item_Visibility"] = 0.0
    frame.loc[rng.random(len(frame)) < 0.17, "Item_Weight"] = np.nan

    is_test = rng.random(len(frame)) < 0.4
    train = frame[~is_test].reset_index(drop=True)
    test = frame[is_test].drop(columns=[TARGET]).reset_index(drop=True)
    return train, test
