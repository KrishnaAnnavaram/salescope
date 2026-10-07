# Data

The repository does not contain a data file. Git ignores everything in `/data/` except this README.

## Track A: BigMart sales (tabular regression)

| Item | Value |
|---|---|
| Source | "Big Mart Sales" practice problem. Kaggle mirrors it, for example `https://www.kaggle.com/datasets/lokeshmendake/big-mart-sales-dataset` |
| License and terms | The terms of the Kaggle page and of the original practice problem. Read them before you use the data. Do not commit the files |
| Expected files | `data/retail_mart_train.csv` (8,523 rows, with target) and `data/retail_mart_test.csv` (5,681 rows, no target) |
| Download | Download the train and test CSV from the Kaggle page. Rename them to the two file names above, or set `SALESCOPE_DATA_DIR` |

Expected columns (the validator in `src/salescope/schema.py` checks each rule):

| Column | Type | Rule |
|---|---|---|
| `Item_Identifier` | text | Pattern `^[A-Z]{3}\d{2}$`. The first two letters give the category: `FD`, `DR`, `NC` |
| `Item_Weight` | number | More than 0. Can be empty |
| `Item_Fat_Content` | text | `Low Fat` or `Regular`. The loader changes `LF`, `low fat` and `reg` |
| `Item_Visibility` | number | 0 to 1. A value of 0 means "not recorded" |
| `Item_Type` | text | Not empty |
| `Item_MRP` | number | More than 0 |
| `Outlet_Identifier` | text | Pattern `^OUT\d{3}$` |
| `Outlet_Establishment_Year` | whole number | 1900 to 2100 |
| `Outlet_Size` | text | `Small`, `Medium` or `High`. Can be empty |
| `Outlet_Location_Type` | text | `Tier 1`, `Tier 2` or `Tier 3` |
| `Outlet_Type` | text | `Grocery Store` or `Supermarket Type1` / `Type2` / `Type3` |
| `Item_Outlet_Sales` | number | 0 or more. Only in the train file |

Each (`Item_Identifier`, `Outlet_Identifier`) pair must occur only once. The data has **no dates**.

## Track B: dated retail series (forecasting)

The forecast track needs a long CSV with the columns `series_id`, `date` and `sales`.
Each series must be a gap-free monthly series with month-start dates.
Examples of dated sources: the M5 forecasting data (Walmart, Kaggle competition terms) or the Walmart store sales data (Kaggle terms).
Aggregate them to monthly sales for each store before you use them.

## Synthetic data (no download)

```bash
salescope synth --out data/synthetic        # BigMart-like train and test files with the same quirks
salescope backtest                          # uses SYNTHETIC dated series made in memory
```

The synthetic generator (`src/salescope/synthetic.py`) never reads the real data. The tests and the offline demo use only synthetic data.
