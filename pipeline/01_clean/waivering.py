# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: kernelspec,jupytext
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: waived-in-america (3.13)
#     language: python
#     name: python3
# ---

# %% [markdown]
# This notebook cleans and merges the waiver nonavailability procurement data from madeinamerica.gov.

# %%
import sys
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    try:
        start = Path(__file__).resolve().parent  # script execution
    except NameError:
        start = Path.cwd()  # notebook kernel
    p = start
    while not (p / "pyproject.toml").exists():
        if p.parent == p:
            raise FileNotFoundError(f"repo root (pyproject.toml) not found above {start}")
        p = p.parent
    return p


sys.path.insert(0, str(_repo_root() / "pipeline" / "lib"))
from paths import OUTPUT, PROCUREMENT_PARQUET, RAW_WAIVERS, WAIVERS_CLEANED

OUTPUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(RAW_WAIVERS)

# %%
# Drop columns that don't have any values below the header
df = df.dropna(axis=1, how='all', subset=df.index[1:])

# %%
# Convert created column to datetime
df["created"] = pd.to_datetime(
    df["created"],
    format="mixed",   # created timestamps arrive in inconsistent formats
    utc=True,
    errors="raise"
)

df["fiscal_year"] = df["created"].dt.year + (df["created"].dt.month >= 10).astype(int)

fy = df.pop("fiscal_year")
df.insert(3, "fiscal_year", fy)


# %%
# How many rows have empty PIIDs?
n_missing_piid = df["Procurement_Instrument_Identifier(s)_(PIID)_for_this_waiver_(if_applicable)"].isna().sum()
print(f"Waivers without a PIID: {n_missing_piid} of {len(df)}")

# %% [markdown]
# For rows with PIIDs, get total_dollars_obligated so we can compare dollar values

# %%
# import procurement parquet data into polars DataFrame
import polars as pl

procurement_df = pl.scan_parquet(PROCUREMENT_PARQUET)

# %%
# Convert the waivers df to polars DataFrame
waivers_pl_df = pl.from_pandas(df)

# %%
# For each waiver, find matching procurement(s) based on Solicitation_ID
# and get federal_action_obligation and total_dollars_obligated and place_of_manufacture to the waiver DataFrame
# Noting that some waivers may not have a Solicitation_ID, so we do a left join with Polars

# Aggregate to one row per PIID before joining: a Solicitation_ID that
# matches multiple FPDS modifications would otherwise fan out into
# duplicate waiver rows. Per award: total obligated = sum of per-action
# obligations; total_dollars_obligated = max; place_of_manufacture =
# first reported value.
procurement_selected = (
    procurement_df.select(
        ["award_id_piid", "federal_action_obligation", "total_dollars_obligated", "place_of_manufacture"]
    )
    .group_by("award_id_piid")
    .agg(
        pl.col("federal_action_obligation").cast(pl.Float64).sum(),
        pl.col("total_dollars_obligated").cast(pl.Float64).max(),
        pl.col("place_of_manufacture").drop_nulls().first(),
    )
)

waiver_with_procurement = (
    waivers_pl_df.lazy()
    .join(
        procurement_selected,
        left_on="Solicitation_ID",
        right_on="award_id_piid",
        how="left",
        suffix="_procurement",
    )
    .collect()
)

# %%
# Export the cleaned DataFrame to a new CSV file
waiver_with_procurement.write_csv(WAIVERS_CLEANED)

