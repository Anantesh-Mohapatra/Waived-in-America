# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
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
# # Treatment NSN Identification & Transaction Database
#
# **Purpose**: Identifies which waiver NSNs qualify as treatment units and
# builds their transaction database.
#
# **Method**: BidLink procurement history only. No CAGE+PSC matching.
#
# **Treatment eligibility criteria**:
# 1. NSN appears in `waiver_nsn_reference.csv` (has a public waiver on madeinamerica.gov)
# 2. NSN has at least one BidLink line-item transaction
# 3. NSN has at least one BidLink transaction *before* its earliest waiver date
# 4. NSN has at least one BidLink transaction after its earliest waiver date
#
# Pre-2021 transaction counts are reported for observation but are NOT a filter.
#
# **Outputs**:
# - `treatment_nsns`: final list of treatment-eligible NSNs
# - `treatment_panel`: all BidLink transactions for treatment NSNs (FY2017+),
#   left-joined to procurement parquet
# - `results/treatment/treatment_nsn_panel.csv`: exported panel for downstream scripts

# %%
from pathlib import Path
from datetime import timedelta

import polars as pl

# Use ASCII table formatting to avoid cp1252 encoding issues on Windows
pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(14)

# %%
MIN_PROXY_PRICE_OBS = 2

# %%
# --- Resolve project root (script or notebook kernel) ---
import sys

try:
    _start = Path(__file__).resolve().parent
except NameError:
    _start = Path.cwd()
ROOT = _start
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise FileNotFoundError(f"repo root (pyproject.toml) not found above {_start}")
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT / "pipeline" / "lib"))
from paths import (
    NSN_REFERENCE,
    PROCUREMENT_PARQUET,
    RAW_BIDLINK,
    RESULTS,
    TREATMENT_DATES,
    UNIT_COST_CLIN_TAGGING,
)

print(f"Project root: {ROOT}")

# %% [markdown]
# ---
# ## 1. Load the Waiver NSN Universe
#
# Starting point: the 37 unique NSNs extracted from madeinamerica.gov waivers.
# Each has been validated and classified in `waiver_nsn_reference.csv`.

# %%
nsn_ref = pl.read_csv(
    NSN_REFERENCE / "waiver_nsn_reference.csv",
    infer_schema_length=0,
)
print(f"Waiver NSN universe: {len(nsn_ref)} NSNs")
nsn_ref

# %% [markdown]
# ---
# ## 2. Extract Earliest Waiver Date per NSN
#
# From `waiver_dod_identifiers_manual.csv`, which maps every identifier
# (NSN or part number) to its waiver record and posting timestamp (`created`).
# We filter to NSN-type identifiers and take the earliest `created` date per NSN.

# %%
identifiers = pl.read_csv(
    NSN_REFERENCE / "waiver_dod_identifiers_manual.csv",
    infer_schema_length=0,
)

# Keep only NSN identifiers, parse date, get earliest per NSN
waiver_dates = (
    identifiers
    .filter(pl.col("identifier_type") == "nsn")
    .with_columns(
        pl.col("created").str.slice(0, 10).str.to_date("%Y-%m-%d").alias("waiver_date")
    )
    .group_by("identifier")
    .agg(
        pl.col("waiver_date").min().alias("earliest_waiver_date"),
        pl.col("waiver_date").max().alias("latest_waiver_date"),
        pl.col("waiver_id").n_unique().alias("n_waiver_records"),
        pl.col("waiver_title").first().alias("waiver_title"),
        pl.col("contracting_agency").first().alias("contracting_agency"),
    )
    .rename({"identifier": "nsn"})
    .sort("earliest_waiver_date")
)

print(f"NSNs with waiver dates: {len(waiver_dates)}")

# %% [markdown]
# ### Waiver NSN Universe with Dates
#
# Join reference table with waiver dates. Every NSN should have a date.

# %%
universe = (
    nsn_ref
    .join(waiver_dates, on="nsn", how="left")
    .sort("earliest_waiver_date")
)

# Check for any NSNs missing dates
missing_dates = universe.filter(pl.col("earliest_waiver_date").is_null())
if len(missing_dates) > 0:
    print(f"WARNING: {len(missing_dates)} NSNs have no waiver date in identifiers file:")
    print(missing_dates.select("nsn", "nsn_formatted", "classification", "example_waiver_title"))
else:
    print("All NSNs have waiver dates.")

print()
universe.select(
    "nsn_formatted", "fsc", "classification",
    "earliest_waiver_date", "n_waiver_records",
    "contracting_agency", "waiver_title",
)

# %% [markdown]
# ---
# ## 3. Load BidLink Procurement History
#
# Load ALL BidLink NSN line-item files. These are the only source of
# NSN-level procurement transactions we use.
#
# **Important**: We load everything first, then filter to waiver NSNs.
# This lets us see exactly which NSNs have data and which don't.

# %%
import glob as _glob

bidlink_files = sorted(
    _glob.glob(str(RAW_BIDLINK / "nsn" / "procurement_history_line_items" / "*.csv"))
)
print(f"BidLink line-item files found: {len(bidlink_files)}")

# Load and concatenate all files
dfs = []
for f in bidlink_files:
    df = pl.read_csv(f, infer_schema_length=1000, infer_schema=False)
    dfs.append(df)

bidlink_all = pl.concat(dfs, how="diagonal")
print(f"Total BidLink rows (all NSNs): {len(bidlink_all)}")

# Standardize column names
bidlink_all = bidlink_all.rename({c: c.strip() for c in bidlink_all.columns})

# Parse date
bidlink_all = bidlink_all.with_columns(
    pl.col("Date").str.strip_chars().str.to_date("%Y/%m/%d", strict=False).alias("date_parsed"),
)

# Show unique NSNs in BidLink
bidlink_nsns = bidlink_all["NSN"].unique().sort().to_list()
print(f"Unique NSNs in BidLink: {len(bidlink_nsns)}")

# %% [markdown]
# ### 3a. Which waiver NSNs have BidLink data?
#
# Match using the formatted NSN (dashed format, e.g. `1660-01-600-4909`).

# %%
# Set of formatted waiver NSNs
waiver_formatted = set(nsn_ref["nsn_formatted"].to_list())

# Which waiver NSNs appear in BidLink?
found_in_bidlink = sorted(set(bidlink_nsns) & waiver_formatted)
not_in_bidlink = sorted(waiver_formatted - set(bidlink_nsns))

print(f"Waiver NSNs WITH BidLink data: {len(found_in_bidlink)}")
print(f"Waiver NSNs WITHOUT BidLink data: {len(not_in_bidlink)}")

# %% [markdown]
# ### NSNs DROPPED: No BidLink Data
#
# These NSNs have public waivers but zero rows in BidLink procurement history.
# They cannot be treatment units because we have no procurement data to analyze.

# %%
dropped_no_bidlink = (
    universe
    .filter(pl.col("nsn_formatted").is_in(not_in_bidlink))
    .select(
        "nsn_formatted", "fsc", "classification",
        "earliest_waiver_date", "contracting_agency", "waiver_title",
    )
)
print(f"DROPPED (no BidLink data): {len(dropped_no_bidlink)} NSNs")
dropped_no_bidlink

# %% [markdown]
# ### NSNs Remaining After BidLink Filter

# %%
remaining_after_bidlink = (
    universe
    .filter(pl.col("nsn_formatted").is_in(found_in_bidlink))
    .select(
        "nsn_formatted", "fsc", "classification",
        "earliest_waiver_date", "n_waiver_records",
        "contracting_agency", "waiver_title",
    )
)
print(f"Remaining: {len(remaining_after_bidlink)} NSNs with BidLink data")
remaining_after_bidlink

# %% [markdown]
# ---
# ## 4. Compute Pre/Post-Waiver Transaction Counts
#
# For each remaining NSN, count BidLink transactions in three windows:
# - **Pre-waiver**: before the earliest waiver date
# - **Post-waiver**: after the earliest waiver date
# - **Pre-2021** (observational): before January 1, 2021

# %%
from datetime import date

# Filter BidLink to waiver NSNs only
bidlink_waiver = bidlink_all.filter(pl.col("NSN").is_in(found_in_bidlink))
print(f"BidLink rows for waiver NSNs: {len(bidlink_waiver)}")

# Join waiver dates (need 13-digit NSN for the join)
nsn_format_map = nsn_ref.select("nsn", "nsn_formatted")
bidlink_waiver = (
    bidlink_waiver
    .join(
        nsn_format_map,
        left_on="NSN",
        right_on="nsn_formatted",
        how="left",
    )
    .join(
        waiver_dates.select("nsn", "earliest_waiver_date"),
        on="nsn",
        how="left",
    )
)

# Compute time-window flags
bidlink_waiver = bidlink_waiver.with_columns(
    (pl.col("date_parsed") < pl.col("earliest_waiver_date")).alias("is_pre_waiver"),
    (pl.col("date_parsed") > pl.col("earliest_waiver_date")).alias("is_post_waiver_any"),
    (pl.col("date_parsed") < date(2021, 1, 1)).alias("is_pre_2021"),
)

# Aggregate counts per NSN
nsn_counts = (
    bidlink_waiver
    .group_by("NSN")
    .agg(
        pl.len().alias("total_bidlink_rows"),
        pl.col("is_pre_waiver").sum().alias("n_pre_waiver"),
        pl.col("is_post_waiver_any").sum().alias("n_post_waiver_any"),
        pl.col("is_pre_2021").sum().alias("n_pre_2021"),
        pl.col("date_parsed").min().alias("earliest_transaction"),
        pl.col("date_parsed").max().alias("latest_transaction"),
        pl.col("earliest_waiver_date").first().alias("waiver_date"),
    )
    .sort("NSN")
)

# %% [markdown]
# ### Full Transaction Count Table
#
# Every waiver NSN with BidLink data. The eligibility filters
# (pre_waiver > 0 AND eligible post-waiver > 0) have NOT been applied yet.

# %%
# Enrich with metadata for display
count_display = (
    nsn_counts
    .join(
        nsn_ref.select("nsn_formatted", "fsc", "classification"),
        left_on="NSN",
        right_on="nsn_formatted",
        how="left",
    )
    .select(
        "NSN", "fsc", "classification",
        "waiver_date",
        "total_bidlink_rows",
        "n_pre_waiver", "n_post_waiver_any", "n_pre_2021",
        "earliest_transaction", "latest_transaction",
    )
    .sort("NSN")
)

print("Transaction counts for all waiver NSNs with BidLink data:")
count_display

# %% [markdown]
# ---
# ## 5. Apply Treatment Eligibility Filters
#
# **Filter 1**: `n_pre_waiver > 0` -- must have purchases before the waiver
#
# **Filter 2**: `n_post_waiver_any > 0` -- must have purchases after the waiver date
#
# NSNs failing either filter are shown below with their characteristics.

# %%
# Identify failures
fails_pre = nsn_counts.filter(pl.col("n_pre_waiver") == 0)
fails_post = nsn_counts.filter(
    (pl.col("n_pre_waiver") > 0) & (pl.col("n_post_waiver_any") == 0)
)

# %% [markdown]
# ### NSNs DROPPED: No Pre-Waiver Transactions
#
# These NSNs only have transactions on or after their waiver date.
# Without a pre-period, we cannot measure a treatment effect.

# %%
if len(fails_pre) > 0:
    dropped_pre = (
        fails_pre
        .join(nsn_ref.select("nsn_formatted", "fsc"), left_on="NSN", right_on="nsn_formatted", how="left")
        .join(waiver_dates.select("nsn", "waiver_title", "contracting_agency"),
              left_on=pl.col("NSN").str.replace_all("-", "", literal=True),
              right_on="nsn", how="left")
        .select(
            "NSN", "fsc", "waiver_date",
            "total_bidlink_rows", "n_pre_waiver", "n_post_waiver_any",
            "earliest_transaction", "latest_transaction",
            "contracting_agency", "waiver_title",
        )
    )
    print(f"DROPPED (no pre-waiver transactions): {len(dropped_pre)} NSNs")
    print(dropped_pre)
else:
    print("No NSNs dropped for lacking pre-waiver transactions.")

# %% [markdown]
# ### NSNs DROPPED: No Post-Waiver Transactions
#
# These NSNs have pre-waiver data but no transactions after their waiver date,
# so we cannot observe a post-treatment outcome.

# %%
if len(fails_post) > 0:
    dropped_post = (
        fails_post
        .join(nsn_ref.select("nsn_formatted", "fsc"), left_on="NSN", right_on="nsn_formatted", how="left")
        .join(waiver_dates.select("nsn", "waiver_title", "contracting_agency"),
              left_on=pl.col("NSN").str.replace_all("-", "", literal=True),
              right_on="nsn", how="left")
        .select(
            "NSN", "fsc", "waiver_date",
            "total_bidlink_rows", "n_pre_waiver", "n_post_waiver_any",
            "earliest_transaction", "latest_transaction",
            "contracting_agency", "waiver_title",
        )
    )
    print(f"DROPPED (no post-waiver transactions): {len(dropped_post)} NSNs")
    print(dropped_post)
else:
    print("No NSNs dropped for lacking post-waiver transactions.")

# %% [markdown]
# ---
# ## 6. Final Treatment NSN List
#
# These NSNs pass all filters: they have a public waiver, BidLink data,
# pre-waiver transactions, and eligible post-waiver transactions.

# %%
treatment_nsns_df = (
    nsn_counts
    .filter(
        (pl.col("n_pre_waiver") > 0) & (pl.col("n_post_waiver_any") > 0)
    )
    .join(
        nsn_ref.select("nsn_formatted", "fsc", "classification"),
        left_on="NSN",
        right_on="nsn_formatted",
        how="left",
    )
    .join(
        waiver_dates.select("nsn", "waiver_title", "contracting_agency"),
        left_on=pl.col("NSN").str.replace_all("-", "", literal=True),
        right_on="nsn",
        how="left",
    )
    .select(
        "NSN", "fsc", "classification",
        "waiver_date", "waiver_title", "contracting_agency",
        "total_bidlink_rows",
        "n_pre_waiver", "n_post_waiver_any", "n_pre_2021",
        "earliest_transaction", "latest_transaction",
    )
    .sort("NSN")
)

treatment_nsns = treatment_nsns_df["NSN"].to_list()

n_container = sum(1 for n in treatment_nsns if n.startswith("8150"))
n_non_container = len(treatment_nsns) - n_container

print(f"FINAL TREATMENT NSNs: {len(treatment_nsns)}")
print(f"  Non-container: {n_non_container}")
print(f"  Container (8150): {n_container}")
print()
treatment_nsns_df

# %% [markdown]
# ### Drop Summary
#
# A complete accounting of all 37 waiver NSNs and their disposition.

# %%
# Build disposition table
disposition = (
    nsn_ref
    .select("nsn_formatted", "fsc", "classification")
    .with_columns(
        pl.when(pl.col("nsn_formatted").is_in(treatment_nsns))
        .then(pl.lit("TREATMENT"))
        .when(pl.col("nsn_formatted").is_in(not_in_bidlink))
        .then(pl.lit("DROPPED: no BidLink data"))
        .when(pl.col("nsn_formatted").is_in(fails_pre["NSN"].to_list()))
        .then(pl.lit("DROPPED: no pre-waiver txns"))
        .when(pl.col("nsn_formatted").is_in(fails_post["NSN"].to_list()))
        .then(pl.lit("DROPPED: no post-waiver txns"))
        .otherwise(pl.lit("UNKNOWN"))
        .alias("disposition")
    )
    .sort("disposition", "nsn_formatted")
)

print("Disposition of all 37 waiver NSNs:")
print(disposition.group_by("disposition").len().sort("disposition"))
print()
disposition

# %% [markdown]
# ---
# ## 7. Build Treatment Transaction Database
#
# All BidLink line-item transactions for the treatment NSNs, filtered to
# FY2017+ (Oct 2016 onward), left-joined to the procurement parquet.

# %%
# Filter BidLink to treatment NSNs and compute fiscal year
treatment_bidlink = (
    bidlink_all
    .filter(pl.col("NSN").is_in(treatment_nsns))
    .rename({c: f"bl_{c.strip().lower()}" for c in bidlink_all.columns})
    .with_columns(
        pl.col("bl_date").str.strip_chars().str.to_date("%Y/%m/%d", strict=False).alias("bl_date_parsed"),
    )
    .with_columns(
        pl.col("bl_date_parsed").dt.year().alias("_yr"),
        pl.col("bl_date_parsed").dt.month().alias("_mo"),
    )
    .with_columns(
        pl.when(pl.col("_mo") >= 10)
        .then(pl.col("_yr") + 1)
        .otherwise(pl.col("_yr"))
        .alias("bl_fiscal_year")
    )
    .drop(["_yr", "_mo"])
)

# Filter to FY2017+ (i.e. dates on or after Oct 1, 2016)
fy_cutoff = date(2016, 10, 1)
pre_cutoff = treatment_bidlink.filter(pl.col("bl_date_parsed") < fy_cutoff)
treatment_bidlink = treatment_bidlink.filter(pl.col("bl_date_parsed") >= fy_cutoff)

print(f"BidLink rows for treatment NSNs (all time): {len(treatment_bidlink) + len(pre_cutoff)}")
print(f"  Dropped (pre-FY2017): {len(pre_cutoff)}")
print(f"  Kept (FY2017+): {len(treatment_bidlink)}")

# %%
# NSN group label
treatment_bidlink = treatment_bidlink.with_columns(
    pl.when(pl.col("bl_nsn").str.starts_with("8150"))
    .then(pl.lit("container"))
    .otherwise(pl.lit("non_container"))
    .alias("nsn_group")
)

# Quick summary
print("\nBidLink rows by NSN (FY2017+):")
(
    treatment_bidlink
    .group_by("bl_nsn")
    .agg(
        pl.len().alias("n_rows"),
        pl.col("bl_date_parsed").min().alias("earliest"),
        pl.col("bl_date_parsed").max().alias("latest"),
        pl.col("bl_fiscal_year").min().alias("fy_min"),
        pl.col("bl_fiscal_year").max().alias("fy_max"),
    )
    .sort("bl_nsn")
)

# %% [markdown]
# ### 7a. Load Procurement Parquet and Join

# %%
PARQUET_COLS = [
    "award_id_piid",
    "contract_transaction_unique_key",
    "modification_number",
    "action_date",
    "action_date_fiscal_year",
    "federal_action_obligation",
    "total_dollars_obligated",
    "base_and_all_options_value",
    "recipient_name",
    "cage_code",
    "awarding_agency_name",
    "awarding_sub_agency_name",
    "product_or_service_code",
    "product_or_service_code_description",
    "naics_code",
    "transaction_description",
    "country_of_product_or_service_origin_code",
    "country_of_product_or_service_origin",
    "place_of_manufacture_code",
    "place_of_manufacture",
    "extent_competed_code",
    "extent_competed",
    "number_of_offers_received",
    "award_type_code",
    "award_type",
]

# Filter parquet to contracts appearing in BidLink treatment data
bidlink_contracts = treatment_bidlink["bl_contract"].unique().to_list()

parquet_filtered = (
    pl.scan_parquet(PROCUREMENT_PARQUET)
    .select(PARQUET_COLS)
    .filter(pl.col("award_id_piid").is_in(bidlink_contracts))
    .collect(engine="streaming")
)
print(f"Parquet rows matching BidLink contracts: {len(parquet_filtered)}")

# Rename parquet cols with proc_ prefix
parquet_renamed = parquet_filtered.rename(
    {c: f"proc_{c}" for c in parquet_filtered.columns if c != "award_id_piid"}
)

# %%
# Left join: BidLink -> parquet on contract PIID
treatment_panel = treatment_bidlink.join(
    parquet_renamed,
    left_on="bl_contract",
    right_on="award_id_piid",
    how="left",
)

treatment_panel = treatment_panel.with_columns(
    pl.col("proc_contract_transaction_unique_key").is_not_null().alias("matched_to_parquet")
)

# Add treatment-timing context so downstream consumers can use the exported panel
# without recomputing the first-waiver cutoff logic.
treatment_dates_lookup = treatment_nsns_df.select(
    pl.col("NSN").alias("bl_nsn"),
    pl.col("waiver_date").alias("first_waiver_date"),
)

treatment_panel = (
    treatment_panel
    .join(treatment_dates_lookup, on="bl_nsn", how="left")
    .with_columns(
        (pl.col("bl_date_parsed") < pl.col("first_waiver_date")).alias("is_pre_first_waiver")
    )
)

print(f"\nJoined panel rows: {len(treatment_panel)}")
print(f"Unique BidLink rows: {len(treatment_bidlink)}")

# When BidLink records only zero prices, allow obligation/qty reconstruction
# for contracts manually tagged as safe in `unit_cost_clin_tagging.csv`.
# To avoid treating a single reconstructed contract as an NSN-level price
# series, only write proxy prices into the existing `bl_price` field for NSNs
# with at least `MIN_PROXY_PRICE_OBS` valid reconstructed contracts.
treatment_panel = treatment_panel.with_columns(
    pl.col("bl_price")
    .cast(pl.Utf8)
    .str.strip_chars()
    .str.replace_all(r"[\$,]", "")
    .cast(pl.Float64, strict=False)
    .alias("bl_price_f"),
    pl.col("bl_qty")
    .cast(pl.Utf8)
    .str.strip_chars()
    .str.replace_all(",", "")
    .cast(pl.Float64, strict=False)
    .alias("bl_qty_f"),
)

treatment_panel = treatment_panel.with_columns(
    (
        pl.col("bl_price_f").is_not_null()
        & (pl.col("bl_price_f") > 0)
        & (pl.col("bl_price_f") != 0.01)
    ).alias("has_valid_direct_bidlink_unit_price"),
)

contract_total_qty = (
    treatment_panel
    .select("bl_nsn", "bl_contract", "bl_order", "bl_line", "bl_date", "bl_qty_f")
    .unique(subset=["bl_nsn", "bl_contract", "bl_order", "bl_line", "bl_date"])
    .filter(pl.col("bl_qty_f").is_not_null() & (pl.col("bl_qty_f") > 0))
    .group_by(["bl_nsn", "bl_contract"])
    .agg(pl.col("bl_qty_f").sum().alias("contract_total_bidlink_qty"))
)

contract_mod0_awards = (
    treatment_panel
    .filter(pl.col("matched_to_parquet") & pl.col("proc_contract_transaction_unique_key").is_not_null())
    .filter(pl.col("proc_modification_number").cast(pl.Utf8) == "0")
    .unique(subset=["bl_nsn", "proc_contract_transaction_unique_key"])
    .with_columns(
        pl.col("proc_federal_action_obligation")
        .cast(pl.Float64, strict=False)
        .alias("contract_mod0_obligation")
    )
    .group_by(["bl_nsn", "bl_contract"])
    .agg(pl.col("contract_mod0_obligation").first().alias("contract_mod0_obligation"))
)

proxy_price_targets = (
    treatment_panel
    .filter(pl.col("matched_to_parquet"))
    .filter(pl.col("proc_modification_number").cast(pl.Utf8) == "0")
    .with_columns(pl.col("bl_line").cast(pl.Utf8).alias("bl_line_text"))
    .group_by(["bl_nsn", "bl_contract"])
    .agg(pl.col("bl_line_text").sort().first().alias("proxy_target_line"))
)

clin_tags = (
    pl.read_csv(
        UNIT_COST_CLIN_TAGGING,
        infer_schema_length=0,
    )
    .with_columns(pl.col("clin_tag").cast(pl.Int32, strict=False))
    .rename({"nsn": "bl_nsn", "piid": "bl_contract"})
)

contract_proxy_prices = (
    contract_mod0_awards
    .join(contract_total_qty, on=["bl_nsn", "bl_contract"], how="left")
    .join(clin_tags, on=["bl_nsn", "bl_contract"], how="left")
    .with_columns(
        pl.col("clin_tag").is_in([1, 2]).alias("proxy_contract_tag_valid")
    )
    .with_columns(
        pl.when(
            pl.col("proxy_contract_tag_valid")
            & pl.col("contract_total_bidlink_qty").is_not_null()
            & (pl.col("contract_total_bidlink_qty") > 0)
            & pl.col("contract_mod0_obligation").is_not_null()
            & (pl.col("contract_mod0_obligation") > 0)
        )
        .then(pl.col("contract_mod0_obligation") / pl.col("contract_total_bidlink_qty"))
        .otherwise(None)
        .alias("proxy_unit_price")
    )
)

proxy_nsn_coverage = (
    contract_proxy_prices
    .group_by("bl_nsn")
    .agg(
        pl.col("proxy_unit_price").is_not_null().sum().alias("n_proxy_contract_prices"),
    )
    .with_columns(
        (pl.col("n_proxy_contract_prices") >= MIN_PROXY_PRICE_OBS).alias("proxy_price_nsn_eligible")
    )
)

treatment_panel = (
    treatment_panel
    .join(
        contract_proxy_prices.select([
            "bl_nsn",
            "bl_contract",
            "contract_total_bidlink_qty",
            "contract_mod0_obligation",
            "clin_tag",
            "proxy_contract_tag_valid",
            "proxy_unit_price",
        ]),
        on=["bl_nsn", "bl_contract"],
        how="left",
    )
    .join(proxy_price_targets, on=["bl_nsn", "bl_contract"], how="left")
    .join(proxy_nsn_coverage, on="bl_nsn", how="left")
    .with_columns(
        pl.col("n_proxy_contract_prices").fill_null(0),
        pl.col("proxy_price_nsn_eligible").fill_null(False),
        (pl.col("bl_line").cast(pl.Utf8) == pl.col("proxy_target_line")).alias("is_contract_base_row"),
    )
    .with_columns(
        pl.when(
            ~pl.col("has_valid_direct_bidlink_unit_price")
            & pl.col("proxy_price_nsn_eligible")
            & pl.col("proxy_unit_price").is_not_null()
            & pl.col("is_contract_base_row")
        )
        .then(pl.col("proxy_unit_price"))
        .otherwise(pl.col("bl_price_f"))
        .alias("resolved_bl_price_f"),
        pl.when(
            ~pl.col("has_valid_direct_bidlink_unit_price")
            & pl.col("proxy_price_nsn_eligible")
            & pl.col("proxy_unit_price").is_not_null()
            & pl.col("is_contract_base_row")
        )
        .then(pl.format("{}", pl.col("proxy_unit_price")))
        .otherwise(pl.col("bl_price").cast(pl.Utf8))
        .alias("bl_price"),
    )
)

analysis_price_summary = (
    treatment_panel
    .group_by(["bl_nsn", "bl_contract"])
    .agg(
        pl.col("resolved_bl_price_f").max().alias("resolved_contract_price"),
        pl.col("proxy_unit_price").max().alias("proxy_unit_price"),
    )
    .group_by("bl_nsn")
    .agg(
        (
            pl.col("resolved_contract_price").is_not_null()
            & (pl.col("resolved_contract_price") > 0)
        ).sum().alias("n_analysis_price_contracts"),
        pl.col("proxy_unit_price").is_not_null().sum().alias("n_proxy_price_contracts"),
    )
    .sort("bl_nsn")
)

print("\nAnalysis-ready unit-price coverage by NSN:")
print(analysis_price_summary)

# %% [markdown]
# ### 7b. Derived Event-Year Treatment Panel
#
# Build an analysis-ready treatment summary at NSN x event-year grain while
# preserving `treatment_panel` as the raw transaction-level export.
#
# Event-year is based on calendar-date windows anchored to each NSN's first
# waiver date:
# - `event_year = -1`: 1 to 365 days before the waiver date
# - `event_year = 0`: waiver date through 364 days after the waiver date
# - `event_year = 1`: 365 to 729 days after the waiver date
#
# This means transactions in the same calendar year as the waiver can be split:
# a January purchase before a March waiver is pre-waiver, while an April purchase
# after that waiver is event-year 0.

# %%
def event_day_expr(date_col: str) -> pl.Expr:
    return (pl.col(date_col) - pl.col("first_waiver_date")).dt.total_days()


def event_year_expr(date_col: str) -> pl.Expr:
    days = event_day_expr(date_col)
    return (
        pl.when(days < 0)
        .then(-(((-days - 1) // 365) + 1))
        .otherwise(days // 365)
        .cast(pl.Int64)
    )


bidlink_event_rows = (
    treatment_panel
    .select(
        "bl_nsn", "nsn_group", "first_waiver_date",
        "bl_date", "bl_contract", "bl_order", "bl_line",
        "bl_date_parsed", "bl_fiscal_year", "bl_price", "bl_qty",
    )
    .unique(subset=["bl_nsn", "bl_contract", "bl_order", "bl_line", "bl_date"])
    .with_columns(
        pl.col("bl_price")
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.replace_all(r"[\$,]", "")
        .cast(pl.Float64, strict=False)
        .alias("bl_price_f"),
        pl.col("bl_qty")
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.replace_all(",", "")
        .cast(pl.Float64, strict=False)
        .alias("bl_qty_f"),
        event_day_expr("bl_date_parsed").alias("event_day"),
        event_year_expr("bl_date_parsed").alias("event_year"),
    )
    .with_columns(
        (
            pl.col("bl_price_f").is_not_null()
            & (pl.col("bl_price_f") > 0)
            & (pl.col("bl_price_f") != 0.01)
        ).alias("is_valid_bl_price"),
        pl.when(pl.col("event_day") < 0)
        .then(pl.lit("pre"))
        .when(pl.col("event_day") == 0)
        .then(pl.lit("waiver_day"))
        .otherwise(pl.lit("post"))
        .alias("event_period"),
    )
)

event_keys = ["bl_nsn", "nsn_group", "first_waiver_date", "event_year"]

bidlink_event_features = (
    bidlink_event_rows
    .group_by(event_keys)
    .agg(
        pl.col("event_day").min().alias("min_event_day"),
        pl.col("event_day").max().alias("max_event_day"),
        pl.col("bl_date_parsed").min().alias("first_transaction_date"),
        pl.col("bl_date_parsed").max().alias("last_transaction_date"),
        pl.len().alias("n_bidlink_transactions"),
        pl.col("bl_contract").n_unique().alias("n_bidlink_contracts"),
        pl.col("bl_qty_f").sum().alias("total_bidlink_quantity"),
        pl.col("is_valid_bl_price").sum().alias("n_valid_bidlink_prices"),
        pl.col("bl_price_f").filter(pl.col("is_valid_bl_price")).mean().alias("mean_bidlink_unit_price"),
        pl.col("bl_price_f").filter(pl.col("is_valid_bl_price")).median().alias("median_bidlink_unit_price"),
        pl.col("bl_price_f").filter(pl.col("is_valid_bl_price")).log().mean().alias("mean_log_bidlink_unit_price"),
        (pl.col("event_period") == "pre").sum().alias("n_pre_transactions"),
        (pl.col("event_period") == "waiver_day").sum().alias("n_waiver_day_transactions"),
        (pl.col("event_period") == "post").sum().alias("n_post_transactions"),
    )
)

# Procurement attributes can have action dates that differ from the BidLink line
# date after the contract-level join, so place procurement-side features into
# event-years using `proc_action_date`, not `bl_date_parsed`.
proc_event_rows = (
    treatment_panel
    .filter(pl.col("matched_to_parquet") & pl.col("proc_contract_transaction_unique_key").is_not_null())
    .filter(pl.col("proc_modification_number").cast(pl.Utf8) == "0")
    .unique(subset=["bl_nsn", "proc_contract_transaction_unique_key"])
    .with_columns(
        pl.col("proc_action_date").cast(pl.Date, strict=False).alias("proc_action_date_parsed")
    )
    .filter(pl.col("proc_action_date_parsed").is_not_null())
    .with_columns(
        event_day_expr("proc_action_date_parsed").alias("proc_event_day"),
        event_year_expr("proc_action_date_parsed").alias("event_year"),
        pl.col("proc_federal_action_obligation").cast(pl.Float64, strict=False).alias("proc_obligation_f"),
        pl.col("proc_number_of_offers_received").cast(pl.Float64, strict=False).alias("proc_offers_f"),
        pl.col("proc_place_of_manufacture_code")
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.to_uppercase()
        .alias("proc_mfg_code_clean"),
    )
    .with_columns(
        (pl.col("proc_mfg_code_clean") == "D").alias("is_domestic_mfg"),
        (pl.col("proc_mfg_code_clean").is_not_null() & (pl.col("proc_mfg_code_clean") != "")).alias("has_mfg_code"),
    )
)

proc_event_features = (
    proc_event_rows
    .group_by(event_keys)
    .agg(
        pl.col("proc_event_day").min().alias("min_proc_event_day"),
        pl.col("proc_event_day").max().alias("max_proc_event_day"),
        pl.col("proc_action_date_parsed").min().alias("first_proc_action_date"),
        pl.col("proc_action_date_parsed").max().alias("last_proc_action_date"),
        pl.len().alias("n_proc_original_award_transactions"),
        pl.col("proc_obligation_f").sum().alias("total_proc_original_award_obligation"),
        pl.col("proc_offers_f").drop_nulls().len().alias("n_proc_original_award_offer_values"),
        pl.col("proc_offers_f").mean().alias("mean_proc_original_award_offers"),
        pl.col("has_mfg_code").sum().alias("n_proc_original_award_mfg_known"),
        pl.col("is_domestic_mfg").filter(pl.col("has_mfg_code")).sum().alias("n_proc_original_award_domestic_mfg"),
        pl.col("is_domestic_mfg").filter(pl.col("has_mfg_code")).mean().alias("domestic_mfg_share"),
    )
)

all_event_keys = (
    pl.concat(
        [
            bidlink_event_features.select(event_keys),
            proc_event_features.select(event_keys),
        ],
        how="vertical",
    )
    .unique()
)

treatment_event_year_panel = (
    all_event_keys
    .join(bidlink_event_features, on=event_keys, how="left")
    .join(proc_event_features, on=event_keys, how="left")
    .rename({"bl_nsn": "nsn"})
    .sort(["nsn", "event_year"])
)

print("\nTreatment NSN x event-year panel:")
print(f"  Rows: {len(treatment_event_year_panel)}")
print(f"  Event-year range: {treatment_event_year_panel['event_year'].min()} to {treatment_event_year_panel['event_year'].max()}")

# %% [markdown]
# ### 7c. Join Diagnostics
#
# How well did BidLink contracts match to the procurement parquet?

# %%
# Per-NSN match rates (based on unique BidLink rows, not exploded join rows)
match_by_nsn = (
    treatment_panel
    .group_by("bl_nsn")
    .agg(
        pl.len().alias("panel_rows"),
        pl.col("matched_to_parquet").sum().alias("n_matched"),
        pl.col("matched_to_parquet").mean().alias("match_rate"),
    )
    .with_columns(
        pl.when(pl.col("bl_nsn").str.starts_with("8150"))
        .then(pl.lit("container"))
        .otherwise(pl.lit("non_container"))
        .alias("nsn_group")
    )
    .sort("bl_nsn")
)

print("Parquet match rate by NSN:")
match_by_nsn

# %%
# Overall match rates by group
print("\nOverall match rates by group:")
(
    match_by_nsn
    .group_by("nsn_group")
    .agg(
        pl.col("panel_rows").sum().alias("total_rows"),
        pl.col("n_matched").sum().alias("total_matched"),
    )
    .with_columns(
        (pl.col("total_matched") / pl.col("total_rows")).alias("match_rate")
    )
)

# %%
# Unmatched contracts: what are they?
unmatched = (
    treatment_panel
    .filter(~pl.col("matched_to_parquet"))
    .select("bl_nsn", "bl_contract", "bl_date_parsed", "bl_fiscal_year")
    .unique()
    .sort("bl_nsn", "bl_date_parsed")
)

print(f"\nUnmatched BidLink rows: {len(unmatched)}")
print("Unmatched by fiscal year:")
unmatched.group_by("bl_fiscal_year").len().sort("bl_fiscal_year")

# %% [markdown]
# ### 7d. Final Panel Summary

# %%
print(f"\n{'='*60}")
print(f"TREATMENT PANEL SUMMARY")
print(f"{'='*60}")
print(f"Treatment NSNs: {len(treatment_nsns)}")
print(f"  Non-container: {n_non_container}")
print(f"  Container: {n_container}")
print(f"Panel rows (FY2017+): {len(treatment_panel)}")
print(f"Unique BidLink transactions: {len(treatment_bidlink)}")
print(f"Parquet match rate: {treatment_panel['matched_to_parquet'].mean():.1%}")
print(f"FY range: {treatment_panel['bl_fiscal_year'].min()}-{treatment_panel['bl_fiscal_year'].max()}")
print(f"{'='*60}")

# %%
# Only the first-waiver-date lookup below feeds the rest of the pipeline.
(RESULTS / "treatment").mkdir(parents=True, exist_ok=True)

# Export treatment NSN -> first waiver date lookup for downstream control selection
treatment_dates_path = TREATMENT_DATES
(
    treatment_nsns_df
    .select(
        pl.col("NSN").alias("nsn"),
        pl.col("waiver_date").alias("first_waiver_date"),
    )
    .write_csv(treatment_dates_path)
)
print(f"Saved: {treatment_dates_path.relative_to(ROOT)}")
print(f"  {len(treatment_nsns_df)} rows, 2 columns")

# %% [markdown]
# ### Treatment NSN List (for downstream use)
#
# Copy this list into downstream scripts that need the canonical treatment NSNs.

# %%
print("TREATMENT_NSNS = [")
for nsn in sorted(treatment_nsns):
    print(f"    '{nsn}',")
print("]")
