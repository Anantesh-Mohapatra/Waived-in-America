"""Shared BidLink loading, FPDS enrichment, and event-year functions.

All functions are pure (stateless). They accept and return Polars DataFrames.
File I/O is handled by callers, not here, except for the FPDS parquet scan
(which must stream from disk due to size).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FY2017_CUTOFF = date(2016, 10, 1)  # FY2017 starts Oct 1, 2016

# FPDS columns to select for enrichment (original-award rows only).
# award_id_piid is used for the join but not prefixed (dropped after join).
FPDS_ENRICHMENT_COLS = [
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
    "type_of_set_aside_code",
]


# ---------------------------------------------------------------------------
# BidLink standardization
# ---------------------------------------------------------------------------

def standardize_bidlink(df: pl.DataFrame) -> pl.DataFrame:
    """Clean raw BidLink CSV columns into a standardized bl_* schema.

    Input columns (raw):  Date, Contract, Order, Line, NSN, Price, Qty, Cage, Company
    Output columns:       bl_date, bl_contract, bl_order, bl_line, bl_nsn,
                          bl_price, bl_qty, bl_cage, bl_company,
                          bl_date_parsed, bl_fiscal_year

    All string columns are stripped of whitespace and surrounding quotes.
    Price is cast to Float64, Qty to Int64 (both with strict=False).
    bl_date_parsed is a Date column; bl_fiscal_year is the federal FY (Oct = next year).
    """
    # Strip whitespace and quotes from column names first
    df = df.rename({c: c.strip().strip('"') for c in df.columns})

    # Rename to bl_* prefix
    rename_map = {
        "Date": "bl_date",
        "Contract": "bl_contract",
        "Order": "bl_order",
        "Line": "bl_line",
        "NSN": "bl_nsn",
        "Price": "bl_price",
        "Qty": "bl_qty",
        "Cage": "bl_cage",
        "Company": "bl_company",
    }
    df = df.rename(rename_map)

    # Clean string values (strip whitespace + quotes)
    str_cols = ["bl_date", "bl_contract", "bl_order", "bl_line",
                "bl_nsn", "bl_cage", "bl_company"]
    df = df.with_columns(
        [pl.col(c).cast(pl.Utf8).str.strip_chars().str.strip_chars('"') for c in str_cols]
    )

    # Parse price and quantity
    df = df.with_columns(
        pl.col("bl_price")
        .cast(pl.Utf8).str.strip_chars().str.strip_chars('"')
        .str.replace_all(",", "")
        .cast(pl.Float64, strict=False)
        .alias("bl_price"),
        pl.col("bl_qty")
        .cast(pl.Utf8).str.strip_chars().str.strip_chars('"')
        .str.replace_all(",", "")
        .cast(pl.Int64, strict=False)
        .alias("bl_qty"),
    )

    # Parse date and compute fiscal year
    df = df.with_columns(
        pl.col("bl_date").str.to_date("%Y/%m/%d", strict=False).alias("bl_date_parsed"),
    )
    df = df.with_columns(
        pl.when(pl.col("bl_date_parsed").dt.month() >= 10)
        .then(pl.col("bl_date_parsed").dt.year() + 1)
        .otherwise(pl.col("bl_date_parsed").dt.year())
        .alias("bl_fiscal_year"),
    )

    return df


def filter_fy2017_plus(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only rows with bl_date_parsed >= Oct 1, 2016 (FY2017+)."""
    return df.filter(pl.col("bl_date_parsed") >= FY2017_CUTOFF)


# ---------------------------------------------------------------------------
# FPDS join
# ---------------------------------------------------------------------------

def _order_is_piid_expr() -> pl.Expr:
    """Polars expression: True if bl_order looks like a real PIID."""
    return (
        pl.col("bl_order").is_not_null()
        & (pl.col("bl_order") != "")
        & pl.col("bl_order").str.contains(r"[A-Za-z]")
        & pl.col("bl_order").str.contains(r"[0-9]")
        & (pl.col("bl_order").str.len_chars() >= 10)
    )


def _order_is_short_suffix_expr() -> pl.Expr:
    """Polars expression: True if bl_order is a short suffix (concat with Contract)."""
    return (
        pl.col("bl_order").is_not_null()
        & (pl.col("bl_order") != "")
        & (pl.col("bl_order").str.len_chars() <= 8)
        & (pl.col("bl_order").str.len_chars() > 0)
    )


def collect_candidate_piids(df: pl.DataFrame) -> list[str]:
    """Extract all candidate FPDS PIIDs from a BidLink DataFrame.

    Four sources:
    1. bl_contract (always, unless "-")
    2. bl_order when it looks like a full PIID (letters+digits, len>=10)
    3. bl_order as-is when it's a short suffix (for Method D parent-constrained join)
    4. bl_contract + bl_order when Order is a short suffix (Method B concat, kept for compat)
    """
    contracts = (
        df.filter(pl.col("bl_contract") != "-")
        ["bl_contract"].unique().to_list()
    )
    full_orders = (
        df.filter(_order_is_piid_expr())
        ["bl_order"].unique().to_list()
    )
    # Short Order codes as-is (for Method D: parent-constrained join)
    short_orders = (
        df.filter(_order_is_short_suffix_expr())
        ["bl_order"].unique().to_list()
    )
    # Also keep concat form (Method B, for backward compat)
    short_concats = (
        df.filter(_order_is_short_suffix_expr())
        .select((pl.col("bl_contract") + pl.col("bl_order")).alias("concat"))
        ["concat"].unique().to_list()
    )
    return list(set(contracts + full_orders + short_orders + short_concats))


def compute_join_piid(df: pl.DataFrame, fpds_piid_set: set[str]) -> pl.DataFrame:
    """Add join_piid and used_order_fallback columns.

    Logic (extended with short-suffix concat):
    1. Default: join_piid = bl_contract
    2. If bl_contract not in FPDS and bl_order is a full PIID: join_piid = bl_order
    3. If still not matched and bl_order is short: join_piid = bl_contract + bl_order
    """
    contract_in_fpds = pl.col("bl_contract").is_in(fpds_piid_set)
    order_is_piid = _order_is_piid_expr()
    order_is_short = _order_is_short_suffix_expr()

    df = df.with_columns(
        pl.when(contract_in_fpds)
        .then(pl.col("bl_contract"))
        .when(order_is_piid)
        .then(pl.col("bl_order"))
        .when(order_is_short)
        .then(pl.col("bl_contract") + pl.col("bl_order"))
        .otherwise(pl.col("bl_contract"))
        .alias("join_piid"),
    )
    df = df.with_columns(
        (pl.col("join_piid") != pl.col("bl_contract")).alias("used_order_fallback"),
    )
    return df


def load_fpds_for_piids(
    parquet_path: Path,
    piid_list: list[str],
) -> pl.DataFrame:
    """Scan FPDS parquet, filter to PIIDs and mod 0, return proc_* prefixed columns.

    Only original-award rows (modification_number == "0" or null) are returned.
    This ensures a 1:1 join with BidLink line items (no row multiplication from mods).

    Also fetches parent_award_id_piid (needed for Method D parent-constrained join).
    """
    cols = FPDS_ENRICHMENT_COLS + ["parent_award_id_piid"]
    df = (
        pl.scan_parquet(parquet_path)
        .select(cols)
        .filter(pl.col("award_id_piid").is_in(piid_list))
        .filter(
            pl.col("modification_number").is_null()
            | (pl.col("modification_number").cast(pl.Utf8) == "0")
        )
        .collect(engine="streaming")
    )
    # Prefix all columns except join keys
    keep_raw = {"award_id_piid", "parent_award_id_piid"}
    rename = {c: f"proc_{c}" for c in df.columns if c not in keep_raw}
    df = df.rename(rename)
    # Drop proc_modification_number (always "0" or null after filter)
    if "proc_modification_number" in df.columns:
        df = df.drop("proc_modification_number")
    return df


def enrich_with_fpds(
    bidlink_df: pl.DataFrame,
    fpds_df: pl.DataFrame,
) -> pl.DataFrame:
    """Left join BidLink to FPDS on join_piid = award_id_piid.

    Adds matched_to_parquet boolean column. Drops parent_award_id_piid from
    the FPDS side (used only for Method D, not needed in output).
    """
    fpds_for_join = fpds_df.drop("parent_award_id_piid")
    joined = bidlink_df.join(
        fpds_for_join,
        left_on="join_piid",
        right_on="award_id_piid",
        how="left",
    )
    joined = joined.with_columns(
        pl.col("proc_contract_transaction_unique_key").is_not_null().alias("matched_to_parquet"),
    )
    return joined


def enrich_unmatched_via_parent(
    panel: pl.DataFrame,
    fpds_df: pl.DataFrame,
) -> pl.DataFrame:
    """Second-pass enrichment for BPA call orders with short Order codes.

    Method D: For unmatched rows where bl_order is a short suffix, try joining
    on award_id_piid = bl_order AND parent_award_id_piid = bl_contract.
    This recovers BPA call orders where FPDS stores the short call order code
    as the award_id_piid (e.g., 'X38V') with the parent BPA as parent_award_id_piid.
    """
    matched = panel.filter(pl.col("matched_to_parquet"))
    unmatched = panel.filter(~pl.col("matched_to_parquet"))

    if unmatched.height == 0:
        return panel

    # Identify candidates: unmatched rows with short Order that wasn't already
    # used as a full PIID (Method A)
    candidates = unmatched.filter(_order_is_short_suffix_expr())

    if candidates.height == 0:
        return panel

    # Non-candidate unmatched rows stay as-is
    non_candidates = unmatched.filter(~_order_is_short_suffix_expr())

    # Prepare FPDS side: need both award_id_piid and parent_award_id_piid
    fpds_parent = fpds_df.select(
        "award_id_piid",
        "parent_award_id_piid",
        *[c for c in fpds_df.columns if c.startswith("proc_")],
    )

    # Drop existing proc_* null columns from candidates before re-joining
    proc_cols = [c for c in candidates.columns if c.startswith("proc_")]
    candidates = candidates.drop(proc_cols + ["matched_to_parquet"])

    # Join on (bl_order = award_id_piid, bl_contract = parent_award_id_piid)
    rejoined = candidates.join(
        fpds_parent,
        left_on=["bl_order", "bl_contract"],
        right_on=["award_id_piid", "parent_award_id_piid"],
        how="left",
    )
    rejoined = rejoined.with_columns(
        pl.col("proc_contract_transaction_unique_key").is_not_null().alias("matched_to_parquet"),
    )

    # Update join metadata for rows that matched via Method D
    rejoined = rejoined.with_columns(
        pl.when(pl.col("matched_to_parquet"))
        .then(pl.col("bl_order"))
        .otherwise(pl.col("join_piid"))
        .alias("join_piid"),
        pl.when(pl.col("matched_to_parquet"))
        .then(pl.lit(True))
        .otherwise(pl.col("used_order_fallback"))
        .alias("used_order_fallback"),
    )

    # Recombine: matched (pass 1) + rejoined (pass 2) + non-candidates (still unmatched)
    return pl.concat([matched, rejoined, non_candidates], how="diagonal")


# ---------------------------------------------------------------------------
# Event-year computation
# ---------------------------------------------------------------------------

def compute_event_year(event_day: int) -> int:
    """Scalar event-year from event-day offset."""
    if event_day < 0:
        return -(((-event_day - 1) // 365) + 1)
    return event_day // 365


def event_day_expr(date_col: str, waiver_col: str = "first_waiver_date") -> pl.Expr:
    """Polars expression: days from waiver date."""
    return (pl.col(date_col) - pl.col(waiver_col)).dt.total_days()


def event_year_expr(event_day_col: str = "event_day") -> pl.Expr:
    """Polars expression: piecewise integer event year from event_day column."""
    d = pl.col(event_day_col)
    neg = -(((-d - 1) // 365) + 1)
    pos = d // 365
    return pl.when(d < 0).then(neg).otherwise(pos).cast(pl.Int64)


def assert_event_year_vectors() -> None:
    """Verify event-year formula with test vectors."""
    vectors = [
        (-1, -1),
        (-365, -1),
        (-366, -2),
        (0, 0),
        (364, 0),
        (365, 1),
        (730, 2),
    ]
    for day, expected in vectors:
        got = compute_event_year(day)
        assert got == expected, f"event_year({day}) = {got}, expected {expected}"


# Run assertion on import to catch formula drift early
assert_event_year_vectors()
