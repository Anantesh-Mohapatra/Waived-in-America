"""Load and standardize treatment BidLink data.

Reads the consolidated treatment BidLink CSV and the treatment NSN list,
filters to FY2017+, attaches waiver dates and event timing.

Inputs:
  - data/clean/bidlink_nsn.csv (consolidated treatment BidLink, all NSNs)
  - results/treatment/treatment_nsn_first_waiver_dates.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from bidlink_core import standardize_bidlink, filter_fy2017_plus, event_day_expr, event_year_expr
from paths import BIDLINK_NSN_CSV, TREATMENT_DATES


def load_treatment_bidlink() -> pl.DataFrame:
    """Return standardized treatment BidLink rows (FY2017+) with waiver timing.

    Columns added beyond core bl_* fields:
      - role = "treatment"
      - nsn_group = "container" | "non_container"
      - first_waiver_date, is_pre_first_waiver, event_day, event_year
    """
    # --- Load treatment NSN list ---
    waiver_dates = pl.read_csv(
        TREATMENT_DATES,
        infer_schema_length=0,
    ).with_columns(
        pl.col("first_waiver_date").str.to_date("%Y-%m-%d"),
        # Convert dashed NSN to 13-digit undashed for consistency
        pl.col("nsn").str.replace_all("-", "").alias("nsn_undashed"),
    )
    treatment_nsns_dashed = waiver_dates["nsn"].to_list()

    # --- Load BidLink ---
    bidlink = pl.read_csv(
        BIDLINK_NSN_CSV,
        infer_schema=False,
    )

    # Filter to treatment NSNs before standardization (uses raw NSN column)
    bidlink = bidlink.rename({c: c.strip() for c in bidlink.columns})
    bidlink = bidlink.filter(pl.col("NSN").str.strip_chars().is_in(treatment_nsns_dashed))

    # Standardize
    bidlink = standardize_bidlink(bidlink)
    bidlink = filter_fy2017_plus(bidlink)

    # Convert bl_nsn to 13-digit undashed
    bidlink = bidlink.with_columns(
        pl.col("bl_nsn").str.replace_all("-", "").alias("bl_nsn"),
    )

    # --- Attach waiver dates ---
    bidlink = bidlink.join(
        waiver_dates.select("nsn_undashed", "first_waiver_date"),
        left_on="bl_nsn",
        right_on="nsn_undashed",
        how="left",
    )

    # Compute treatment timing
    bidlink = bidlink.with_columns(
        (pl.col("bl_date_parsed") < pl.col("first_waiver_date")).alias("is_pre_first_waiver"),
        event_day_expr("bl_date_parsed").alias("event_day"),
    )
    bidlink = bidlink.with_columns(
        event_year_expr("event_day").alias("event_year"),
    )

    # --- Add role and group ---
    bidlink = bidlink.with_columns(
        pl.lit("treatment").alias("role"),
        pl.when(pl.col("bl_nsn").str.slice(0, 4) == "8150")
        .then(pl.lit("container"))
        .otherwise(pl.lit("non_container"))
        .alias("nsn_group"),
    )

    return bidlink
