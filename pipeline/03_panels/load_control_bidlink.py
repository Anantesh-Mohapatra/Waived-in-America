"""Load and standardize control donor BidLink data.

Reads the raw control_contracts CSV files and maps each row to its canonical
donor NSN by NIIN (resolved against the DLA contract history via control_nsns),
then standardizes to the shared bl_* schema.

Inputs:
  - raw_data/bidlink/control_contracts/*.csv

Each control donor has a unique NIIN (no overlap with treatment NSNs). Some
files carry a NIIN under more than one FSC prefix (reclassification); keying on
the NIIN resolves every row to the one canonical NSN.
"""
from __future__ import annotations

import glob as _glob
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from bidlink_core import standardize_bidlink, filter_fy2017_plus
from control_nsns import control_donor_map
from paths import RAW_BIDLINK


def load_control_bidlink() -> pl.DataFrame:
    """Return standardized control BidLink rows (FY2017+).

    Columns added beyond core bl_* fields:
      - role = "control"
      - nsn_group = "container" | "non_container"
      - first_waiver_date = null (controls have no waiver)
      - is_pre_first_waiver = null
      - event_day = null
      - event_year = null
    """
    # --- NIIN -> canonical donor NSN map (control_contracts folder + DLA) ---
    niin_to_canonical = control_donor_map()

    # --- Load all control BidLink files ---
    bidlink_dir = RAW_BIDLINK / "control_contracts"
    bidlink_files = sorted(_glob.glob(str(bidlink_dir / "*.csv")))
    assert len(bidlink_files) == 63, f"Expected 63 files, found {len(bidlink_files)}"

    dfs = []
    for f in bidlink_files:
        df = pl.read_csv(f, infer_schema_length=1000, infer_schema=False)
        dfs.append(df)

    raw = pl.concat(dfs, how="diagonal")

    # --- Standardize ---
    raw = standardize_bidlink(raw)
    raw = filter_fy2017_plus(raw)

    # --- Map to canonical donor NSN via NIIN ---
    # Extract NIIN: last 9 digits after stripping dashes
    raw = raw.with_columns(
        pl.col("bl_nsn").str.replace_all("-", "").str.slice(-9).alias("_niin"),
    )
    raw = raw.with_columns(
        pl.col("_niin").replace_strict(niin_to_canonical, default=None).alias("_canonical_nsn"),
    )

    # Check for mapping failures
    n_unmapped = raw.filter(pl.col("_canonical_nsn").is_null()).height
    if n_unmapped > 0:
        unmapped_niins = raw.filter(pl.col("_canonical_nsn").is_null())["_niin"].unique().to_list()
        print(f"WARNING: {n_unmapped} rows with unmapped NIINs: {unmapped_niins}")

    # Replace bl_nsn with the canonical 13-digit undashed NSN
    raw = raw.with_columns(
        pl.col("_canonical_nsn").alias("bl_nsn"),
    ).drop(["_niin", "_canonical_nsn"])

    # Drop any rows that failed to map (should be 0)
    raw = raw.filter(pl.col("bl_nsn").is_not_null())

    # Drop exact-duplicate line items present in the raw control export. The
    # scrape emitted 3 fully identical rows for one donor (8145-01-523-4040,
    # contract SPE8ED14D0002, order SPE8ED18F0261P0, 2018-01-02), which would
    # otherwise inflate that donor's pre-waiver transaction count in synth.
    # Treatment BidLink has no such duplicates (verified).
    n_before = raw.height
    raw = raw.unique(maintain_order=True)
    if raw.height != n_before:
        print(f"  dropped {n_before - raw.height} exact-duplicate raw control line item(s)")

    # --- Add role, group, and null treatment-timing columns ---
    raw = raw.with_columns(
        pl.lit("control").alias("role"),
        pl.when(pl.col("bl_nsn").str.slice(0, 4).is_in(["8145", "8150"]))
        .then(pl.lit("container"))
        .otherwise(pl.lit("non_container"))
        .alias("nsn_group"),
        pl.lit(None).cast(pl.Date).alias("first_waiver_date"),
        pl.lit(None).cast(pl.Boolean).alias("is_pre_first_waiver"),
        pl.lit(None).cast(pl.Int64).alias("event_day"),
        pl.lit(None).cast(pl.Int64).alias("event_year"),
    )

    return raw
