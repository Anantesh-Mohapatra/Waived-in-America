"""
Clean and merge DLA FOIA contract history and procurement history files.

Source: raw_data/DLA_FOIA/ (pipe-delimited text files in zips, some nested)
        data/clean/procurement_data.parquet (USAspending, for enrichment join)
Output: data/clean/dla_contract_history.parquet
        data/clean/dla_procurement_history.parquet
        data/clean/dla_contract_enriched_base.parquet    (DLA + USAspending mod=0 fallback)
        data/clean/dla_contract_enriched_latest.parquet  (DLA + USAspending latest-mod fallback)

Contract history covers: FY2017-FY2026.
  Source files are CY2016-CY2026. Transactions before FY2017
  (i.e. before Oct 1, 2016) are filtered out after loading.
Procurement history covers: FY2018, FY2026 only (unclear why only these two years exist)

Quirks handled:
  - Three different zip naming conventions across years (DLA changed formats)
  - 2019-2022: nested zips (monthly zips inside yearly zips)
  - 2020/Jun: bare .txt inside zip (not nested further)
  - Feb 2020 and the 2019Jan01-2020Jan17 overlap file are missing PART_NUMBER column
  - 2020 contains a '2019Jan01-2020Jan17' file that overlaps with 2019 data
"""

import gc
import io
import sys
import zipfile
from datetime import date
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from paths import CLEAN, PROCUREMENT_PARQUET, RAW_DLA_FOIA

FY2017_START = date(2016, 10, 1)

RAW_DIR = RAW_DLA_FOIA
OUT_DIR = CLEAN
USA_PARQUET = PROCUREMENT_PARQUET

# ── Contract history schema ──────────────────────────────────────────────
# Standard 15-column schema (most files)
CONTRACT_COLS_FULL = [
    "NIIN", "SECURITY_CLASSIFICATION", "FSC", "UNIT", "CAGE",
    "CONTRACT_NUMBER", "ORDER_QTY", "AWARD_DATE", "NETPRICE",
    "PO_NUM", "PO_ITMNO", "ITEM_NAME", "PART_NUMBER", "STD_U_PRICE", "NSN",
]
# 14-column schema (Feb 2020, 2019Jan-2020Jan overlap) -- missing PART_NUMBER
CONTRACT_COLS_NO_PART = [
    "NIIN", "SECURITY_CLASSIFICATION", "FSC", "UNIT", "CAGE",
    "CONTRACT_NUMBER", "ORDER_QTY", "AWARD_DATE", "NETPRICE",
    "PO_NUM", "PO_ITMNO", "ITEM_NAME", "STD_U_PRICE", "NSN",
]

# ── Procurement history schema ───────────────────────────────────────────
PROC_COLS = [
    "NIIN", "SECURITY_CLASSIFICATION", "FSC", "CLIN", "UNIT", "CAGE",
    "CONTRACT_NUMBER", "REFERENCED_PIID", "ORDER_QTY", "AWARD_DATE",
    "NETPRICE", "STD_U_PRICE", "PO_NUM", "PO_ITMNO", "PIINSPIINMod",
    "SOLIC_AMENDMENT_NUMBER",
]

# Numeric columns to cast
CONTRACT_NUMERIC = {"ORDER_QTY": pl.Float64, "NETPRICE": pl.Float64, "STD_U_PRICE": pl.Float64}
PROC_NUMERIC = {"ORDER_QTY": pl.Float64, "NETPRICE": pl.Float64, "STD_U_PRICE": pl.Float64}


def read_pipe_text(raw_bytes: bytes, label: str) -> pl.DataFrame:
    """Read pipe-delimited text bytes into a Polars DataFrame (all string cols)."""
    text = raw_bytes.decode("latin-1")
    return pl.read_csv(
        io.StringIO(text),
        separator="|",
        has_header=True,
        infer_schema=False,  # read everything as string first
        ignore_errors=True,
    )


def extract_txt_from_zip(zf: zipfile.ZipFile, name: str) -> bytes:
    """Read a .txt file from an open ZipFile."""
    return zf.read(name)


def extract_txt_from_nested_zip(zf: zipfile.ZipFile, inner_zip_name: str) -> bytes:
    """Read the first .txt from a zip nested inside another zip."""
    inner_bytes = zf.read(inner_zip_name)
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as z2:
        txt_files = [n for n in z2.namelist() if n.endswith(".txt")]
        assert txt_files, f"No .txt found inside {inner_zip_name}"
        return z2.read(txt_files[0])


# ── Contract history loading ─────────────────────────────────────────────

def load_contract_history() -> pl.DataFrame:
    # Incremental concat to avoid holding all source frames simultaneously.
    # Each file is parsed, appended to the running result, then freed.
    merged: pl.DataFrame | None = None
    file_counts: list[tuple[str, int]] = []

    def append_frame(raw: bytes, label: str):
        nonlocal merged
        df = read_pipe_text(raw, label)
        # Add PART_NUMBER column if missing (null-filled)
        if "PART_NUMBER" not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.String).alias("PART_NUMBER"))
        # Reorder to standard schema
        df = df.select(CONTRACT_COLS_FULL)
        file_counts.append((label, len(df)))
        if merged is None:
            merged = df
        else:
            merged = pl.concat([merged, df], how="vertical_relaxed")
        del df

    # 2016, 2017, 2018: single txt in zip
    for year in [2016, 2017, 2018]:
        zname = f"contract_history_{year}.zip"
        print(f"  Reading {zname}")
        with zipfile.ZipFile(RAW_DIR / zname) as zf:
            txt = [n for n in zf.namelist() if n.endswith(".txt")][0]
            append_frame(zf.read(txt), f"contract_{year}")

    # 2019: zip > subfolder > nested zip > txt
    print("  Reading contract_history_2019.zip")
    with zipfile.ZipFile(RAW_DIR / "contract_history_2019.zip") as zf:
        inner_zips = [n for n in zf.namelist() if n.endswith(".zip")]
        for iz in inner_zips:
            raw = extract_txt_from_nested_zip(zf, iz)
            append_frame(raw, f"contract_2019/{iz}")

    # 2020-2022: zip > subfolder > monthly zips/txts
    for year in [2020, 2021, 2022]:
        zname = f"contract_history_{year}.zip"
        print(f"  Reading {zname}")
        with zipfile.ZipFile(RAW_DIR / zname) as zf:
            for name in sorted(zf.namelist()):
                if name.endswith("/"):
                    continue
                if name.endswith(".txt"):
                    # Bare txt (e.g. Jun 2020)
                    raw = zf.read(name)
                    append_frame(raw, f"contract_{year}/{name}")
                elif name.endswith(".zip"):
                    raw = extract_txt_from_nested_zip(zf, name)
                    append_frame(raw, f"contract_{year}/{name}")

    # 2023, 2024, 2025: single txt in zip (different naming conventions per year)
    for zname, label in [("2023ContHist.zip", "contract_2023"),
                         ("2024ContHist.zip", "contract_2024"),
                         ("contracthist2025.zip", "contract_2025")]:
        print(f"  Reading {zname}")
        with zipfile.ZipFile(RAW_DIR / zname) as zf:
            txt = [n for n in zf.namelist() if n.endswith(".txt")][0]
            append_frame(zf.read(txt), label)

    # 2026
    zname = "2026-03-17-0905_contracthist.zip"
    print(f"  Reading {zname}")
    with zipfile.ZipFile(RAW_DIR / zname) as zf:
        txt = [n for n in zf.namelist() if n.endswith(".txt")][0]
        append_frame(zf.read(txt), "contract_2026")

    # Report per-file row counts
    print("\n  Contract history file counts:")
    for label, n in file_counts:
        print(f"    {label}: {n:,} rows")

    assert merged is not None, "No contract history files found"
    return merged


# ── Procurement history loading ──────────────────────────────────────────

def load_procurement_history() -> pl.DataFrame:
    frames = []

    # 2018
    zname = "prochist2018.zip"
    print(f"  Reading {zname}")
    with zipfile.ZipFile(RAW_DIR / zname) as zf:
        txt = [n for n in zf.namelist() if n.endswith(".txt")][0]
        df = read_pipe_text(zf.read(txt), "proc_2018")
        frames.append(("proc_2018", df))

    # Procurement history only exists for 2018 and 2026 in the FOIA release.
    # No procurement history files for 2017, 2019-2025.

    # 2026
    zname = "2026-03-17-0905_prochist.zip"
    print(f"  Reading {zname}")
    with zipfile.ZipFile(RAW_DIR / zname) as zf:
        txt = [n for n in zf.namelist() if n.endswith(".txt")][0]
        df = read_pipe_text(zf.read(txt), "proc_2026")
        frames.append(("proc_2026", df))

    print("\n  Procurement history file counts:")
    for label, df in frames:
        print(f"    {label}: {len(df):,} rows")

    merged = pl.concat([df for _, df in frames], how="vertical_relaxed")
    return merged


# ── Type casting & cleanup ───────────────────────────────────────────────

def cast_contract_types(df: pl.DataFrame) -> pl.DataFrame:
    """Cast numeric columns, parse AWARD_DATE, split CONTRACT_NUMBER into BASE_PIID + MOD_NUMBER, and rebuild NSN from FSC||NIIN."""
    return df.with_columns(
        pl.col("ORDER_QTY").cast(pl.Float64, strict=False),
        pl.col("NETPRICE").cast(pl.Float64, strict=False),
        pl.col("STD_U_PRICE").cast(pl.Float64, strict=False),
        pl.col("AWARD_DATE").str.to_date("%Y%m%d", strict=False),
        # NSN rebuild.
        # The DLA FOIA contract-history source files ship both FSC and NSN as
        # separate columns, and they disagree for ~2.9% of rows (~353K rows
        # across ~10.9K NIINs). Example: NIIN 015737424 has FSC=8150 but
        # source NSN=8145015737424 across all 11 contract rows, 2017-2022.
        # BidLink and nsnlookup (external sources of truth) confirm the FSC
        # column is the trustworthy one and the source NSN column holds stale
        # prefixes from earlier DLA classifications. Without this rebuild,
        # NSN-keyed joins against the enriched parquets silently drop rows
        # for ~10.9K items that have been FSC-reclassified over their
        # lifetime. cast_proc_types (below) already does NSN = FSC || NIIN
        # for the procurement-history branch (line ~220); this line brings
        # contract-history into parity so NSN = FSC || NIIN is a uniform
        # invariant across all DLA parquets.
        (pl.col("FSC") + pl.col("NIIN")).alias("NSN"),
        # BASE_PIID: CONTRACT_NUMBER with trailing modification suffix stripped
        pl.col("CONTRACT_NUMBER").str.replace(r"P\d{5}$", "").alias("BASE_PIID"),
        # MOD_NUMBER: trailing P##### suffix if present, else null
        pl.col("CONTRACT_NUMBER").str.extract(r"(P\d{5})$").alias("MOD_NUMBER"),
    )


def cast_proc_types(df: pl.DataFrame) -> pl.DataFrame:
    """Cast numeric columns, parse AWARD_DATE, derive NSN, and split CONTRACT_NUMBER."""
    return df.with_columns(
        pl.col("ORDER_QTY").cast(pl.Float64, strict=False),
        pl.col("NETPRICE").cast(pl.Float64, strict=False),
        pl.col("STD_U_PRICE").cast(pl.Float64, strict=False),
        pl.col("AWARD_DATE").str.to_date("%Y%m%d", strict=False),
        (pl.col("FSC") + pl.col("NIIN")).alias("NSN"),
        # BASE_PIID: CONTRACT_NUMBER with trailing modification suffix stripped
        pl.col("CONTRACT_NUMBER").str.replace(r"P\d{5}$", "").alias("BASE_PIID"),
        # MOD_NUMBER: trailing P##### suffix if present, else null
        pl.col("CONTRACT_NUMBER").str.extract(r"(P\d{5})$").alias("MOD_NUMBER"),
    )


# ── USAspending enrichment ───────────────────────────────────────────────

# Columns to pull from USAspending into the enriched DLA parquet.
USA_COLS = [
    "federal_action_obligation",
    "total_dollars_obligated",
    "current_total_value_of_award",
    "base_and_all_options_value",
    "number_of_offers_received",
    "country_of_product_or_service_origin_code",
    "country_of_product_or_service_origin",
    "place_of_manufacture_code",
    "place_of_manufacture",
    "extent_competed_code",
    "extent_competed",
    "type_of_contract_pricing_code",
    "type_of_contract_pricing",
    "type_of_set_aside_code",
    "type_of_set_aside",
    "domestic_or_foreign_entity_code",
    "domestic_or_foreign_entity",
    "product_or_service_code",
    "naics_code",
    "recipient_name",
    "recipient_country_code",
    "recipient_country_name",
]

USA_NUMERIC_COLS = [
    "federal_action_obligation",
    "total_dollars_obligated",
    "current_total_value_of_award",
    "base_and_all_options_value",
    "number_of_offers_received",
]


def _usa_scan() -> pl.LazyFrame:
    """Lazy scan of USAspending with DLA_PIID join keys added.

    DLA CONTRACT_NUMBERs map to USAspending in three ways:
      - 13-char standalone award (no parent) -> award_id_piid directly
      - 17-char delivery order (e.g. SPE7MX15D00480619) ->
        parent_award_id_piid (13) + award_id_piid (4)
      - 13-char delivery order -> award_id_piid directly
        (DLA FOIA stores only the order PIID, not the parent prefix)

    Rows with a parent get TWO join keys (parent+award AND award alone)
    so both 17-char and 13-char DLA formats can match. These are exploded
    into separate rows so the lookup is keyed uniformly by DLA_PIID.
    """
    data_cols = ["modification_number"] + USA_COLS
    return (
        pl.scan_parquet(USA_PARQUET)
        .with_columns(
            pl.when(pl.col("parent_award_id_piid").is_not_null())
            .then(pl.concat_list(
                pl.col("parent_award_id_piid") + pl.col("award_id_piid"),
                pl.col("award_id_piid"),
            ))
            .otherwise(pl.concat_list(pl.col("award_id_piid")))
            .alias("DLA_PIID")
        )
        .select(["DLA_PIID"] + data_cols)
        .explode("DLA_PIID")
    )


def _build_usa_mod_lookup(dla_piids: pl.Series) -> pl.DataFrame:
    """Build USAspending lookup keyed by (DLA_PIID, modification_number),
    filtered to only PIIDs that exist in the DLA contract history.

    Without filtering, USAspending has ~36M+ (piid, mod) pairs across all
    agencies. Pre-filtering to DLA-matchable PIIDs keeps memory manageable.

    Uses streaming collect to avoid materializing the full intermediate
    join result in memory at once.
    """
    piid_df = dla_piids.unique().to_frame("DLA_PIID")
    return (
        _usa_scan()
        .join(piid_df.lazy(), on="DLA_PIID", how="semi")
        .group_by("DLA_PIID", "modification_number").first()
        .collect(engine="streaming")
    )


def _build_latest_mod_map(usa_mod_lookup: pl.DataFrame) -> pl.DataFrame:
    """Small PIID -> latest modification_number mapping (two string cols)."""
    return (
        usa_mod_lookup.lazy()
        .select("DLA_PIID", "modification_number")
        .sort("modification_number", descending=True)
        .group_by("DLA_PIID").first()
        .rename({"modification_number": "_LATEST_MOD"})
        .collect()
    )


def _write_enriched_pass(
    contract_path: Path,
    out_path: Path,
    usa_mod_lookup: pl.DataFrame,
    mode: str,
    latest_mod_map: pl.DataFrame | None,
    chunk_size: int,
) -> tuple[int, int]:
    """Write one enriched parquet (base or latest) in chunks.

    Returns (matched_count, total_rows).
    Separated from the other mode so only the lookup tables needed for this
    mode are in memory at once.
    """
    total_rows = pl.scan_parquet(contract_path).select(pl.len()).collect().item()
    n_chunks = (total_rows + chunk_size - 1) // chunk_size
    matched = 0
    row_total = 0
    writer = None

    try:
        for i in range(n_chunks):
            offset = i * chunk_size
            chunk = pl.scan_parquet(contract_path).slice(offset, chunk_size).collect()
            row_total += len(chunk)

            chunk = chunk.with_columns(
                pl.col("BASE_PIID").str.slice(0, 17).alias("_JOIN_PIID"),
            )

            if mode == "base":
                keyed = chunk.with_columns(
                    pl.col("MOD_NUMBER").fill_null("0").alias("_JOIN_MOD"),
                )
            else:
                keyed = (
                    chunk.lazy()
                    .join(latest_mod_map.lazy(),
                          left_on="_JOIN_PIID", right_on="DLA_PIID", how="left")
                    .with_columns(
                        pl.col("MOD_NUMBER").fill_null(pl.col("_LATEST_MOD")).alias("_JOIN_MOD"),
                    )
                    .drop("_LATEST_MOD")
                    .collect()
                )
            del chunk

            enriched = (
                keyed.lazy()
                .join(usa_mod_lookup.lazy(),
                      left_on=["_JOIN_PIID", "_JOIN_MOD"],
                      right_on=["DLA_PIID", "modification_number"],
                      how="left")
                .drop("_JOIN_PIID", "_JOIN_MOD")
                .with_columns(
                    pl.col(c).cast(pl.Float64, strict=False) for c in USA_NUMERIC_COLS
                )
                .collect()
            )
            del keyed

            matched += enriched.select(
                pl.col("recipient_name").is_not_null().sum()
            ).item()

            arrow_table = enriched.to_arrow()
            del enriched
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), arrow_table.schema, compression="zstd")
            writer.write_table(arrow_table)
            del arrow_table
            gc.collect()
            print(f"    Chunk {i+1}/{n_chunks} done ({row_total:,} rows)")
    finally:
        if writer is not None:
            writer.close()

    return matched, row_total


def enrich_dla_with_usaspending(contract_path: Path):
    """Build two enriched DLA parquets joined with USAspending data.

    Both parquets match DLA modification rows (P#####) to their exact
    USAspending modification. They differ only in how unmodified DLA rows
    are matched:

      dla_contract_enriched_base.parquet:
        Unmodded DLA rows -> USAspending mod=0 (original award)

      dla_contract_enriched_latest.parquet:
        Unmodded DLA rows -> USAspending latest mod (final award state)

    All original DLA columns are preserved, including MOD_NUMBER.

    Memory-efficient: USAspending lookup is pre-filtered to DLA PIIDs only.
    Base and latest enrichments are processed in separate passes so the
    latest-mod map doesn't coexist with base-mode processing.
    """
    CHUNK_SIZE = 500_000

    # Collect unique DLA join keys (one string column, ~7M unique values)
    print("\n  Collecting DLA PIID join keys...")
    dla_piids = (
        pl.scan_parquet(contract_path)
        .select(pl.col("BASE_PIID").str.slice(0, 17))
        .unique()
        .collect()
        .to_series()
    )
    print(f"  Unique DLA PIIDs: {len(dla_piids):,}")

    print("  Building USAspending modification-level lookup (DLA-filtered)...")
    usa_mod_lookup = _build_usa_mod_lookup(dla_piids)
    print(f"  Lookup: {len(usa_mod_lookup):,} unique (piid, mod) pairs")
    del dla_piids
    gc.collect()

    base_path = OUT_DIR / "dla_contract_enriched_base.parquet"
    latest_path = OUT_DIR / "dla_contract_enriched_latest.parquet"

    # Pass 1: base enrichment (no latest_mod_map needed)
    print("\n  --- Pass 1: base enrichment (mod=0 fallback) ---")
    base_matched, total_rows = _write_enriched_pass(
        contract_path, base_path, usa_mod_lookup,
        mode="base", latest_mod_map=None, chunk_size=CHUNK_SIZE,
    )
    pct = base_matched / total_rows * 100
    print(f"  {base_path.name}: {base_matched:,} / {total_rows:,} matched "
          f"({pct:.1f}%), {base_path.stat().st_size / 1e6:.1f} MB")

    # Pass 2: latest enrichment (needs latest_mod_map)
    print("\n  --- Pass 2: latest enrichment (latest-mod fallback) ---")
    print("  Building latest-mod mapping...")
    latest_mod_map = _build_latest_mod_map(usa_mod_lookup)
    print(f"  Latest-mod map: {len(latest_mod_map):,} unique awards")

    latest_matched, _ = _write_enriched_pass(
        contract_path, latest_path, usa_mod_lookup,
        mode="latest", latest_mod_map=latest_mod_map, chunk_size=CHUNK_SIZE,
    )
    pct = latest_matched / total_rows * 100
    print(f"  {latest_path.name}: {latest_matched:,} / {total_rows:,} matched "
          f"({pct:.1f}%), {latest_path.stat().st_size / 1e6:.1f} MB")

    del usa_mod_lookup, latest_mod_map
    gc.collect()

    print("\n  (Unmatched rows expected: USAspending parquet excludes")
    print("   place_of_manufacture_code='C' and may not cover all DLA award types)")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading contract history...")
    contract = load_contract_history()
    print(f"\n  Total contract history rows (raw): {len(contract):,}")
    print(f"  Columns: {contract.columns}")

    # Deduplicate: the 2020 archive contains a 2019Jan01-2020Jan17 overlap file
    pre_dedup = len(contract)
    contract = contract.unique()
    post_dedup = len(contract)
    print(f"  After dedup: {post_dedup:,} rows (dropped {pre_dedup - post_dedup:,} exact duplicates)")

    contract = cast_contract_types(contract)

    # Filter to FY2017+ (Oct 1, 2016 onward). CY2016 source file contains
    # Jan-Sep 2016 transactions that fall in FY2016; drop those.
    # Null AWARD_DATE rows are kept (they shouldn't be silently discarded).
    pre_fy17 = contract.filter(
        pl.col("AWARD_DATE").is_not_null() & (pl.col("AWARD_DATE") < FY2017_START)
    )
    contract = contract.filter(
        pl.col("AWARD_DATE").is_null() | (pl.col("AWARD_DATE") >= FY2017_START)
    )
    n_null_dates = contract.select(pl.col("AWARD_DATE").is_null().sum()).item()
    print(f"  Dropped {len(pre_fy17):,} rows before FY2017 (before {FY2017_START})")
    print(f"  Rows with null AWARD_DATE (kept): {n_null_dates:,}")
    del pre_fy17

    date_range = contract.select(
        pl.col("AWARD_DATE").min().alias("min"),
        pl.col("AWARD_DATE").max().alias("max"),
    ).row(0)
    print(f"  Date range: {date_range[0]} to {date_range[1]}")
    print(f"  Final contract history rows: {len(contract):,}")

    out_path = OUT_DIR / "dla_contract_history.parquet"
    contract.write_parquet(out_path, compression="zstd")
    print(f"  Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    print("\n" + "=" * 60)
    print("Loading procurement history...")
    proc = load_procurement_history()
    print(f"\n  Total procurement history rows (raw): {len(proc):,}")
    print(f"  Columns: {proc.columns}")

    pre_dedup = len(proc)
    proc = proc.unique()
    post_dedup = len(proc)
    print(f"  After dedup: {post_dedup:,} rows (dropped {pre_dedup - post_dedup:,} exact duplicates)")

    proc = cast_proc_types(proc)
    date_range = proc.select(
        pl.col("AWARD_DATE").min().alias("min"),
        pl.col("AWARD_DATE").max().alias("max"),
    ).row(0)
    print(f"  Date range: {date_range[0]} to {date_range[1]}")

    out_path = OUT_DIR / "dla_procurement_history.parquet"
    proc.write_parquet(out_path, compression="zstd")
    print(f"  Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # Free all dataframes before enrichment -- it re-reads from parquet
    contract_path = OUT_DIR / "dla_contract_history.parquet"
    del contract, proc
    gc.collect()

    # ── Enriched DLA + USAspending parquets ──
    print("\n" + "=" * 60)
    print("Enriching DLA contract history with USAspending data...")
    enrich_dla_with_usaspending(contract_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
