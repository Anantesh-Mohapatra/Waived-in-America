"""Shared transaction-loading, standardization, and aggregation logic for the
NSN x FY panel builds.

Used by:
  - pipeline/03_panels/build_event_panel.py     (single-anchor, per-NSN waiver date)
  - pipeline/03_panels/build_anchored_panel.py  (per-treated anchor)

The split lets both pipelines reuse identical DLA dedup, BidLink overlay, and
(nsn, fy, event_year) cell-split aggregation. Differences are encoded in the
`anchor_df` passed to standardize_dla / standardize_bidlink.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

# ============================================================
# Constants
# ============================================================

# Day-precise event-year sentinel for rows whose anchor waiver_date is null
# after the left join (i.e., NSN absent from anchor_df). In the event-study
# panel this is every untreated NSN; in the matched-controls anchored panel,
# the anchor_df covers every NSN in scope so this should not fire.
EVENT_YEAR_SENTINEL = 1000

ACTION_DATE_CUTOFF_DEFAULT = date(2026, 1, 31)
MIN_PROXY_PRICE_OBS_DEFAULT = 2

DLA_COLS = [
    "NSN", "FSC", "BASE_PIID", "MOD_NUMBER", "PO_ITMNO", "AWARD_DATE",
    "ORDER_QTY", "NETPRICE",
    "federal_action_obligation", "number_of_offers_received",
    "place_of_manufacture_code", "extent_competed_code", "CAGE",
]

DLA_DEDUP_KEY = ["NSN", "BASE_PIID", "mod_n", "line_n", "AWARD_DATE"]
BL_DEDUP_KEY = ["nsn", "piid", "mod", "line", "action_date"]

STANDARDIZED_COLS = [
    "nsn", "fsc", "piid", "mod", "line", "action_date", "fy", "event_year",
    "qty", "unit_price", "obligation", "offers", "mfg_code", "extent_competed", "cage",
]


# ============================================================
# Utilities
# ============================================================

def normalize_nsn(s: pl.Expr) -> pl.Expr:
    return s.str.replace_all("-", "", literal=True).str.strip_chars()


def fed_fy(d: pl.Expr) -> pl.Expr:
    """Federal FY: month >= 10 -> year + 1, else year."""
    return pl.when(d.dt.month() >= 10).then(d.dt.year() + 1).otherwise(d.dt.year())


def event_year_expr(action_date_col: str, waiver_date_col: str = "first_waiver_date") -> pl.Expr:
    """Day-precise event_year for rows with a waiver_date; sentinel for rows without.

    floor((action_date - waiver_date) / 365.25). Rows where waiver_date_col is null
    (e.g., untreated NSNs after left-join) fall through to EVENT_YEAR_SENTINEL.
    """
    return (
        pl.when(pl.col(waiver_date_col).is_not_null())
          .then(((pl.col(action_date_col) - pl.col(waiver_date_col)).dt.total_days() / 365.25)
                .floor().cast(pl.Int32))
          .otherwise(pl.lit(EVENT_YEAR_SENTINEL, dtype=pl.Int32))
          .alias("event_year")
    )


def assert_schema_match(left: pl.DataFrame, right: pl.DataFrame, left_name: str, right_name: str) -> None:
    """Fail loudly if two frames disagree on column set or dtype.
    Used before vertical concats to catch silent dtype upcasts that
    vertical_relaxed would otherwise paper over."""
    left_cols = set(left.columns)
    right_cols = set(right.columns)
    only_left = left_cols - right_cols
    only_right = right_cols - left_cols
    if only_left or only_right:
        raise ValueError(
            f"Schema mismatch ({left_name} vs {right_name}): "
            f"only in {left_name}={sorted(only_left)}; only in {right_name}={sorted(only_right)}"
        )
    diffs = []
    for col in sorted(left_cols):
        if left.schema[col] != right.schema[col]:
            diffs.append(f"  {col}: {left_name}={left.schema[col]} vs {right_name}={right.schema[col]}")
    if diffs:
        raise ValueError(f"Dtype mismatch ({left_name} vs {right_name}):\n" + "\n".join(diffs))


# ============================================================
# DLA load + dedup (anchor-agnostic)
# ============================================================

def load_dla(
    dla_parq: Path,
    action_date_cutoff: date = ACTION_DATE_CUTOFF_DEFAULT,
    log_filter=None,
    anomalies_out: Path | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Stages 2 + 3: load full DLA, drop null dates/NSNs, line-level safe dedup.

    Returns (dla_deduped, anomalies). `anomalies` is the set of dup groups where
    qty / price / obligation vary within the group; these are kept un-deduped.
    If anomalies_out is provided and len(anomalies) > 0, also writes a CSV.
    """
    print(f"Reading DLA parquet via pyarrow (cutoff: action_date <= {action_date_cutoff})...")
    dla_arrow = pq.read_table(dla_parq, columns=DLA_COLS,
        filters=[("AWARD_DATE", "<=", action_date_cutoff)])
    print(f"pyarrow read: {dla_arrow.num_rows:,} rows, {dla_arrow.nbytes/1e9:.2f} GB in arrow buffers")
    dla = pl.from_arrow(dla_arrow)
    del dla_arrow
    if log_filter:
        log_filter("dla_action_date_cutoff", "DLA full (~13.5M)", len(dla),
                   f"keep action_date <= {action_date_cutoff}")

    n = len(dla)
    dla = dla.filter(pl.col("AWARD_DATE").is_not_null() & pl.col("NSN").is_not_null())
    if log_filter:
        log_filter("dla_drop_null_date_or_nsn", n, len(dla), "drop rows with null AWARD_DATE or NSN")

    # Line-level dedup
    dla = dla.with_columns(
        pl.when((pl.col("MOD_NUMBER").is_null()) | (pl.col("MOD_NUMBER").str.strip_chars() == ""))
          .then(pl.lit("BASE"))
          .otherwise(pl.col("MOD_NUMBER").str.strip_chars())
          .alias("mod_n"),
        pl.col("PO_ITMNO").str.strip_chars().str.strip_chars_start("0").alias("line_n"),
    )

    print("Scanning for anomalous dup groups (qty/price/obligation vary within group)...")
    grp = dla.group_by(DLA_DEDUP_KEY).agg(
        pl.len().alias("_n"),
        pl.col("ORDER_QTY").n_unique().alias("_nq"),
        pl.col("NETPRICE").n_unique().alias("_np"),
        pl.col("federal_action_obligation").n_unique().alias("_no"),
    )
    n_unique_keys = len(grp)
    n_dup_groups = grp.filter(pl.col("_n") > 1).height
    anomalies = grp.filter((pl.col("_n") > 1) & ((pl.col("_nq") > 1) | (pl.col("_np") > 1) | (pl.col("_no") > 1)))
    print(f"  Total unique line keys: {n_unique_keys:,}")
    print(f"  Dup groups (n>1): {n_dup_groups:,}")
    print(f"  Anomalous (qty/price/obl vary): {len(anomalies):,}")

    if len(anomalies) > 0:
        if anomalies_out is not None:
            # Sorted on the dedup key: group_by emits rows in uncontracted
            # order, which reshuffled this committed log on every run. The
            # sort affects only the written CSV's row order.
            anomalies.sort(DLA_DEDUP_KEY).write_csv(anomalies_out)
            print(f"  Wrote {anomalies_out}")
        anom_keys = anomalies.select(DLA_DEDUP_KEY)
        n = len(dla)
        clean = dla.join(anom_keys, on=DLA_DEDUP_KEY, how="anti")
        keep_anom = dla.join(anom_keys, on=DLA_DEDUP_KEY, how="inner")
        deduped_clean = clean.unique(subset=DLA_DEDUP_KEY, keep="first")
        dla = pl.concat([deduped_clean, keep_anom])
        if log_filter:
            log_filter("dla_safe_dedup", n, len(dla),
                       f"safe-dedup on {DLA_DEDUP_KEY}; {len(anomalies):,} anomalous groups left un-deduped")
    else:
        n = len(dla)
        dla = dla.unique(subset=DLA_DEDUP_KEY, keep="first")
        if log_filter:
            log_filter("dla_safe_dedup", n, len(dla), f"dedup on {DLA_DEDUP_KEY}; no anomalies found")

    del grp
    return dla, anomalies


# ============================================================
# DLA standardize (per anchor)
# ============================================================

def standardize_dla(
    dla_tx: pl.DataFrame,
    anchor_df: pl.DataFrame,
    anchor_nsn_col: str = "nsn",
    anchor_date_col: str = "first_waiver_date",
) -> pl.DataFrame:
    """Stage 4: join anchor (nsn, waiver_date) onto DLA transactions and compute
    day-precise event_year per row. Rows whose NSN is absent from anchor_df get
    null waiver_date -> EVENT_YEAR_SENTINEL.

    anchor_df must have at least columns [anchor_nsn_col, anchor_date_col].
    """
    dla_with_anchor = dla_tx.join(
        anchor_df.select([
            pl.col(anchor_nsn_col).alias("_nsn_join"),
            pl.col(anchor_date_col),
        ]),
        left_on="NSN", right_on="_nsn_join", how="left",
    )
    return dla_with_anchor.select(
        pl.col("NSN").alias("nsn"),
        pl.col("FSC").alias("fsc"),
        pl.col("BASE_PIID").alias("piid"),
        pl.col("mod_n").alias("mod"),
        pl.col("line_n").alias("line"),
        pl.col("AWARD_DATE").alias("action_date"),
        fed_fy(pl.col("AWARD_DATE")).alias("fy"),
        event_year_expr("AWARD_DATE", waiver_date_col=anchor_date_col),
        pl.col("ORDER_QTY").cast(pl.Float64).alias("qty"),
        pl.col("NETPRICE").cast(pl.Float64).alias("unit_price"),
        pl.col("federal_action_obligation").cast(pl.Float64).alias("obligation"),
        pl.col("number_of_offers_received").cast(pl.Float64).alias("offers"),
        pl.col("place_of_manufacture_code").alias("mfg_code"),
        pl.col("extent_competed_code").alias("extent_competed"),
        pl.col("CAGE").alias("cage"),
    )


# ============================================================
# BidLink load + standardize (per anchor)
# ============================================================

def load_bidlink(
    bidlink_parq: Path,
    nsn_universe: list[str] | set[str],
    action_date_cutoff: date = ACTION_DATE_CUTOFF_DEFAULT,
    log_filter=None,
) -> pl.DataFrame:
    """Stage 5 load: read BidLink panel, filter to the NSN universe and date cutoff."""
    bl_raw = pl.read_parquet(bidlink_parq)
    print(f"BidLink combined panel: {len(bl_raw):,} rows")
    n = len(bl_raw)
    bl = bl_raw.filter(
        pl.col("bl_nsn").is_in(list(nsn_universe))
        & pl.col("bl_date_parsed").is_not_null()
        & (pl.col("bl_date_parsed") <= action_date_cutoff)
    )
    if log_filter:
        log_filter("bidlink_universe_filter", n, len(bl),
                   f"keep rows for the {len(set(nsn_universe))} NSNs with valid date and action_date <= {action_date_cutoff}")
    return bl


def standardize_bidlink(
    bl_tx: pl.DataFrame,
    anchor_df: pl.DataFrame,
    anchor_nsn_col: str = "nsn",
    anchor_date_col: str = "first_waiver_date",
    log_filter=None,
) -> pl.DataFrame:
    """Stage 5: join anchor (nsn, waiver_date) onto BidLink rows, compute event_year,
    standardize columns, and run BidLink scrape-artifact dedup."""
    bl_with_anchor = bl_tx.join(
        anchor_df.select([
            pl.col(anchor_nsn_col).alias("_nsn_join"),
            pl.col(anchor_date_col),
        ]),
        left_on="bl_nsn", right_on="_nsn_join", how="left",
    )
    bl_std = bl_with_anchor.select(
        pl.col("bl_nsn").alias("nsn"),
        pl.col("bl_nsn").str.slice(0, 4).alias("fsc"),
        pl.col("bl_contract").alias("piid"),
        pl.when((pl.col("bl_order").is_null()) | (pl.col("bl_order").str.strip_chars() == ""))
          .then(pl.lit("BASE"))
          .otherwise(pl.col("bl_order").str.strip_chars())
          .alias("mod"),
        pl.col("bl_line").str.strip_chars().str.strip_chars_start("0").alias("line"),
        pl.col("bl_date_parsed").alias("action_date"),
        fed_fy(pl.col("bl_date_parsed")).alias("fy"),
        event_year_expr("bl_date_parsed", waiver_date_col=anchor_date_col),
        pl.col("bl_qty").cast(pl.Float64).alias("qty"),
        pl.col("bl_price").cast(pl.Float64).alias("unit_price"),
        pl.col("proc_federal_action_obligation").cast(pl.Float64).alias("obligation"),
        pl.col("proc_number_of_offers_received").cast(pl.Float64).alias("offers"),
        pl.col("proc_place_of_manufacture_code").alias("mfg_code"),
        pl.col("proc_extent_competed_code").alias("extent_competed"),
        pl.col("bl_cage").alias("cage"),
    )
    print(f"BidLink standardized: {len(bl_std):,} rows")

    n = len(bl_std)
    bl_std = bl_std.unique(subset=BL_DEDUP_KEY, keep="first")
    if log_filter:
        log_filter("bidlink_internal_dedup", n, len(bl_std), "BidLink scrape-artifact dedup")
    return bl_std


# ============================================================
# Proxy unit-price recovery (Stage 5b)
# ============================================================

def proxy_unit_price_recovery(
    bl_std: pl.DataFrame,
    clin_tags_csv: Path,
    proc_parq: Path,
    min_proxy_obs: int = MIN_PROXY_PRICE_OBS_DEFAULT,
    log_filter=None,
) -> pl.DataFrame:
    """For BidLink rows where bl_price = 0 (admin mods, unfunded orders), recover
    a unit price as mod-0 federal_action_obligation / sum(bl_qty across the
    contract's line items). Applies only to contracts manually tagged as
    single-NSN (clin_tag in {1, 2}) and only when the NSN has at least
    min_proxy_obs valid contracts.

    Returns bl_std with unit_price column overwritten on the base row of each
    eligible (nsn, contract).
    """
    if not clin_tags_csv.exists():
        print(f"  WARN: {clin_tags_csv} not found; skipping proxy recovery.")
        return bl_std

    # 1. Load CLIN tagging file, keep only proxy-valid rows.
    clin_tags = (
        pl.read_csv(clin_tags_csv, infer_schema_length=0)
        .with_columns(
            normalize_nsn(pl.col("nsn")).alias("nsn"),
            pl.col("clin_tag").cast(pl.Int32, strict=False),
        )
        .filter(pl.col("clin_tag").is_in([1, 2]))
        .rename({"piid": "_piid"})
        .select("nsn", "_piid", "clin_tag")
    )
    print(f"  CLIN tagging: {len(clin_tags)} (nsn, contract) pairs tagged 1 or 2 (proxy-valid)")
    proxy_contracts = clin_tags.select(pl.col("_piid").unique()).to_series().to_list()

    # 2. Per (nsn, contract): sum BidLink qty across all line items.
    contract_qty = (
        bl_std
        .filter(pl.col("piid").is_in(proxy_contracts))
        .filter(pl.col("qty").is_not_null() & (pl.col("qty") > 0))
        .group_by(["nsn", "piid"])
        .agg(pl.col("qty").sum().alias("contract_total_bidlink_qty"))
    )

    # 3. Per contract: mod-0 federal_action_obligation from procurement parquet.
    mod0_obligations = (
        pl.scan_parquet(proc_parq)
        .select([
            "award_id_piid",
            "contract_transaction_unique_key",
            "modification_number",
            "federal_action_obligation",
        ])
        .filter(pl.col("award_id_piid").is_in(proxy_contracts))
        .filter(pl.col("modification_number").cast(pl.Utf8) == "0")
        .collect(engine="streaming")
        .unique(subset=["contract_transaction_unique_key"], keep="first")
        .group_by("award_id_piid")
        .agg(pl.col("federal_action_obligation")
               .cast(pl.Float64, strict=False)
               .sum()
               .alias("contract_mod0_obligation"))
        .rename({"award_id_piid": "_piid"})
    )

    # 4. Compute proxy price per (nsn, contract).
    proxy_per_contract = (
        clin_tags
        .join(contract_qty, left_on=["nsn", "_piid"], right_on=["nsn", "piid"], how="inner")
        .join(mod0_obligations, on="_piid", how="inner")
        .filter(
            (pl.col("contract_total_bidlink_qty") > 0)
            & pl.col("contract_mod0_obligation").is_not_null()
            & (pl.col("contract_mod0_obligation") > 0)
        )
        .with_columns(
            (pl.col("contract_mod0_obligation") / pl.col("contract_total_bidlink_qty"))
            .alias("proxy_unit_price")
        )
        .rename({"_piid": "piid"})
        .select("nsn", "piid", "proxy_unit_price")
    )
    print(f"  proxy_per_contract: {len(proxy_per_contract)} contracts with valid proxy unit prices")

    # 5. Require >= min_proxy_obs valid contracts per NSN.
    nsn_proxy_counts = proxy_per_contract.group_by("nsn").len().rename({"len": "n_proxy"})
    eligible_nsns = (
        nsn_proxy_counts.filter(pl.col("n_proxy") >= min_proxy_obs)
        .select("nsn").to_series().to_list()
    )
    print(f"  NSNs eligible for proxy (>= {min_proxy_obs} valid contracts): {len(eligible_nsns)}")
    if len(eligible_nsns) > 0:
        for n_id in sorted(eligible_nsns):
            n_contracts = nsn_proxy_counts.filter(pl.col("nsn") == n_id)["n_proxy"][0]
            print(f"    {n_id}: {n_contracts} proxy contracts")

    proxy_final = proxy_per_contract.filter(pl.col("nsn").is_in(eligible_nsns))

    # 6. Base row per (nsn, contract): lexicographically smallest line.
    base_rows = (
        bl_std
        .filter(pl.col("piid").is_in(proxy_final["piid"].to_list()))
        .group_by(["nsn", "piid"])
        .agg(pl.col("line").cast(pl.Utf8).sort().first().alias("base_line"))
    )

    # 7. Inject proxy_unit_price into bl_std on base rows where unit_price is missing/zero.
    n_before_proxy = (
        bl_std.filter(pl.col("unit_price").is_not_null() & (pl.col("unit_price") > 0)).height
    )
    bl_std = (
        bl_std
        .join(proxy_final, on=["nsn", "piid"], how="left")
        .join(base_rows, on=["nsn", "piid"], how="left")
        .with_columns(
            pl.when(
                pl.col("proxy_unit_price").is_not_null()
                & (pl.col("line").cast(pl.Utf8) == pl.col("base_line"))
                & ((pl.col("unit_price").is_null()) | (pl.col("unit_price") <= 0))
            )
            .then(pl.col("proxy_unit_price"))
            .otherwise(pl.col("unit_price"))
            .alias("unit_price")
        )
        .drop(["proxy_unit_price", "base_line"])
    )
    n_after_proxy = (
        bl_std.filter(pl.col("unit_price").is_not_null() & (pl.col("unit_price") > 0)).height
    )
    n_injected = n_after_proxy - n_before_proxy
    print(f"  proxy unit prices injected into bl_std: {n_injected} rows "
          f"(rows with unit_price > 0: {n_before_proxy:,} -> {n_after_proxy:,})")
    if log_filter:
        log_filter("bidlink_proxy_unit_price", n_before_proxy, n_after_proxy,
                   f"proxy unit price injection for {len(eligible_nsns)} eligible NSN(s)")
    return bl_std


# ============================================================
# Aggregation to (..., fy, event_year) cells
# ============================================================

def aggregate_to_panel(
    df: pl.DataFrame,
    group_keys: list[str] | None = None,
    label: str = "",
) -> pl.DataFrame:
    """Stage 7: aggregate transaction-level rows to cells keyed by group_keys.
    Default group_keys = ["nsn", "fy", "event_year"] (event-study panel shape).
    For the matched-controls anchored panel, pass ["anchor_nsn", "nsn", "fy", "event_year"]
    after adding an anchor_nsn column upstream.

    Cell-split design: where the event_year boundary falls inside an FY, that
    FY gets multiple sub-cells, each with uniform event_year by construction,
    preserving day-precise event time across aggregation.

    Outcomes:
      - max_log_unit_price          (price)
      - competitive_share           (competition; built from extent_competed_code)
      - domestic_share              (BAA)
    Secondary / diagnostic:
      - mean_offers, n_offers_obs
      - sole_source_share
    """
    if group_keys is None:
        group_keys = ["nsn", "fy", "event_year"]
    print(f"  Aggregating {label}: {len(df):,} rows -> {group_keys} cells...")
    # FPDS v1.4 competition codes per data_defs/competition_fields.md:
    #   competed     = {A, D, F, CDO}
    #   not_competed = {B, C, G, NDO}
    extent_competitive = pl.col("extent_competed").is_in(["A", "D", "F", "CDO"])
    extent_sole_source = pl.col("extent_competed").is_in(["B", "C", "G", "NDO"])
    extent_reported = pl.col("extent_competed").is_not_null() & (pl.col("extent_competed").str.strip_chars() != "")
    agg = (
        df.group_by(group_keys)
        .agg(
            pl.col("fsc").mode().first().alias("fsc"),
            pl.len().alias("n_transactions"),
            pl.col("unit_price").filter(pl.col("unit_price") > 0).max().log().alias("max_log_unit_price"),
            pl.col("unit_price").filter(pl.col("unit_price") > 0).len().alias("n_unit_price_obs"),
            extent_reported.sum().alias("n_extent_reported"),
            (extent_reported & extent_competitive).sum().alias("n_competitive"),
            (extent_reported & extent_sole_source).sum().alias("n_sole_source"),
            pl.col("mfg_code").is_not_null().sum().alias("n_mfg_obs"),
            (pl.col("mfg_code") == "D").sum().alias("n_domestic"),
            pl.col("offers").mean().alias("mean_offers"),
            pl.col("offers").is_not_null().sum().alias("n_offers_obs"),
            pl.col("obligation").sum().alias("total_obligation"),
        )
        .with_columns(
            (pl.col("n_domestic") / pl.col("n_mfg_obs").cast(pl.Float64)).alias("domestic_share"),
            (pl.col("n_competitive") / pl.col("n_extent_reported").cast(pl.Float64)).alias("competitive_share"),
            (pl.col("n_sole_source") / pl.col("n_extent_reported").cast(pl.Float64)).alias("sole_source_share"),
        )
    )
    print(f"  -> {len(agg):,} cells")
    return agg
