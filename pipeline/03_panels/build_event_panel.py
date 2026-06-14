"""
build_event_panel.py

Build NSN x FY panels for the event study, using the FULL DLA enriched parquet
as the base (~13.5M rows, ~952k NSNs) and a transaction-record enrichment
overlay for the 91 NSNs with line-item coverage (28 treated + 63 controls).

Outputs:
  - data/panels/panel_dla_only.parquet   NSN x FY panel from DLA only (every DLA NSN)
  - data/panels/panel_enriched.parquet   same, but the 91 covered NSNs use DLA + overlay
  - results/main/panels/filter_log.csv   every filter / dedup logged with row counts
  - results/main/panels/dla_dedup_anomalies.csv  DLA dup groups where qty/price/obligation vary
"""

from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime, date
import re
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from panel_build_lib import (
    EVENT_YEAR_SENTINEL,
    normalize_nsn,
    fed_fy,
    assert_schema_match,
    load_dla,
    standardize_dla,
    load_bidlink,
    standardize_bidlink,
    proxy_unit_price_recovery,
    aggregate_to_panel,
)

# Cutoff: drop any transaction with action_date AFTER this date.
# 2026-01-31 chosen to trim FPDS reporting-lag tail in FY2026; Oct 2025-Jan 2026
# remains as a partial FY2026 (4 months) flagged via is_partial_fy.
ACTION_DATE_CUTOFF = date(2026, 1, 31)

# -------------------------- paths --------------------------
import paths as P
from control_nsns import control_donor_nsns

TREATED_CSV   = P.TREATMENT_DATES
WAIVER_CSV    = P.WAIVERS_CLEANED
BIDLINK_PARQ  = P.COMBINED_BIDLINK_PANEL
DLA_PARQ      = P.DLA_ENRICHED_LATEST
CLIN_TAGS_CSV = P.UNIT_COST_CLIN_TAGGING
PROC_PARQ     = P.PROCUREMENT_PARQUET

# Proxy unit-price recovery: when BidLink reports bl_price = 0 (admin mods,
# unfunded orders), we can recover a unit price as
#   mod-0 federal_action_obligation / sum(bl_qty across the contract's line items)
# but only when (a) the contract covers a single NSN (clin_tag in {1, 2} per
# the manually validated unit_cost_clin_tagging.csv) and (b) the NSN has at
# least MIN_PROXY_PRICE_OBS such valid contracts, so we have more than one
# observation behind the price series. Mirrors the mechanism in
# pipeline/02_treatment/identify_treatment_nsns.py.
MIN_PROXY_PRICE_OBS = 2

OUT_DIR        = P.PANELS
LOGS_DIR       = P.panel_logs("main")
PANEL_DLA_ONLY  = P.PANEL_DLA_ONLY
PANEL_ENRICHED  = P.PANEL_ENRICHED
FILTER_LOG_OUT  = LOGS_DIR / "filter_log.csv"
DLA_ANOM_OUT    = LOGS_DIR / "dla_dedup_anomalies.csv"
NSN_UNIVERSE_OUT = LOGS_DIR / "nsn_universe.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------- filter log --------------------------
_FILTER_LOG: list[dict] = []
def log_filter(stage: str, n_in, n_out: int, reason: str) -> None:
    n_in_int = n_in if isinstance(n_in, int) else None
    n_dropped = (n_in_int - n_out) if n_in_int is not None else None
    pct = (round(100.0 * n_dropped / max(n_in_int, 1), 4)) if n_in_int is not None else None
    _FILTER_LOG.append({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "stage": stage, "n_in": str(n_in), "n_out": n_out,
        "n_dropped": n_dropped, "pct_dropped": pct, "reason": reason,
    })
    if n_in_int is not None:
        print(f"[FILTER] {stage}: {n_in_int:>12,} -> {n_out:>12,} ({n_dropped:>10,} dropped, {pct}%) | {reason}")
    else:
        print(f"[FILTER] {stage}: {n_in!s:>15} -> {n_out:>12,} | {reason}")

# Utility helpers (normalize_nsn, fed_fy, assert_schema_match) are imported
# from panel_build_lib.

# ============================================================
# Stage 1: NSN universe (treated + BidLink-covered)
# ============================================================
print("\n=== Stage 1: NSN universe ===")

treated = (
    pl.read_csv(TREATED_CSV)
    .with_columns(
        normalize_nsn(pl.col("nsn")).alias("nsn"),
        pl.col("first_waiver_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
    )
    .with_columns(
        pl.lit(1, dtype=pl.Int8).alias("treated"),
        fed_fy(pl.col("first_waiver_date")).alias("waiver_fy"),
    )
    .select(["nsn", "treated", "first_waiver_date", "waiver_fy"])
)
print(f"Treated NSNs: {len(treated)}")

# Per-treated-NSN waiver-coverage classification.
# `has_multiproc = 1` iff the NSN has at least one matched waiver with
# Waiver_Coverage = "Multi-procurement waiver" (formal ongoing-coverage waiver).
# Otherwise 0 (instant-only individual waivers, or no matched waiver — the French
# NATO NSN 2895145543146 falls in the latter and defaults to 0).
# Carried on the panel for the waiver-coverage classification.
print("Classifying treated NSNs by waiver coverage type...")
_w = pl.read_csv(WAIVER_CSV, infer_schema_length=10000)
def _classify(nsn_no_dash: str) -> int:
    nsn_dashed = f"{nsn_no_dash[:4]}-{nsn_no_dash[4:6]}-{nsn_no_dash[6:9]}-{nsn_no_dash[9:]}"
    # Word-boundary regex on the no-dash form — without it, a 13-digit NSN can
    # false-match inside a longer numeric string (e.g., a contract number).
    # The dashed form has natural delimiters and is matched literally.
    pat_no_dash = r"\b" + re.escape(nsn_no_dash) + r"\b"
    cols = ["Waiver_Title",
            "Summary_of_Procurement_(500_word_max)",
            "Procurement_Instrument_Identifier(s)_(PIID)_for_this_waiver_(if_applicable)",
            "Solicitation_ID"]
    for col in cols:
        if col not in _w.columns:
            continue
        hits = _w.filter(
            pl.col(col).str.contains(nsn_dashed, literal=True)
            | pl.col(col).str.contains(pat_no_dash)
        )
        if (hits["Waiver_Coverage"] == "Multi-procurement waiver").any():
            return 1
    return 0
treated = treated.with_columns(
    pl.col("nsn").map_elements(_classify, return_dtype=pl.Int8).alias("has_multiproc")
)
del _w
n_mp = int(treated["has_multiproc"].sum())
print(f"  has_multiproc=1 (multi-procurement waivered): {n_mp}/{len(treated)}")
print(f"  has_multiproc=0 (instant-only or unmatched):  {len(treated) - n_mp}/{len(treated)}")

bl_covered = (
    pl.DataFrame({"nsn": control_donor_nsns()})
    .with_columns(normalize_nsn(pl.col("nsn")).alias("nsn"))
    .unique()
)
print(f"BidLink-covered non-treated NSNs: {len(bl_covered)}")

# 91-NSN universe: treated + BL-covered. These are the rows that will be enriched.
enriched_nsns = pl.concat([treated.select("nsn"), bl_covered]).unique()["nsn"].to_list()
print(f"Enriched (BidLink-overlay) NSN set: {len(enriched_nsns)}")

# Universe table for diagnostics
universe = (
    pl.concat([
        treated.select(["nsn", "treated", "first_waiver_date"]),
        bl_covered.with_columns(
            pl.lit(0, dtype=pl.Int8).alias("treated"),
            pl.lit(None, dtype=pl.Date).alias("first_waiver_date"),
        ).select(["nsn", "treated", "first_waiver_date"]),
    ])
    .with_columns(pl.lit(True).alias("bidlink_enriched"))
    # Deterministic row order (treated first, then NSN): the donor rows come
    # out of .unique() in uncontracted order, which reshuffled this committed
    # diagnostic on every run. Sort only affects the written CSV.
    .sort(["treated", "nsn"], descending=[True, False])
)
universe.write_csv(NSN_UNIVERSE_OUT)
print(f"Wrote {NSN_UNIVERSE_OUT.relative_to(P.REPO_ROOT)}")

TREATED_NSNS = set(treated["nsn"].to_list())
ENRICHED_NSNS = set(enriched_nsns)

# ============================================================
# Stage 2-3: load + safe dedup DLA (delegated to panel_build_lib)
# ============================================================
print("\n=== Stage 2-3: load full DLA + line-level safe dedup ===")
dla, _anomalies = load_dla(
    DLA_PARQ,
    action_date_cutoff=ACTION_DATE_CUTOFF,
    log_filter=log_filter,
    anomalies_out=DLA_ANOM_OUT,
)

# ============================================================
# Stage 4: standardize DLA into common schema (with day-precise event_year)
# ============================================================
print("\n=== Stage 4: standardize DLA ===")
# anchor_df = treated (28 NSNs with first_waiver_date). DLA rows for untreated
# NSNs left-join to null and fall through to the sentinel in event_year_expr().
dla_std = standardize_dla(dla, treated.select(["nsn", "first_waiver_date"]))
del dla
print(f"DLA standardized: {len(dla_std):,} rows")
n_sentinel = (dla_std["event_year"] == EVENT_YEAR_SENTINEL).sum()
n_treated_rows = len(dla_std) - n_sentinel
print(f"  event_year populated: {n_treated_rows:,} treated rows, {n_sentinel:,} control rows (sentinel={EVENT_YEAR_SENTINEL})")

# ============================================================
# Stage 5: load + standardize BidLink (only the 91 NSNs that have it)
# ============================================================
print("\n=== Stage 5: load + standardize BidLink ===")
bl = load_bidlink(BIDLINK_PARQ, ENRICHED_NSNS, action_date_cutoff=ACTION_DATE_CUTOFF, log_filter=log_filter)
bl_std = standardize_bidlink(bl, treated.select(["nsn", "first_waiver_date"]), log_filter=log_filter)
del bl

# ============================================================
# Stage 5b: proxy unit-price recovery for treated NSNs
# ============================================================
print("\n=== Stage 5b: proxy unit-price recovery (treated NSNs only) ===")
bl_std = proxy_unit_price_recovery(
    bl_std,
    clin_tags_csv=CLIN_TAGS_CSV,
    proc_parq=PROC_PARQ,
    min_proxy_obs=MIN_PROXY_PRICE_OBS,
    log_filter=log_filter,
)

# ============================================================
# Stage 6: BidLink overlay = BidLink rows that are NOT already in DLA
# ============================================================
print("\n=== Stage 6: identify BidLink-only rows (not present in DLA) ===")
overlay_keys = bl_std.select(["nsn","piid","mod","line","action_date"])
dla_91 = dla_std.filter(pl.col("nsn").is_in(list(ENRICHED_NSNS))).select(["nsn","piid","mod","line","action_date"])
bl_only = bl_std.join(dla_91, on=["nsn","piid","mod","line","action_date"], how="anti")
print(f"  BidLink rows total (91 NSNs): {len(bl_std):,}")
print(f"  DLA rows for 91 NSNs:        {dla_91.height:,}")
print(f"  BidLink-only (no DLA match): {len(bl_only):,}")
log_filter("bidlink_only_overlay", len(bl_std), len(bl_only), "BidLink rows with no matching DLA line — appended to enriched dataset")
del overlay_keys, dla_91

# ============================================================
# Stage 7: aggregate to NSN x FY panel (aggregate_to_panel from panel_build_lib)
# ============================================================
print("\n=== Stage 7: aggregate to NSN x FY panel ===")

# Panel A: pure DLA
panel_dla = aggregate_to_panel(dla_std, label="DLA-only")

# Panel B: enriched. For the 91 NSNs, recompute the cells using DLA + BidLink-only overlay.
print("Building enriched panel (DLA + BidLink overlay for 91 NSNs)...")
# Slice DLA to 91 NSNs, append BidLink-only rows, aggregate
dla_91_full = dla_std.filter(pl.col("nsn").is_in(list(ENRICHED_NSNS)))
# Schema-equality assertion catches silent dtype upcasts that vertical_relaxed
# would otherwise paper over (e.g., int->float, string->categorical).
assert_schema_match(dla_91_full, bl_only, "dla_91_full", "bl_only")
enriched_91_tx = pl.concat([dla_91_full, bl_only], how="vertical_relaxed")
print(f"  91-NSN merged transactions: {len(enriched_91_tx):,} rows (DLA: {len(dla_91_full):,} + BL-only: {len(bl_only):,})")
panel_91_enriched = aggregate_to_panel(enriched_91_tx, label="91-NSN enriched")

# Replace the 91 NSNs' cells in panel_dla with the enriched cells, AND add any new NSNs
panel_enriched = pl.concat([
    panel_dla.filter(~pl.col("nsn").is_in(list(ENRICHED_NSNS))),
    panel_91_enriched,
], how="vertical_relaxed")
print(f"Enriched panel total cells: {len(panel_enriched):,} (vs DLA-only: {len(panel_dla):,})")

# ============================================================
# Stage 8: tag treated / event_year / bidlink_enriched
# ============================================================
print("\n=== Stage 8: tag treated, event_year, bidlink_enriched ===")

def tag(panel: pl.DataFrame, label: str) -> pl.DataFrame:
    # event_year is already correct from the transaction-level join in Stage 4/5
    # (day-precise for treated, EVENT_YEAR_SENTINEL=1000 for untreated). Don't recompute.
    p = panel.join(
        treated.select(["nsn", "first_waiver_date", "waiver_fy", "has_multiproc"]),
        on="nsn", how="left"
    ).with_columns(
        pl.col("first_waiver_date").is_not_null().cast(pl.Int8).alias("treated"),
        pl.col("nsn").is_in(list(ENRICHED_NSNS)).alias("bidlink_enriched"),
        # FY2026 is a partial fiscal year (Oct 2025 - Jan 2026 only, ~4 months);
        # n_transactions/obligation in these cells are mechanically smaller. Flag for
        # downstream analysis (drop or weight differently in robustness checks).
        (pl.col("fy") == 2026).alias("is_partial_fy"),
        # has_multiproc fills with 0 for control NSNs (vacuously not multi-procurement-waivered).
        pl.col("has_multiproc").fill_null(0).cast(pl.Int8).alias("has_multiproc"),
    )
    n_treated_cells = p.filter(pl.col("treated") == 1).height
    n_treated_nsns = p.filter(pl.col("treated") == 1).select("nsn").n_unique()
    n_control_nsns = p.filter(pl.col("treated") == 0).select("nsn").n_unique()
    print(f"  {label}: {len(p):,} cells | treated NSNs: {n_treated_nsns}, control NSNs: {n_control_nsns:,}, treated cells: {n_treated_cells:,}")
    return p

panel_dla_final = tag(panel_dla, "panel_dla_only")
panel_enriched_final = tag(panel_enriched, "panel_enriched")

# Final col order
COLS = ["nsn", "fy", "fsc", "treated", "bidlink_enriched", "is_partial_fy", "has_multiproc",
        "first_waiver_date", "waiver_fy", "event_year",
        "n_transactions", "max_log_unit_price", "n_unit_price_obs",
        "n_extent_reported", "n_competitive", "n_sole_source", "competitive_share", "sole_source_share",
        "n_mfg_obs", "n_domestic", "domestic_share",
        "mean_offers", "n_offers_obs",
        "total_obligation"]
panel_dla_final = panel_dla_final.select(COLS).sort(["nsn", "fy", "event_year"])
panel_enriched_final = panel_enriched_final.select(COLS).sort(["nsn", "fy", "event_year"])

# ============================================================
# Stage 9: write outputs + summary
# ============================================================
print("\n=== Stage 9: write outputs ===")
panel_dla_final.write_parquet(PANEL_DLA_ONLY, compression="zstd", compression_level=3)
print(f"Wrote {PANEL_DLA_ONLY.relative_to(P.REPO_ROOT)} ({PANEL_DLA_ONLY.stat().st_size/1e6:.1f} MB)")
panel_enriched_final.write_parquet(PANEL_ENRICHED, compression="zstd", compression_level=3)
print(f"Wrote {PANEL_ENRICHED.relative_to(P.REPO_ROOT)} ({PANEL_ENRICHED.stat().st_size/1e6:.1f} MB)")
pl.DataFrame(_FILTER_LOG).write_csv(FILTER_LOG_OUT)
print(f"Wrote {FILTER_LOG_OUT.relative_to(P.REPO_ROOT)}")

print("\n=== Summary ===")
print(f"DLA-only panel:  {len(panel_dla_final):,} cells, {panel_dla_final.select('nsn').n_unique():,} NSNs")
print(f"Enriched panel:  {len(panel_enriched_final):,} cells, {panel_enriched_final.select('nsn').n_unique():,} NSNs")
print(f"Treated NSNs in DLA-only:  {panel_dla_final.filter(pl.col('treated')==1).select('nsn').n_unique()}")
print(f"Treated NSNs in enriched:  {panel_enriched_final.filter(pl.col('treated')==1).select('nsn').n_unique()}")

print("\nDONE.")
