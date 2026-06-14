"""build_anchored_panel.py - Per-treated transaction-aggregated panel with
day-precise event_year for treated AND donor NSNs.

For each of the 28 treated NSNs, computes event_year per-transaction relative
to THAT treated's first_waiver_date for every NSN in scope (treated + donor
universe), then aggregates to (nsn, fy, event_year) cells and tags with
anchor_nsn.

Why this exists: panel_enriched.parquet's event_year is anchored to each NSN's
own first_waiver_date (null/sentinel for donors). Donor rows therefore have
event_year = 1000 (constant), losing the day-precision needed for correct
pre/post assignment in the matched-controls DiD. This script re-anchors every
NSN to each treated's date, producing the correct day-precise event_year for
the donor side of every (treated, donor) pair.

Variants (--variant):
  main      DLA base + transaction-record overlay for the 91-NSN universe
            (28 treated + 63 covered donors); proxy unit-price recovery
            applied to the overlay before merging.
            -> data/anchored/anchored_panel.parquet
  dla_only  DLA transactions are the sole source (overlay removed). Used by
            the dla_only robustness variant.
            -> data/anchored/anchored_panel_dla_only.parquet

Both variants: output keyed by (anchor_nsn, nsn, fy, event_year); outcome
columns identical to panel_enriched.parquet; pre-filtered to event_year in
[-5, +10] per anchor. The overlay conditional is the ONLY behavioral
difference between variants (verified against the original two-script
implementation, which this file unifies).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from panel_build_lib import (  # noqa: E402
    normalize_nsn,
    assert_schema_match,
    load_dla,
    standardize_dla,
    load_bidlink,
    standardize_bidlink,
    proxy_unit_price_recovery,
    aggregate_to_panel,
)
import paths as P  # noqa: E402
from control_nsns import control_donor_nsns  # noqa: E402

# -------------------------- paths --------------------------
TREATED_CSV   = P.TREATMENT_DATES
DONOR_CSV     = P.donor_universe_path("main")
BIDLINK_PARQ  = P.COMBINED_BIDLINK_PANEL
DLA_PARQ      = P.DLA_ENRICHED_LATEST
CLIN_TAGS_CSV = P.UNIT_COST_CLIN_TAGGING
PROC_PARQ     = P.PROCUREMENT_PARQUET

ACTION_DATE_CUTOFF = date(2026, 1, 31)
# Per-anchor EY window. Pre-side fixed at -5 to match the covariate stage's
# pre-window and the strict-coverage check; post-side capped at +10 (generous
# beyond longest sensible analysis horizon) to keep the file compact.
EVENT_YEAR_WINDOW = (-5, 10)
MIN_PROXY_PRICE_OBS = 2

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


def recompute_event_year(df: pl.DataFrame, anchor_date: date) -> pl.DataFrame:
    """Overwrite event_year with floor((action_date - anchor_date)/365.25).
    Day-precise. anchor_date is the treated NSN's first_waiver_date."""
    return df.with_columns(
        (((pl.col("action_date").cast(pl.Date) - pl.lit(anchor_date))
          .dt.total_days() / 365.25)
         .floor().cast(pl.Int32))
        .alias("event_year")
    )


def main(variant: str) -> None:
    with_overlay = variant == "main"
    out_panel = P.anchored_panel_path(variant)
    logs_dir = P.matched_logs(variant)
    filter_log_path = logs_dir / "stage0_filter_log.csv"
    out_panel.parent.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Anchored panel build (variant: {variant}) ===")
    print(f"  cutoff: action_date <= {ACTION_DATE_CUTOFF}")
    print(f"  event_year window: [{EVENT_YEAR_WINDOW[0]}, {EVENT_YEAR_WINDOW[1]}]")
    print(f"  transaction overlay: {'ON' if with_overlay else 'OFF (DLA only)'}")

    # ----- Load treated + donor universe -----
    print("\n=== Treated + donor universe ===")
    treated_df = (
        pl.read_csv(TREATED_CSV, schema_overrides={"nsn": pl.Utf8})
        .with_columns(
            normalize_nsn(pl.col("nsn")).alias("nsn"),
            pl.col("first_waiver_date").cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        )
        .select(["nsn", "first_waiver_date"])
    )
    treated_dates: dict[str, date] = {
        row["nsn"]: row["first_waiver_date"] for row in treated_df.iter_rows(named=True)
    }
    print(f"  treated NSNs: {len(treated_dates)}")

    donors_df = pl.read_csv(
        DONOR_CSV,
        schema_overrides={"nsn": pl.Utf8, "fsg": pl.Utf8, "fsc": pl.Utf8},
    )
    donor_nsns = set(donors_df["nsn"].to_list())
    print(f"  donor universe: {len(donor_nsns):,}")

    tx_universe = sorted(set(treated_dates.keys()) | donor_nsns)
    print(f"  transaction NSN universe (treated + donors): {len(tx_universe):,}")

    # 91-NSN overlay universe — same source as build_event_panel.py
    # so the panel construction stays comparable.
    bl_covered = (
        pl.DataFrame({"nsn": control_donor_nsns()})
        .with_columns(normalize_nsn(pl.col("nsn")).alias("nsn"))
        .unique()
    )
    enriched_nsns = sorted(set(treated_dates.keys()) | set(bl_covered["nsn"].to_list()))
    print(f"  overlay-enriched NSNs (28 treated + covered donors): {len(enriched_nsns)}")

    # ----- Load DLA (anchor-agnostic) + filter to tx universe -----
    print("\n=== Load DLA ===")
    dla, _anomalies = load_dla(
        DLA_PARQ,
        action_date_cutoff=ACTION_DATE_CUTOFF,
        log_filter=log_filter,
    )
    n = len(dla)
    dla = dla.filter(pl.col("NSN").is_in(tx_universe))
    log_filter("dla_filter_to_universe", n, len(dla),
               f"keep rows for {len(tx_universe):,} NSNs (treated + donors)")

    if with_overlay:
        # ----- Load BidLink (anchor-agnostic) -----
        print("\n=== Load BidLink ===")
        bl = load_bidlink(BIDLINK_PARQ, enriched_nsns, action_date_cutoff=ACTION_DATE_CUTOFF, log_filter=log_filter)
    else:
        # DLA-ONLY: transaction overlay removed
        # (BidLink load skipped — DLA transactions are the sole source here.)
        bl = None

    # ----- Standardize once with an empty anchor -----
    # event_year is filled with EVENT_YEAR_SENTINEL by the sentinel branch of
    # event_year_expr (since the left-join finds no anchor rows). We then
    # overwrite event_year per anchor below. action_date is preserved on each
    # row so per-anchor recomputation is just an arithmetic over_with_columns.
    print("\n=== Standardize DLA + BidLink with empty anchor (event_year filled with sentinel; overwritten per-anchor below) ===")
    empty_anchor = pl.DataFrame({
        "nsn": pl.Series([], dtype=pl.Utf8),
        "first_waiver_date": pl.Series([], dtype=pl.Date),
    })
    dla_std_base = standardize_dla(dla, empty_anchor)
    del dla
    print(f"  DLA standardized: {len(dla_std_base):,} rows")

    if with_overlay:
        bl_std_base = standardize_bidlink(bl, empty_anchor, log_filter=log_filter)
        del bl
        bl_std_base = proxy_unit_price_recovery(
            bl_std_base,
            clin_tags_csv=CLIN_TAGS_CSV,
            proc_parq=PROC_PARQ,
            min_proxy_obs=MIN_PROXY_PRICE_OBS,
            log_filter=log_filter,
        )
        print(f"  BidLink standardized + proxy-patched: {len(bl_std_base):,} rows")
    else:
        # DLA-ONLY: transaction overlay removed
        # (BidLink standardize + proxy unit-price recovery skipped.)
        bl_std_base = None

    # ----- Per-anchor loop -----
    print(f"\n=== Per-anchor aggregation ({len(treated_dates)} anchors, EY window {EVENT_YEAR_WINDOW}) ===")
    anchored_chunks: list[pl.DataFrame] = []

    for i, (tnsn, fwd) in enumerate(sorted(treated_dates.items()), start=1):
        if with_overlay:
            # Recompute event_year for every row relative to this anchor's date.
            dla_std = recompute_event_year(dla_std_base, fwd)
            bl_std = recompute_event_year(bl_std_base, fwd)

            # Overlay = BL rows for enriched NSNs not in DLA.
            dla_enriched_slice = (
                dla_std.filter(pl.col("nsn").is_in(enriched_nsns))
                .select(["nsn", "piid", "mod", "line", "action_date"])
            )
            bl_only = bl_std.join(
                dla_enriched_slice,
                on=["nsn", "piid", "mod", "line", "action_date"],
                how="anti",
            )

            # Concat: full DLA + BL-only rows (covers all NSNs; enriched ones get the overlay).
            if len(bl_only) > 0:
                assert_schema_match(dla_std, bl_only, "dla_std", "bl_only")
                full_tx = pl.concat([dla_std, bl_only], how="vertical_relaxed")
            else:
                full_tx = dla_std
        else:
            # Recompute event_year for every row relative to this anchor's date.
            # DLA-ONLY: transaction overlay removed (no BidLink anti-join / concat).
            full_tx = recompute_event_year(dla_std_base, fwd)

        # Pre-filter to EY window to bound output size before aggregation.
        n_full = len(full_tx)
        full_tx = full_tx.filter(
            pl.col("event_year").is_between(EVENT_YEAR_WINDOW[0], EVENT_YEAR_WINDOW[1], closed="both")
        )

        # Aggregate to (nsn, fy, event_year) cells; tag with anchor.
        panel_i = aggregate_to_panel(
            full_tx,
            group_keys=["nsn", "fy", "event_year"],
            label=f"anchor={tnsn}",
        ).with_columns(pl.lit(tnsn).alias("anchor_nsn"))

        anchored_chunks.append(panel_i)
        print(f"  [{i:02d}/{len(treated_dates)}] anchor {tnsn} (waiver {fwd}): "
              f"{n_full:,} tx -> {len(panel_i):,} cells in window")

    # ----- Concat + write -----
    print("\n=== Concat + write ===")
    result = pl.concat(anchored_chunks, how="vertical_relaxed").select([
        "anchor_nsn", "nsn", "fy", "event_year", "fsc",
        "n_transactions",
        "max_log_unit_price", "n_unit_price_obs",
        "n_extent_reported", "n_competitive", "n_sole_source",
        "competitive_share", "sole_source_share",
        "n_mfg_obs", "n_domestic", "domestic_share",
        "mean_offers", "n_offers_obs",
        "total_obligation",
    ]).sort(["anchor_nsn", "nsn", "fy", "event_year"])

    print(f"  total cells: {len(result):,}")
    print(f"  anchors: {result['anchor_nsn'].n_unique()}")
    print(f"  unique NSNs across all anchors: {result['nsn'].n_unique():,}")
    print(f"  event_year range: [{result['event_year'].min()}, {result['event_year'].max()}]")

    result.write_parquet(out_panel, compression="zstd", compression_level=3)
    print(f"  wrote: {out_panel} ({out_panel.stat().st_size/1e6:.1f} MB)")

    pl.DataFrame(_FILTER_LOG).write_csv(filter_log_path)
    print(f"  wrote: {filter_log_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=["main", "dla_only"], default="main",
                    help="main = DLA + transaction overlay; dla_only = DLA sole source")
    args = ap.parse_args()
    main(args.variant)
