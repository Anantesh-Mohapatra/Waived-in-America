"""Stage 2 - Per-(treated, outcome) balanced-panel donor pools.

TIME DIMENSION = day-precise event_year. Source: anchored_panel.parquet (built
by pipeline/03_panels/build_anchored_panel.py), keyed by
(anchor_nsn, nsn, fy, event_year). event_year is per-transaction relative to
each treated NSN's first_waiver_date; cell-splits exist where EY boundaries
fall mid-FY. We collapse the FY split *inside* an event_year cell here. This
is NOT FY-pooling.

For each (treated NSN, outcome) pair:
  1. Treated's observed EYs = EYs where treated has non-null outcome obs at
     (anchor_nsn=treated, nsn=treated), AFTER aggregating cell-split rows
     within EY.
  2. Donor candidates = donor universe with eligible_<outcome> = 1.
  3. Strict-coverage filter: donor qualifies iff its observed-EY set (under
     same anchor, same outcome) is a superset of treated's observed-EY set.
     NO fixed window. If treated has post-period EYs (e.g., 0..3), donors
     must observe those too.
  4. Aggregate (fy, event_year) -> event_year:
       max_log_unit_price -> max() across cells (skip nulls)
       mean_offers        -> weighted mean by n_offers_obs
       domestic_share     -> weighted mean by n_mfg_obs
  5. Output long parquet: unit_id, event_year, y, n_obs, is_treated.

Treated NSNs expected to drop:
  - 6695-01-266-2248 — waiver 2025-07-10, no pre-period under day-precise EY.
  - 1680-01-622-9189 for max_log_unit_price only — no pre-period unit-price obs.

Coverage rule rationale: synthdid's post-period counterfactual is evaluated at
every post-EY against actual donor data. No imputation. If donor lacks a post-EY,
it can't anchor the counterfactual for that period.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths as P

# ---------- paths (variant-routed) ----------
_ap = argparse.ArgumentParser()
_ap.add_argument("--variant", choices=["main", "dla_only"], default="main")
_VARIANT = _ap.parse_args().variant

ANCHORED_PANEL = P.anchored_panel_path(_VARIANT)
TREATED_CSV = P.TREATMENT_DATES
DONOR_CSV = P.donor_universe_path(_VARIANT)

OUT_DIR = P.DONOR_POOLS / _VARIANT
_LOGS_DIR = P.synth_logs(_VARIANT)
POOL_SIZES_LOG = _LOGS_DIR / "stage2_pool_sizes.csv"
SKIP_LOG = _LOGS_DIR / "stage2_skipped.csv"
_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- constants ----------
OUTCOMES = ("max_log_unit_price", "mean_offers", "domestic_share")
# Cell-counts that determine "observed" and feed weighted-mean aggregation.
COUNT_COL = {
    "max_log_unit_price": "n_unit_price_obs",
    "mean_offers": "n_offers_obs",
    "domestic_share": "n_mfg_obs",
}
# Pre-period window for the "must have >=1 pre-period obs" treated check.
# Same as matched_controls — defines which treated NSNs are estimable at all.
PRE_WINDOW = (-5, -1)
# Minimum number of pre-period EYs (T0) required for a credible synth fit.
# T0=1 is degenerate (synth fits one period trivially); T0=2 is weak.
# T0>=3 is standard practice for synth-control identification.
MIN_PRE_EYS = 3


def to_no_dash(n: str) -> str:
    return n.replace("-", "")


def aggregate_within_ey(df: pl.DataFrame, outcome: str) -> pl.DataFrame:
    """Collapse (fy, event_year) cells to (event_year) per the locked rule.

    Input columns required: nsn, event_year, fy, <outcome>, <count_col>.
    Output: nsn, event_year, y (the aggregated outcome), n_obs.
    Rows where n_obs == 0 are dropped (unit has no observation at this EY).
    """
    cnt = COUNT_COL[outcome]
    if outcome == "max_log_unit_price":
        # Max across cells; null-safe.
        agg = (
            df.group_by(["nsn", "event_year"])
            .agg(
                pl.col(outcome).max().alias("y"),
                pl.col(cnt).fill_null(0).sum().alias("n_obs"),
            )
        )
    else:
        # Weighted mean by count. weight = count_col, value = outcome.
        # numerator = sum(value * count), denominator = sum(count).
        agg = (
            df.group_by(["nsn", "event_year"])
            .agg(
                (
                    (pl.col(outcome).fill_null(0.0) * pl.col(cnt).fill_null(0))
                    .sum()
                ).alias("_num"),
                pl.col(cnt).fill_null(0).sum().alias("n_obs"),
            )
            .with_columns(
                pl.when(pl.col("n_obs") > 0)
                .then(pl.col("_num") / pl.col("n_obs"))
                .otherwise(None)
                .alias("y")
            )
            .drop("_num")
            .select("nsn", "event_year", "y", "n_obs")
        )
    # Drop EYs with zero observations (treated as "not observed" for this outcome).
    agg = agg.filter(pl.col("n_obs") > 0)
    # Also drop where y is null/NaN (defensive).
    agg = agg.filter(pl.col("y").is_not_null() & ~pl.col("y").cast(pl.Float64).is_nan())
    return agg


def main() -> None:
    print("Stage 2 - per-(treated, outcome) balanced-panel donor pools")
    print(f"  panel: {ANCHORED_PANEL}")
    print(f"  treated: {TREATED_CSV}")
    print(f"  donors: {DONOR_CSV}")

    # ----- load -----
    # Lazy scan: anchored_panel.parquet has row-group statistics on anchor_nsn
    # (verified: 83% of row groups contain a single anchor), so predicate
    # pushdown lets each per-anchor .collect() read ~1/28 of the file instead
    # of all 42.5M rows. Peak RAM stays modest (~one anchor's worth at a time).
    needed_cols = [
        "anchor_nsn", "nsn", "fy", "event_year",
        "max_log_unit_price", "n_unit_price_obs",
        "mean_offers", "n_offers_obs",
        "domestic_share", "n_mfg_obs",
    ]
    panel_lf = pl.scan_parquet(ANCHORED_PANEL).select(needed_cols)
    print(f"  panel: lazy scan of {ANCHORED_PANEL.name}")

    donors_df = pl.read_csv(
        DONOR_CSV,
        schema_overrides={"nsn": pl.Utf8, "fsc": pl.Utf8, "fsg": pl.Utf8},
    )
    donor_eligibility: dict[str, dict[str, int]] = {}
    for row in donors_df.iter_rows(named=True):
        donor_eligibility[row["nsn"]] = {
            o: int(row[f"eligible_{o}"]) for o in OUTCOMES
        }
    print(f"  donor universe: {len(donor_eligibility):,} NSNs")

    treated_df = pl.read_csv(TREATED_CSV, schema_overrides={"nsn": pl.Utf8})
    treated_dates: dict[str, date] = {}
    for row in treated_df.iter_rows(named=True):
        nsn_no_dash = to_no_dash(row["nsn"])
        v = row["first_waiver_date"]
        if isinstance(v, date) and not isinstance(v, datetime):
            d = v
        elif isinstance(v, datetime):
            d = v.date()
        else:
            d = datetime.fromisoformat(str(v)).date()
        treated_dates[nsn_no_dash] = d
    print(f"  treated NSNs: {len(treated_dates)}")
    print()

    pool_log_rows: list[dict] = []
    skip_log_rows: list[dict] = []

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Wipe stale pool parquets. Safe because Stage 2 always produces a complete
    # set of pools derived from upstream state; a previous run with looser
    # filters (e.g., before MIN_PRE_EYS=3 was added) would otherwise leave
    # ghost pools that downstream stages silently pick up.
    for stale in OUT_DIR.glob("*.parquet"):
        stale.unlink()

    # Iterate over treated, then over outcomes inside.
    for i, (tnsn, fwd) in enumerate(sorted(treated_dates.items()), start=1):
        # Predicate-pushdown read: only this anchor's row groups hit disk.
        anchor_slice = panel_lf.filter(pl.col("anchor_nsn") == tnsn).collect()
        # Pre-aggregate keep-set per outcome
        for outcome in OUTCOMES:
            cnt = COUNT_COL[outcome]
            # ----- treated's own aggregated EYs for this outcome -----
            t_rows = anchor_slice.filter(pl.col("nsn") == tnsn).select(
                "nsn", "event_year", "fy", outcome, cnt
            )
            t_agg = aggregate_within_ey(t_rows, outcome)
            t_eys = set(t_agg["event_year"].to_list())

            if not t_eys:
                skip_log_rows.append({
                    "treated_nsn": tnsn, "outcome": outcome,
                    "n_treated_eys": 0,
                    "reason": "treated has no observed EYs for this outcome",
                })
                continue

            # Must have at least one pre-period observation in PRE_WINDOW.
            t_pre_eys = {ey for ey in t_eys
                         if PRE_WINDOW[0] <= ey <= PRE_WINDOW[1]}
            if not t_pre_eys:
                skip_log_rows.append({
                    "treated_nsn": tnsn, "outcome": outcome,
                    "n_treated_eys": len(t_eys),
                    "reason": "treated has no pre-period observation in EY [-5,-1]",
                })
                continue

            # T0 >= MIN_PRE_EYS for credible synth identification.
            # T0=1 is degenerate; T0=2 is weak. Standard floor is T0>=3.
            if len(t_pre_eys) < MIN_PRE_EYS:
                skip_log_rows.append({
                    "treated_nsn": tnsn, "outcome": outcome,
                    "n_treated_eys": len(t_eys),
                    "reason": f"T0={len(t_pre_eys)} pre-EYs < MIN_PRE_EYS={MIN_PRE_EYS}",
                })
                continue

            # ----- candidate donors -----
            cand = [n for n, e in donor_eligibility.items() if e.get(outcome, 0) == 1]
            n_candidates = len(cand)

            # ----- aggregate donors within EY, then coverage check -----
            # Filter to t_eys BEFORE aggregating: aggregating donor rows for
            # event_years outside the treated's observed set is wasted work
            # (we'd drop them at the coverage step anyway).
            t_eys_list = sorted(t_eys)
            d_rows = (
                anchor_slice
                .filter(pl.col("nsn").is_in(cand)
                        & pl.col("event_year").is_in(t_eys_list))
                .select("nsn", "event_year", "fy", outcome, cnt)
            )
            d_agg = aggregate_within_ey(d_rows, outcome)
            donor_ey_sets = (
                d_agg
                .group_by("nsn")
                .agg(pl.col("event_year").n_unique().alias("n_covered"))
            )
            # Donor qualifies iff n_covered == |t_eys|
            qualifying = (
                donor_ey_sets.filter(pl.col("n_covered") == len(t_eys))["nsn"]
                .to_list()
            )
            n_balanced = len(qualifying)

            if n_balanced == 0:
                skip_log_rows.append({
                    "treated_nsn": tnsn, "outcome": outcome,
                    "n_treated_eys": len(t_eys),
                    "reason": f"0 donors with full EY coverage ({n_candidates} candidates)",
                })
                continue

            # ----- build pool parquet -----
            # Treated row: from t_agg, restricted to t_eys (already all of them).
            t_out = t_agg.with_columns(
                pl.lit(True).alias("is_treated"),
            ).rename({"nsn": "unit_id"})

            # Donor rows: d_agg is already filtered to t_eys; restrict to qualifying donors.
            d_out = (
                d_agg
                .filter(pl.col("nsn").is_in(qualifying))
                .with_columns(pl.lit(False).alias("is_treated"))
                .rename({"nsn": "unit_id"})
            )

            combined = pl.concat([t_out, d_out], how="vertical_relaxed").select(
                "unit_id", "event_year", "y", "n_obs", "is_treated"
            )
            # Sanity: every (unit, EY) should be present. Verify panel is rectangular.
            n_units = combined["unit_id"].n_unique()
            n_eys = combined["event_year"].n_unique()
            if len(combined) != n_units * n_eys:
                # Some donors missing a cell (shouldn't happen post-coverage); recheck.
                cell_counts = combined.group_by("unit_id").len()
                bad = cell_counts.filter(pl.col("len") != n_eys)
                print(f"  [{i:02d}] {tnsn} / {outcome}: WARNING — "
                      f"{len(bad):,} units missing cells after coverage filter; "
                      f"dropping them")
                keep_units = cell_counts.filter(pl.col("len") == n_eys)["unit_id"].to_list()
                combined = combined.filter(pl.col("unit_id").is_in(keep_units))
                n_balanced = len([u for u in keep_units if u != tnsn])

            out_path = OUT_DIR / f"{tnsn}__{outcome}.parquet"
            combined.write_parquet(out_path, compression="zstd")

            pool_log_rows.append({
                "treated_nsn": tnsn, "outcome": outcome,
                "n_treated_eys": len(t_eys),
                "treated_eys": ",".join(str(e) for e in sorted(t_eys)),
                "n_pre_eys": sum(1 for e in t_eys if e <= -1),
                "n_post_eys": sum(1 for e in t_eys if e >= 0),
                "n_candidates_pre_coverage": n_candidates,
                "n_balanced_pool": n_balanced,
                # POSIX separators so this log column is platform-independent.
                "out_path": out_path.relative_to(P.REPO_ROOT).as_posix(),
            })

        if i % 5 == 0 or i == len(treated_dates):
            print(f"  [{i:02d}/{len(treated_dates)}] processed through {tnsn} (waiver {fwd})")

    # ----- write logs -----
    if pool_log_rows:
        pl.DataFrame(pool_log_rows).write_csv(POOL_SIZES_LOG)
        print()
        print(f"  wrote: {POOL_SIZES_LOG} ({len(pool_log_rows)} (treated, outcome) pairs)")
    if skip_log_rows:
        pl.DataFrame(skip_log_rows).write_csv(SKIP_LOG)
        print(f"  wrote: {SKIP_LOG} ({len(skip_log_rows)} skipped pairs)")
    else:
        print(f"  no skipped pairs")

    # ----- summary -----
    if pool_log_rows:
        log_df = pl.DataFrame(pool_log_rows)
        print()
        print("  Per-outcome counts (after Stage 2):")
        for o in OUTCOMES:
            sub = log_df.filter(pl.col("outcome") == o)
            n = len(sub)
            if n:
                avg_pool = sub["n_balanced_pool"].mean()
                med_pool = sub["n_balanced_pool"].median()
                print(f"    {o}: {n} treated, mean pool {avg_pool:,.0f}, median {med_pool:,.0f}")
            else:
                print(f"    {o}: 0 treated")


if __name__ == "__main__":
    main()
