"""Stage 2 - Per-(treated, donor) pre-period matching covariates.

For each treated NSN, define its pre-period as the 5 event-years immediately
before its first_waiver_date (EY -5 through -1, day-precise — event_year is
computed per-transaction relative to the treated's waiver date upstream in
`build_anchored_panel.py`). For each candidate donor (from
donor_universe.csv), apply the strict-coverage rule:

  - Donor must have panel data in every event-year within EY [-5, +5]
    where the treated NSN has panel data. Event-years are anchored to the
    treated NSN's first_waiver_date for both treated and donor.
  - "Active event years" for the treated NSN = the set of EYs in [-5, +5]
    where the treated NSN has at least one panel row.
  - If treated has no active EY in [-5, -1] the treated is dropped (no
    pre-period observations to anchor matching). This is unchanged from
    the prior rule.

For surviving (treated, donor) pairs, compute 4 matching covariates + 1
diagnostic-only covariate + FSG label (used as exact-match stratum at Stage 3
only when toggled on).

Volume (zero-weighted, then log1p; missing FY = real zero):
  log_n_transactions_per_fy

Pre-period mean levels (observed-only), outcome-anchored:
  mean_max_log_unit_price       outcome-anchored (unit price)
  mean_domestic_share           outcome-anchored (domestic sourcing)
  mean_pre_offers               outcome-anchored (offer count)

Diagnostic-only (written to parquet, NOT in Stage 3 COVS):
  mean_competitive_share        procurement-environment proxy; retained as a
                                diagnostic-only column.

Per-outcome eligibility (n_obs counts) is also retained so downstream stages
can drop NSNs with insufficient observations on a specific outcome:
  n_obs_max_log_unit_price, n_obs_mean_offers, n_obs_domestic_share

Output schema (long, one row per NSN appearing in any pair, plus 1 treated row
per group): treated_nsn, nsn, is_treated, fsg, n_pre_fy, n_post_fy,
log_n_transactions_per_fy, mean_max_log_unit_price, mean_domestic_share,
mean_pre_offers, mean_competitive_share, n_obs_<outcome>.

Researcher-judgment notes (also surfaced in README):
- Treated pre-period filter: a treated NSN is dropped here if it has zero
  panel rows in EY [-5, -1]. This is *tighter* than the upstream
  treatment-id filter, which only requires n_pre_waiver > 0 against any
  date. See `results/<variant>/matched/logs/stage2_per_treated_log.csv` for the live drop reasons.
- mean_pre_offers is an outcome-anchored matching covariate (parallel to
  mean_max_log_unit_price and mean_domestic_share). FPDS leaves
  number_of_offers_received blank on ~30% of non-treated cells (mostly
  code-A full-and-open awards), so donors with zero pre-period offer
  observations get NaN mean_pre_offers and are dropped from the matching
  pool by Stage 3's is.finite() filter. About 18% of the donor universe
  drops out under this rule.
- mean_competitive_share is computed and written to the parquet as a
  diagnostic-only column. It is not in Stage 3's COVS list.
- Per-outcome NSN eligibility: any treated NSN whose pre-period has no
  observations on a given outcome will have NaN for the corresponding
  `mean_<outcome>` covariate; Stage 3's `is.finite()` filter will then
  drop that NSN from matching on outcomes that use it as a covariate.
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
_SUFFIX = P.variant_suffix(_VARIANT)

# Anchored panel: keyed by (anchor_nsn, nsn, fy, event_year). event_year is
# day-precise per-transaction, anchored to the treated NSN's first_waiver_date.
# Built by pipeline/03_panels/build_anchored_panel.py.
PANEL = P.anchored_panel_path(_VARIANT)
TREATED_CSV = P.TREATMENT_DATES

DONOR_CSV = P.donor_universe_path("main")
OUT_PARQUET = P.ANCHORED / f"matching_covariates{_SUFFIX}.parquet"
_LOGS_DIR = P.matched_logs(_VARIANT)
COVERAGE_LOG = _LOGS_DIR / "stage2_per_treated_log.csv"
_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- constants ----------
# Outcomes are *not* the same as matching covariates. Outcomes drive
# per-NSN eligibility flags (n_obs_<outcome>) so Stage 3 / 5 can drop
# NSNs with insufficient coverage on a specific outcome (e.g., a treated
# NSN with zero pre-period unit-price observations will have
# n_obs_max_log_unit_price = 0 and a NaN mean_max_log_unit_price, which
# Stage 3's is.finite() filter will drop).
OUTCOMES = ("max_log_unit_price", "mean_offers", "domestic_share")

# Written covariates: 1 volume + 3 outcome-anchored matching covariates +
# 1 diagnostic-only covariate (mean_competitive_share, kept for post-hoc
# comparison with the previous proxy design). Stage 3 matches on the first
# four only — its own COVS list selects from this set.
COVARIATES = (
    "log_n_transactions_per_fy",
    "mean_max_log_unit_price",
    "mean_domestic_share",
    "mean_pre_offers",
    "mean_competitive_share",
)

PRE_WINDOW_LEN = 5
# Window over which we measure treated NSN's "active event years" for the
# strict-coverage donor rule. Pre side fixed at -5 (matching pre-period
# length); post side capped at +5 (standard event-study horizon). Donor
# must have panel rows in every active EY the treated has in this window.
COVERAGE_WINDOW = (-5, 5)
PANEL_COLS = (
    "anchor_nsn", "nsn", "fy", "event_year", "fsc", "n_transactions",
    "max_log_unit_price", "mean_offers", "domestic_share",
    "n_extent_reported", "n_competitive",
)


def to_no_dash(n: str) -> str:
    return n.replace("-", "")


def fed_fy(d: date) -> int:
    """Federal FY: month >= 10 belongs to year+1."""
    return d.year + 1 if d.month >= 10 else d.year


def per_nsn_summary(pre_data: pl.DataFrame, n_pre_window: int = PRE_WINDOW_LEN) -> pl.DataFrame:
    """Compute matching covariates + per-outcome eligibility counts per NSN
    from pre-period rows.

    Returns columns:
      n_pre_fy, log_n_transactions_per_fy,
      mean_max_log_unit_price, mean_domestic_share, mean_pre_offers,
      mean_competitive_share, n_obs_<outcome> for each outcome in OUTCOMES.

    pre_data: rows where event_year in [-5, -1] for a given treated NSN's window.
    """
    # Volume (zero-weighted: sum across pre-period FYs / window length, then log1p).
    volume = pre_data.group_by("nsn").agg(
        ((pl.col("n_transactions").fill_null(0).sum() / n_pre_window) + 1).log().alias("log_n_transactions_per_fy"),
        pl.col("fy").n_unique().alias("n_pre_fy"),
    )
    result = volume

    # Per-outcome: observed-only mean + non-null count for eligibility.
    # All three outcomes double as matching covariates under the current design.
    # Output column naming: `mean_<outcome>` except mean_offers, which becomes
    # `mean_pre_offers` (clearer than the literal `mean_mean_offers`).
    covariate_name_for = {
        "max_log_unit_price": "mean_max_log_unit_price",
        "mean_offers": "mean_pre_offers",
        "domestic_share": "mean_domestic_share",
    }
    for outcome in OUTCOMES:
        cov_name = covariate_name_for[outcome]
        obs = pre_data.filter(pl.col(outcome).is_not_null() & ~pl.col(outcome).is_nan())
        per_o = obs.group_by("nsn").agg(
            pl.col(outcome).mean().alias(cov_name),
            pl.col(outcome).count().alias(f"n_obs_{outcome}"),
        )
        result = result.join(per_o, on="nsn", how="left")
        result = result.with_columns(pl.col(f"n_obs_{outcome}").fill_null(0))

    # mean_competitive_share: covariate only (no eligibility flag). Weighted
    # by transaction counts to avoid NSN-FY cells with 1 transaction skewing
    # the mean. Equivalent to sum(n_competitive) / sum(n_extent_reported).
    comp = (
        pre_data.group_by("nsn").agg(
            pl.col("n_competitive").fill_null(0).sum().alias("_n_comp"),
            pl.col("n_extent_reported").fill_null(0).sum().alias("_n_ext"),
        )
        .with_columns(
            pl.when(pl.col("_n_ext") > 0)
            .then(pl.col("_n_comp") / pl.col("_n_ext"))
            .otherwise(None)
            .alias("mean_competitive_share")
        )
        .select("nsn", "mean_competitive_share")
    )
    result = result.join(comp, on="nsn", how="left")

    return result


def main() -> None:
    print("Stage 2 - per-(treated, donor) pre-period matching covariates")
    print(f"  panel: {PANEL}")
    print(f"  donor universe: {DONOR_CSV}")

    panel = pl.read_parquet(PANEL).select(list(PANEL_COLS))
    print(f"  panel rows: {len(panel):,} ({panel['nsn'].n_unique():,} NSNs)")

    donors_df = pl.read_csv(
        DONOR_CSV,
        schema_overrides={"nsn": pl.Utf8, "fsg": pl.Utf8, "fsc": pl.Utf8},
    )
    donor_nsns = donors_df["nsn"].to_list()
    donor_set_size = len(set(donor_nsns))
    print(f"  donor universe: {donor_set_size:,} NSNs")

    # Map nsn -> fsg (donors already have fsg in donor_universe.csv).
    fsg_map = dict(zip(donors_df["nsn"].to_list(), donors_df["fsg"].to_list()))

    # Treated NSNs + waiver dates (canonicalize to no-dash; CSV is dashed).
    treated_df = pl.read_csv(TREATED_CSV, schema_overrides={"nsn": pl.Utf8})
    treated_dates: dict[str, date] = {}
    for row in treated_df.iter_rows(named=True):
        nsn_no_dash = to_no_dash(row["nsn"])
        # CSV may store as date or string; handle both.
        v = row["first_waiver_date"]
        if isinstance(v, date) and not isinstance(v, datetime):
            d = v
        elif isinstance(v, datetime):
            d = v.date()
        else:
            d = datetime.fromisoformat(str(v)).date()
        treated_dates[nsn_no_dash] = d
    print(f"  treated NSNs: {len(treated_dates)}")

    # Treated FSG: derive modal fsc from the panel.
    treated_fsg = (
        panel.filter(pl.col("nsn").is_in(list(treated_dates.keys())))
        .group_by("nsn")
        .agg(pl.col("fsc").drop_nulls().mode().first().alias("fsc"))
        .with_columns(pl.col("fsc").str.slice(0, 2).alias("fsg"))
    )
    treated_fsg_map = dict(zip(treated_fsg["nsn"].to_list(), treated_fsg["fsg"].to_list()))
    missing_treated_fsg = [n for n in treated_dates if n not in treated_fsg_map]
    if missing_treated_fsg:
        print(f"  WARNING: {len(missing_treated_fsg)} treated NSNs have no panel rows for FSG: {missing_treated_fsg[:5]}")

    # Pre-filter panel rows to donors + treated to keep memory tight.
    # Anchored panel is already keyed by anchor_nsn; we filter per-anchor in the loop.
    keep_set = set(donor_nsns) | set(treated_dates.keys())
    panel_kept = panel.filter(pl.col("nsn").is_in(list(keep_set)))
    print(f"  panel rows after pre-filter to donors+treated: {len(panel_kept):,}")
    print()

    coverage_log_rows: list[dict] = []
    pair_summaries: list[pl.DataFrame] = []

    for i, (tnsn, fwd) in enumerate(sorted(treated_dates.items()), start=1):
        wfy = fed_fy(fwd)

        # Select this anchor's slice of the anchored panel. event_year is
        # already day-precise relative to fwd, baked in at build time.
        d = panel_kept.filter(pl.col("anchor_nsn") == tnsn)

        # Treated's own pre-period summary.
        t_pre = d.filter((pl.col("nsn") == tnsn) & pl.col("event_year").is_between(-5, -1, closed="both"))
        if len(t_pre) == 0:
            print(f"  [{i:02d}] {tnsn} (waiver {fwd}, fy{wfy}): SKIP - no panel pre-period rows")
            coverage_log_rows.append({
                "treated_nsn": tnsn, "first_waiver_date": str(fwd), "waiver_fy": wfy,
                "n_pre_fy_treated": 0,
                "n_active_eys_treated": 0,
                "active_eys_treated": "",
                "n_donors_in_universe": donor_set_size,
                "n_donors_after_coverage": 0, "skipped": True, "reason": "no treated pre-period data",
            })
            continue
        t_summary = per_nsn_summary(t_pre, PRE_WINDOW_LEN)
        n_pre_treated = int(t_summary["n_pre_fy"][0])

        # ----- Strict-coverage donor rule -----
        # Compute the treated NSN's active event-years within COVERAGE_WINDOW.
        # A donor qualifies only if its set of active EYs (under the same
        # waiver-anchored event-year clock) is a superset of the treated's
        # active EYs in this window.
        t_active_eys = set(
            d.filter(
                (pl.col("nsn") == tnsn)
                & pl.col("event_year").is_between(COVERAGE_WINDOW[0], COVERAGE_WINDOW[1], closed="both")
            )["event_year"].unique().to_list()
        )

        d_donors = d.filter(pl.col("nsn").is_in(donor_nsns))
        d_in_window = d_donors.filter(
            pl.col("event_year").is_between(COVERAGE_WINDOW[0], COVERAGE_WINDOW[1], closed="both")
        )
        # Build donor EY-set; donor qualifies iff t_active_eys is a subset.
        donor_eys = (
            d_in_window.group_by("nsn")
            .agg(pl.col("event_year").unique().alias("eys"))
        )
        if len(t_active_eys) == 0:
            surviving = []
        else:
            t_active_list = sorted(t_active_eys)
            t_active_set_size = len(t_active_eys)
            surviving = (
                donor_eys.with_columns(
                    pl.col("eys")
                    .list.set_intersection(pl.lit(t_active_list, dtype=pl.List(pl.Int32)))
                    .list.len()
                    .alias("n_covered")
                )
                .filter(pl.col("n_covered") == t_active_set_size)
            )["nsn"].to_list()

        if not surviving:
            print(f"  [{i:02d}] {tnsn} (waiver {fwd}, fy{wfy}, "
                  f"active_eys={sorted(t_active_eys)}): SKIP - no donors after strict-coverage rule")
            coverage_log_rows.append({
                "treated_nsn": tnsn, "first_waiver_date": str(fwd), "waiver_fy": wfy,
                "n_pre_fy_treated": n_pre_treated,
                "n_active_eys_treated": len(t_active_eys),
                "active_eys_treated": ",".join(str(e) for e in sorted(t_active_eys)),
                "n_donors_in_universe": donor_set_size,
                "n_donors_after_coverage": 0, "skipped": True,
                "reason": "0 donors satisfy strict-coverage rule",
            })
            continue

        # Build donor summary off donor pre-period rows (still EY [-5, -1]).
        d_pre = d_donors.filter(pl.col("event_year").is_between(-5, -1, closed="both"))
        d_pre_surv = d_pre.filter(pl.col("nsn").is_in(surviving))
        donor_summary = per_nsn_summary(d_pre_surv, PRE_WINDOW_LEN)

        # Compute donor n_post_fy (post-period FY count, for diagnostics).
        post_fy = (
            d_donors.filter(pl.col("fy") >= wfy)
            .group_by("nsn").agg(pl.col("fy").n_unique().alias("n_post_fy"))
        )
        donor_summary = donor_summary.join(post_fy, on="nsn", how="left")

        # Prepend treated's own row (n_post_fy from treated's data).
        t_post_fy = int(d.filter((pl.col("nsn") == tnsn) & (pl.col("fy") >= wfy))["fy"].n_unique())
        t_row = t_summary.with_columns(pl.lit(t_post_fy).alias("n_post_fy"))

        combined = pl.concat([t_row, donor_summary], how="vertical_relaxed").with_columns(
            pl.lit(tnsn).alias("treated_nsn"),
            (pl.col("nsn") == tnsn).alias("is_treated"),
        )
        # Attach FSG (treated row gets treated_fsg_map; donor rows get fsg_map).
        fsg_lookup = {**fsg_map, **treated_fsg_map}
        fsg_col = pl.col("nsn").replace_strict(fsg_lookup, default=None).alias("fsg")
        combined = combined.with_columns(fsg_col)

        pair_summaries.append(combined)

        coverage_log_rows.append({
            "treated_nsn": tnsn, "first_waiver_date": str(fwd), "waiver_fy": wfy,
            "n_pre_fy_treated": n_pre_treated,
            "n_active_eys_treated": len(t_active_eys),
            "active_eys_treated": ",".join(str(e) for e in sorted(t_active_eys)),
            "n_donors_in_universe": donor_set_size,
            "n_donors_after_coverage": len(surviving),
            "skipped": False, "reason": "",
        })

        if i % 5 == 0 or i == len(treated_dates):
            print(f"  [{i:02d}/{len(treated_dates)}] {tnsn} (waiver {fwd}, fy{wfy}): "
                  f"active_eys={sorted(t_active_eys)}, "
                  f"surviving_donors={len(surviving):,}")

    # ----- write -----
    if not pair_summaries:
        print("  ERROR: no surviving (treated, donor) pairs at all.")
        return

    final = pl.concat(pair_summaries, how="vertical_relaxed")
    # Order columns deterministically: identifiers, then 4 covariates,
    # then per-outcome eligibility counts.
    n_obs_cols = [f"n_obs_{o}" for o in OUTCOMES]
    final = final.select([
        "treated_nsn", "nsn", "is_treated", "fsg", "n_pre_fy", "n_post_fy",
        *COVARIATES,
        *n_obs_cols,
    ])
    final.write_parquet(OUT_PARQUET, compression="zstd")
    print()
    print(f"  wrote: {OUT_PARQUET} ({len(final):,} rows)")

    pl.DataFrame(coverage_log_rows).write_csv(COVERAGE_LOG)
    print(f"  wrote: {COVERAGE_LOG} ({len(coverage_log_rows)} treated NSNs)")

    # Summary stats.
    n_treated_present = int(final.filter(pl.col("is_treated"))["treated_nsn"].n_unique())
    n_donor_pairs = len(final.filter(~pl.col("is_treated")))
    avg_donors = n_donor_pairs / max(n_treated_present, 1)
    print()
    print(f"  treated NSNs with at least 1 donor: {n_treated_present}")
    print(f"  total (treated, donor) pairs: {n_donor_pairs:,}")
    print(f"  avg donors per treated: {avg_donors:,.0f}")

    # Coverage diagnostics for Stage 3 (Mahalanobis needs all 4 covariates
    # defined). Also report per-outcome eligibility for Stage 5.
    donor_rows = final.filter(~pl.col("is_treated"))
    for cov in COVARIATES:
        n_nonnull = int(donor_rows.filter(
            pl.col(cov).is_not_null() & ~pl.col(cov).cast(pl.Float64).is_nan()
        )["nsn"].len())
        print(f"  {cov}: {n_nonnull:,} pairs with defined value")

    all_defined = donor_rows
    for cov in COVARIATES:
        all_defined = all_defined.filter(
            pl.col(cov).is_not_null() & ~pl.col(cov).cast(pl.Float64).is_nan()
        )
    print(f"  pairs with ALL {len(COVARIATES)} matching covariates defined "
          f"(Stage-3 sample): {len(all_defined):,}")

    for o in OUTCOMES:
        n_elig = int(donor_rows.filter(pl.col(f"n_obs_{o}") >= 2)["nsn"].len())
        print(f"  pairs with >=2 pre-period obs on {o}: {n_elig:,}")


if __name__ == "__main__":
    main()
