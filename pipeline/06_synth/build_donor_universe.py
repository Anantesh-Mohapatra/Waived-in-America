"""Stage 1 - Donor universe construction (synth_analysis variant).

Reads `data/panels/panel_enriched.parquet`
and applies five filters:

  01_drop_treated             drop all treated NSNs
  02_drop_waiver_reference    drop every NSN in waiver_nsn_reference.csv
  03_fy_ge_2022               keep NSNs with >=1 panel cell in fy >= 2022
  04_min_obs_per_nsn_union    keep NSNs with >=2 non-null cells in >=1 outcome
  05_waiver_text_regex        drop NSNs whose 13-digit form appears in any
                              waiver text field

Writes:
  data/donor_universe.csv         surviving donor NSNs + per-outcome eligibility
  data/filter_log.csv             row-count audit
  data/waiver_exclusion_log.csv   NSNs dropped at steps 2 and 5

TIME DIMENSION NOTE: This stage operates on `panel_enriched.parquet` (event-study
panel), which already has day-precise per-NSN event_years. The synth pipeline's
day-precise EY constraint is enforced downstream at Stage 2 via the anchored
panel.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths as P

# ---------- paths (variant-routed) ----------
_ap = argparse.ArgumentParser()
_ap.add_argument("--variant", choices=["main", "dla_only"], default="main")
_VARIANT = _ap.parse_args().variant

PANEL = P.PANEL_ENRICHED if _VARIANT == "main" else P.PANEL_DLA_ONLY
TREATED_CSV = P.TREATMENT_DATES
WAIVER_CSV = P.WAIVERS_CLEANED
WAIVER_NSN_REF_CSV = P.NSN_REFERENCE / "waiver_nsn_reference.csv"

DONOR_OUT = P.donor_universe_path(_VARIANT)
_LOGS_DIR = P.synth_logs(_VARIANT)
FILTER_LOG_OUT = _LOGS_DIR / "filter_log.csv"
WAIVER_LOG_OUT = _LOGS_DIR / "waiver_exclusion_log.csv"
_LOGS_DIR.mkdir(parents=True, exist_ok=True)
P.DATA.mkdir(parents=True, exist_ok=True)

# ---------- constants ----------
MIN_OBS_PER_NSN = 2
OUTCOMES = ("max_log_unit_price", "mean_offers", "domestic_share")

WAIVER_TEXT_COLS = (
    "Waiver_Title",
    "Summary_of_Procurement_(500_word_max)",
    "Procurement_Instrument_Identifier(s)_(PIID)_for_this_waiver_(if_applicable)",
    "Solicitation_ID",
)
WAIVER_ID_COL = "_id"


# ---------- log_filter ----------
_FILTER_LOG: list[dict] = []


def log_filter(stage: str, n_in: int, n_out: int, reason: str) -> None:
    n_dropped = n_in - n_out
    pct = (100.0 * n_dropped / n_in) if n_in else 0.0
    _FILTER_LOG.append(
        {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "stage": stage,
            "n_in": int(n_in),
            "n_out": int(n_out),
            "n_dropped": int(n_dropped),
            "pct_dropped": round(pct, 3),
            "reason": reason,
        }
    )
    print(f"  [{stage}] {n_in:,} -> {n_out:,} ({n_dropped:,} dropped, {pct:.2f}%) :: {reason}")


# ---------- helpers ----------
def to_no_dash(nsn_dashed: str) -> str:
    return nsn_dashed.replace("-", "")


def extract_waiver_nsn_mentions(waiver_df: pl.DataFrame) -> dict[str, list[str]]:
    """Return {nsn_no_dash: [waiver_id, ...]} extracted from waiver text columns.

    Two patterns: dashed `\\b\\d{4}-\\d{2}-\\d{3}-\\d{4}\\b` and no-dash `\\b\\d{13}\\b`.
    Both canonicalize to no-dash for set comparison downstream.
    """
    re_dashed = re.compile(r"\b(\d{4}-\d{2}-\d{3}-\d{4})\b")
    re_nodash = re.compile(r"\b(\d{13})\b")
    mentions: dict[str, list[str]] = {}

    cols_present = [c for c in WAIVER_TEXT_COLS if c in waiver_df.columns]
    if WAIVER_ID_COL not in waiver_df.columns:
        raise ValueError(f"Waiver CSV missing identifier column {WAIVER_ID_COL!r}.")

    for row in waiver_df.select([WAIVER_ID_COL, *cols_present]).iter_rows(named=True):
        wid = str(row[WAIVER_ID_COL])
        blob_parts = []
        for col in cols_present:
            v = row[col]
            if v is not None:
                blob_parts.append(str(v))
        if not blob_parts:
            continue
        blob = "\n".join(blob_parts)
        for m in re_dashed.finditer(blob):
            mentions.setdefault(to_no_dash(m.group(1)), []).append(wid)
        for m in re_nodash.finditer(blob):
            mentions.setdefault(m.group(1), []).append(wid)

    return mentions


# ---------- main ----------
def main() -> None:
    print("Stage 1 - donor universe construction (synth_analysis)")
    print(f"  panel: {PANEL}")
    print(f"  treated: {TREATED_CSV}")
    print(f"  waivers: {WAIVER_CSV}")
    print()

    panel = pl.read_parquet(PANEL)
    print(f"  panel rows: {len(panel):,}")
    print(f"  panel unique NSNs: {panel['nsn'].n_unique():,}")
    print()

    n_all_nsns = panel["nsn"].n_unique()
    log_filter("00_start", n_all_nsns, n_all_nsns, "all NSNs in panel_enriched.parquet")

    # ----- step 1: drop treated NSNs -----
    treated_df = pl.read_csv(TREATED_CSV)
    treated_nsns = sorted({to_no_dash(n) for n in treated_df["nsn"].to_list()})
    print(f"  treated NSN count from {TREATED_CSV.name}: {len(treated_nsns)}")
    panel = panel.filter(~pl.col("nsn").is_in(treated_nsns))
    n_after_treated = panel["nsn"].n_unique()
    log_filter("01_drop_treated", n_all_nsns, n_after_treated,
               f"drop the {len(treated_nsns)} treated NSNs")

    # ----- step 2: explicit waiver-NSN drop -----
    waiver_nsn_ref = pl.read_csv(WAIVER_NSN_REF_CSV, infer_schema_length=0)
    waiver_ref_nodash = sorted(set(waiver_nsn_ref["nsn"].to_list()))
    waiver_ref_not_treated = sorted(set(waiver_ref_nodash) - set(treated_nsns))
    panel_nsns_before = set(panel["nsn"].unique().to_list())
    waiver_ref_in_panel = sorted(panel_nsns_before & set(waiver_ref_nodash))
    panel = panel.filter(~pl.col("nsn").is_in(waiver_ref_nodash))
    n_after_waiver_ref = panel["nsn"].n_unique()
    log_filter("02_drop_waiver_reference", n_after_treated, n_after_waiver_ref,
               f"drop {len(waiver_ref_nodash)} NSNs in waiver_nsn_reference.csv "
               f"({len(waiver_ref_not_treated)} not already in treated set; "
               f"{len(waiver_ref_in_panel)} were in panel)")
    waiver_ref_dropped = waiver_ref_in_panel

    # ----- step 3: control filter (>=1 cell in fy >= 2022) -----
    nsns_with_post_2022 = panel.filter(pl.col("fy") >= 2022)["nsn"].unique().to_list()
    panel = panel.filter(pl.col("nsn").is_in(nsns_with_post_2022))
    n_after_post2022 = panel["nsn"].n_unique()
    log_filter("03_fy_ge_2022", n_after_waiver_ref, n_after_post2022,
               "NSN must have >=1 cell with fy >= 2022")

    # ----- step 4: per-outcome eligibility (MIN_OBS_PER_NSN >= 2) -----
    elig_frames = []
    for outcome in OUTCOMES:
        non_null = panel.filter(
            pl.col(outcome).is_not_null() & ~pl.col(outcome).is_nan()
        )
        counts = non_null.group_by("nsn").len().rename({"len": f"n_{outcome}"})
        eligible = counts.filter(pl.col(f"n_{outcome}") >= MIN_OBS_PER_NSN).select(
            pl.col("nsn"),
            pl.lit(1, dtype=pl.Int8).alias(f"eligible_{outcome}"),
        )
        elig_frames.append(eligible)
        n_elig = len(eligible)
        print(f"  per-outcome eligibility ({outcome}): {n_elig:,} NSNs satisfy >=2 non-null cells")

    elig_any = elig_frames[0]
    for e in elig_frames[1:]:
        elig_any = elig_any.join(e, on="nsn", how="full", coalesce=True)
    elig_any = elig_any.with_columns([
        pl.col(f"eligible_{o}").fill_null(0).cast(pl.Int8) for o in OUTCOMES
    ])

    elig_any = elig_any.with_columns(
        (pl.sum_horizontal([pl.col(f"eligible_{o}") for o in OUTCOMES]) >= 1).alias("any_eligible")
    )
    keep_nsns = elig_any.filter(pl.col("any_eligible"))["nsn"].to_list()

    panel = panel.filter(pl.col("nsn").is_in(keep_nsns))
    n_after_minobs = panel["nsn"].n_unique()
    log_filter("04_min_obs_per_nsn_union", n_after_post2022, n_after_minobs,
               f"NSN must have >={MIN_OBS_PER_NSN} non-null cells in >=1 outcome (union)")

    # ----- step 5: waiver-text-regex contamination drop -----
    waivers = pl.read_csv(WAIVER_CSV, infer_schema_length=10000)
    print(f"  waivers loaded: {len(waivers):,} rows")
    mentions = extract_waiver_nsn_mentions(waivers)
    print(f"  extracted {len(mentions):,} unique NSN-shaped tokens from waiver text")

    surviving_nodash = {to_no_dash(n): n for n in panel["nsn"].unique().to_list()}
    contaminated = {dashed: mentions[nodash]
                    for nodash, dashed in surviving_nodash.items()
                    if nodash in mentions}
    print(f"  text-regex-contaminated donors found: {len(contaminated):,}")

    panel = panel.filter(~pl.col("nsn").is_in(list(contaminated.keys())))
    n_after_waiver = panel["nsn"].n_unique()
    log_filter("05_waiver_text_regex", n_after_minobs, n_after_waiver,
               "drop NSNs whose dashed or 13-digit form appears in any waiver text field")

    # ----- combined waiver exclusion log -----
    waiver_ref_lookup = {
        row["nsn"]: row["nsn_formatted"]
        for row in waiver_nsn_ref.iter_rows(named=True)
    }
    log_rows = []
    for nodash in waiver_ref_dropped:
        log_rows.append({
            "nsn_no_dash": nodash,
            "nsn": waiver_ref_lookup.get(nodash, nodash),
            "source": "waiver_nsn_reference",
            "n_waiver_hits": 1,
            "waiver_ids": "",
        })
    for dashed, wids in contaminated.items():
        log_rows.append({
            "nsn_no_dash": to_no_dash(dashed),
            "nsn": dashed,
            "source": "waiver_text_regex",
            "n_waiver_hits": len(wids),
            "waiver_ids": ";".join(sorted(set(wids))),
        })
    if log_rows:
        pl.DataFrame(log_rows).write_csv(WAIVER_LOG_OUT)
    else:
        pl.DataFrame(
            schema={
                "nsn_no_dash": pl.Utf8, "nsn": pl.Utf8, "source": pl.Utf8,
                "n_waiver_hits": pl.Int64, "waiver_ids": pl.Utf8,
            }
        ).write_csv(WAIVER_LOG_OUT)

    # ----- assemble donor_universe.csv -----
    nsn_meta = (
        panel
        .group_by("nsn")
        .agg([
            pl.col("fsc").drop_nulls().mode().first().alias("fsc"),
            pl.len().alias("n_cells"),
            pl.col("fy").n_unique().alias("n_fy"),
            pl.col("fy").min().alias("first_fy"),
            pl.col("fy").max().alias("last_fy"),
        ])
        .with_columns(
            pl.col("fsc").str.slice(0, 2).alias("fsg"),
        )
    )

    donor_universe = nsn_meta.join(elig_any.drop("any_eligible"), on="nsn", how="left").select(
        "nsn", "fsg", "fsc", "n_cells", "n_fy", "first_fy", "last_fy",
        *[f"eligible_{o}" for o in OUTCOMES],
    )

    if len(donor_universe) != n_after_waiver:
        print(f"  WARNING: donor_universe rows ({len(donor_universe):,}) != "
              f"n_after_waiver ({n_after_waiver:,})")

    donor_universe.write_csv(DONOR_OUT)
    print()
    print(f"  wrote: {DONOR_OUT} ({len(donor_universe):,} rows)")
    print(f"  wrote: {WAIVER_LOG_OUT} "
          f"({len(waiver_ref_dropped):,} waiver_nsn_reference + "
          f"{len(contaminated):,} waiver_text_regex)")

    for o in OUTCOMES:
        n = int(donor_universe[f"eligible_{o}"].sum())
        print(f"  donors eligible for {o}: {n:,}")

    intersection = int((
        (donor_universe["eligible_max_log_unit_price"] == 1)
        & (donor_universe["eligible_mean_offers"] == 1)
        & (donor_universe["eligible_domestic_share"] == 1)
    ).sum())
    print(f"  donors eligible for ALL three outcomes: {intersection:,}")

    pl.DataFrame(_FILTER_LOG).write_csv(FILTER_LOG_OUT)
    print(f"  wrote: {FILTER_LOG_OUT} ({len(_FILTER_LOG)} entries)")


if __name__ == "__main__":
    main()
