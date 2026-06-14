"""
USAspending API backfill for FPDS bulk-archive coverage gaps.

Why this script exists
----------------------
The USAspending bulk-archive zips published periodically at usaspending.gov
have a substantial DoD reporting gap for the most recent few months at the
time of each snapshot. Concretely, the FY2026 zip published 2026-04-06
contained only 482 DLA contract transactions for action_date >= 2026-01-01,
versus ~169K (post-C-filter) actually present in the live API. Without a
backfill, dla_contract_enriched_*.parquet shows 0.3% match for January 2026
DLA awards even though the underlying contracts exist in FPDS.

This script pulls the missing DLA rows via the bulk_download API and merges
them into data/clean/procurement_data.parquet. Downstream scripts need no
changes -- they just see a more complete parquet.

Pipeline order
--------------
1. pipeline/01_clean/parquement.py          (build parquet from bulk zips)
2. pipeline/01_clean/usaspending_backfill.py (THIS SCRIPT, fill the gap)
3. pipeline/01_clean/dla_foia.py             (join DLA FOIA -> parquet)

What it does
------------
- Submits one /api/v2/bulk_download/awards/ job filtered to:
    agencies     = Defense Logistics Agency (subtier under DoD)
    award types  = A, B, C, D   (matches bulk archive coverage; no IDVs)
    date_range   = [start_date, end_date]  CLI flags
    date_type    = action_date
- Polls until done, downloads the CSV zip into
    raw_data/procurement/api_backfill/dla_backfill_{start}_{end}_{today}.zip
- Reads the CSV with the same columns_to_keep + place_of_manufacture_code
  filter as parquement.py, then merges into procurement_data.parquet:
    * removes any existing rows whose contract_transaction_unique_key
      appears in the new CSV (replaces stale snapshot data)
    * appends the new rows
  The merge is idempotent -- re-running with the same date range produces
  the same result.

What it does NOT do
-------------------
- Backfill IDV records (bulk archive is contracts-only -- adding IDVs
  would change the meaning of "every parquet row is a money transaction"
  that downstream code relies on).
- Backfill non-DLA agencies. The DLA contract-history join is the only
  downstream consumer affected by the gap.
- Recover Feb-Mar 2026 DLA rows. Those genuinely don't exist in FPDS yet
  (verified via usaspending.gov live search). They will start appearing
  in future API pulls as DLA reports them; re-running this script will
  pick them up.

Known data limit
----------------
Some DLA contract-history PIIDs are never reported to FPDS at all (small
purchases below FPDS threshold, etc). Even with this backfill,
dla_contract_enriched_*.parquet match rate plateaus below 100% for the
gap period.

Caveats
-------
- columns_to_keep below MUST stay in sync with the same list in
  pipeline/01_clean/parquement.py. If you add/remove columns
  there, mirror the change here.
- Cached zip downloads are kept (named with the download date) so re-runs
  on the same day reuse the file instead of resubmitting the job.
"""

import argparse
import gc
import io
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from paths import PROCUREMENT_PARQUET, RAW_API_BACKFILL

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PARQUET_PATH = PROCUREMENT_PARQUET
CACHE_DIR = RAW_API_BACKFILL

API_BASE = "https://api.usaspending.gov"
SUBMIT_URL = f"{API_BASE}/api/v2/bulk_download/awards/"
STATUS_URL = f"{API_BASE}/api/v2/download/status"

POLL_INTERVAL_SEC = 15
POLL_TIMEOUT_SEC = 60 * 60  # 1 hour

# MUST stay in sync with columns_to_keep in
# pipeline/01_clean/parquement.py
COLUMNS_TO_KEEP = [
    "contract_transaction_unique_key",
    "contract_award_unique_key",
    "award_id_piid",
    "modification_number",
    "parent_award_agency_id",
    "parent_award_agency_name",
    "parent_award_id_piid",
    "parent_award_modification_number",
    "federal_action_obligation",
    "total_dollars_obligated",
    "current_total_value_of_award",
    "base_and_all_options_value",
    "action_date",
    "action_date_fiscal_year",
    "period_of_performance_start_date",
    "period_of_performance_current_end_date",
    "solicitation_date",
    "awarding_agency_code",
    "awarding_agency_name",
    "awarding_sub_agency_code",
    "awarding_sub_agency_name",
    "awarding_office_code",
    "awarding_office_name",
    "funding_agency_code",
    "funding_agency_name",
    "funding_sub_agency_code",
    "funding_sub_agency_name",
    "funding_office_code",
    "funding_office_name",
    "foreign_funding",
    "foreign_funding_description",
    "sam_exception",
    "sam_exception_description",
    "recipient_uei",
    "recipient_duns",
    "recipient_name",
    "recipient_name_raw",
    "recipient_doing_business_as_name",
    "cage_code",
    "recipient_parent_uei",
    "recipient_parent_duns",
    "recipient_parent_name",
    "recipient_parent_name_raw",
    "recipient_country_code",
    "recipient_country_name",
    "recipient_address_line_1",
    "recipient_address_line_2",
    "recipient_city_name",
    "prime_award_transaction_recipient_county_fips_code",
    "recipient_county_name",
    "prime_award_transaction_recipient_state_fips_code",
    "recipient_state_code",
    "recipient_state_name",
    "recipient_zip_4_code",
    "prime_award_transaction_recipient_cd_original",
    "prime_award_transaction_recipient_cd_current",
    "primary_place_of_performance_country_code",
    "primary_place_of_performance_country_name",
    "primary_place_of_performance_city_name",
    "prime_award_transaction_place_of_performance_county_fips_code",
    "primary_place_of_performance_county_name",
    "prime_award_transaction_place_of_performance_state_fips_code",
    "primary_place_of_performance_state_code",
    "primary_place_of_performance_state_name",
    "primary_place_of_performance_zip_4",
    "prime_award_transaction_place_of_performance_cd_original",
    "prime_award_transaction_place_of_performance_cd_current",
    "award_or_idv_flag",
    "award_type_code",
    "award_type",
    "idv_type_code",
    "idv_type",
    "multiple_or_single_award_idv_code",
    "multiple_or_single_award_idv",
    "type_of_idc_code",
    "type_of_idc",
    "type_of_contract_pricing_code",
    "type_of_contract_pricing",
    "transaction_description",
    "prime_award_base_transaction_description",
    "product_or_service_code",
    "product_or_service_code_description",
    "naics_code",
    "naics_description",
    "domestic_or_foreign_entity_code",
    "domestic_or_foreign_entity",
    "country_of_product_or_service_origin_code",
    "country_of_product_or_service_origin",
    "place_of_manufacture_code",
    "place_of_manufacture",
    "extent_competed_code",
    "extent_competed",
    "solicitation_procedures_code",
    "solicitation_procedures",
    "type_of_set_aside_code",
    "type_of_set_aside",
    "evaluated_preference_code",
    "evaluated_preference",
    "fair_opportunity_limited_sources_code",
    "fair_opportunity_limited_sources",
    "other_than_full_and_open_competition_code",
    "other_than_full_and_open_competition",
    "number_of_offers_received",
    "commercial_item_acquisition_procedures_code",
    "commercial_item_acquisition_procedures",
    "materials_supplies_articles_equipment_code",
    "materials_supplies_articles_equipment",
    "labor_standards_code",
    "labor_standards",
    "construction_wage_rate_requirements_code",
    "construction_wage_rate_requirements",
    "purchase_card_as_payment_method_code",
    "purchase_card_as_payment_method",
    "contracting_officers_determination_of_business_size",
    "contracting_officers_determination_of_business_size_code",
]


# ---------------------------------------------------------------------------
# API job submission + polling
# ---------------------------------------------------------------------------

def _post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def submit_dla_job(start_date: str, end_date: str) -> dict:
    payload = {
        "filters": {
            "prime_award_types": ["A", "B", "C", "D"],
            "date_type": "action_date",
            "date_range": {"start_date": start_date, "end_date": end_date},
            "agencies": [
                {
                    "type": "awarding",
                    "tier": "subtier",
                    "toptier_name": "Department of Defense",
                    "name": "Defense Logistics Agency",
                }
            ],
        },
        "columns": [],
        "file_format": "csv",
    }
    print(f"Submitting bulk_download job: DLA {start_date} -> {end_date}")
    resp = _post_json(SUBMIT_URL, payload)
    print(f"  job file_name: {resp['file_name']}")
    print(f"  status_url:    {resp['status_url']}")
    print(f"  file_url:      {resp['file_url']}")
    return resp


def poll_until_done(status_url: str) -> dict:
    started = time.time()
    last_status = None
    while True:
        elapsed = time.time() - started
        if elapsed > POLL_TIMEOUT_SEC:
            raise TimeoutError(f"Job did not finish within {POLL_TIMEOUT_SEC}s")
        st = _get_json(status_url)
        status = st.get("status")
        if status != last_status:
            print(f"  [t={elapsed:6.0f}s] status={status}")
            last_status = status
        if status == "finished":
            print(
                f"  rows={st.get('total_rows')} "
                f"size_kb={st.get('total_size')} "
                f"server_elapsed={st.get('seconds_elapsed')}s"
            )
            return st
        if status == "failed":
            raise RuntimeError(f"Bulk download job failed: {st}")
        time.sleep(POLL_INTERVAL_SEC)


def download_zip(file_url: str, dest_path: Path) -> Path:
    if dest_path.exists():
        print(f"Cache hit, reusing {dest_path.name} ({dest_path.stat().st_size/1e6:.1f} MB)")
        return dest_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading -> {dest_path}")
    with urllib.request.urlopen(file_url, timeout=300) as r, open(dest_path, "wb") as f:
        # Stream in 4 MB chunks
        while True:
            chunk = r.read(4 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    print(f"  downloaded {dest_path.stat().st_size/1e6:.1f} MB")
    return dest_path


# ---------------------------------------------------------------------------
# Merge into procurement_data.parquet
# ---------------------------------------------------------------------------

def read_backfill_csv(zip_path: Path) -> pl.DataFrame:
    """Read the bulk_download CSV(s) from the zip, applying the same
    columns_to_keep selection and place_of_manufacture_code != 'C' filter
    as parquement.py."""
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"No CSVs found inside {zip_path}")
        frames = []
        for name in csv_names:
            print(f"  reading {name}")
            df = pl.read_csv(
                zf.open(name),
                columns=COLUMNS_TO_KEEP,
                infer_schema_length=0,
                ignore_errors=True,
                low_memory=True,
            )
            frames.append(df)
    new = pl.concat(frames, how="vertical")
    pre = new.height
    new = new.filter(pl.col("place_of_manufacture_code") != "C")
    print(f"  rows in zip: {pre:,}, after C-filter: {new.height:,}")
    return new


def merge_into_parquet(new: pl.DataFrame, parquet_path: Path) -> dict:
    """Replace any existing rows whose contract_transaction_unique_key is in
    `new`, then append `new` rows. Idempotent across re-runs.

    Memory strategy: stream the existing parquet through pyarrow row-group
    batches. Hold only one batch in memory at a time, plus the (small) set
    of new ctuks to drop.
    """
    if new.is_empty():
        print("No new rows after filtering; nothing to merge.")
        return {"appended": 0, "replaced": 0, "kept": 0}

    new_keys = set(new["contract_transaction_unique_key"].drop_nulls().to_list())
    print(f"Merge keys: {len(new_keys):,} unique ctuks in new data")

    # Align column order with existing parquet
    existing_cols = pl.scan_parquet(parquet_path).collect_schema().names()
    if set(existing_cols) != set(new.columns):
        missing_in_new = set(existing_cols) - set(new.columns)
        extra_in_new = set(new.columns) - set(existing_cols)
        raise RuntimeError(
            f"Schema mismatch.\n  missing in new CSV: {sorted(missing_in_new)}\n"
            f"  extra in new CSV: {sorted(extra_in_new)}"
        )
    new = new.select(existing_cols)

    tmp_path = parquet_path.with_suffix(".tmp.parquet")
    if tmp_path.exists():
        tmp_path.unlink()

    writer = None
    kept = 0
    replaced = 0
    batch_size = 500_000
    n_batches = 0

    # Open the source parquet inside a try/finally so the file handle is
    # always released before we try to unlink/rename on Windows (where an
    # open ParquetFile holds an OS-level lock that blocks file ops).
    pq_in = pq.ParquetFile(str(parquet_path))
    try:
        for batch in pq_in.iter_batches(batch_size=batch_size):
            df = pl.from_arrow(batch)
            before = df.height
            df = df.filter(~pl.col("contract_transaction_unique_key").is_in(list(new_keys)))
            after = df.height
            kept += after
            replaced += before - after

            tbl = df.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(str(tmp_path), tbl.schema, compression="zstd")
            writer.write_table(tbl)
            n_batches += 1
            del df, tbl
            gc.collect()
            print(f"  batch {n_batches}: {after:,} kept, {before-after:,} replaced (running kept={kept:,})")

        # Append the new rows. Cast to the writer's schema so types align
        # with the existing parquet (both are all-String from infer_schema_length=0).
        new_tbl = new.to_arrow().cast(writer.schema)
        writer.write_table(new_tbl)
    finally:
        if writer is not None:
            writer.close()
        pq_in.close()
        del pq_in
        gc.collect()

    parquet_path.unlink()
    tmp_path.rename(parquet_path)

    return {"appended": new.height, "replaced": replaced, "kept": kept}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start-date", default="2026-01-01",
                   help="action_date >= this (YYYY-MM-DD). Default 2026-01-01 (start of known FPDS gap).")
    p.add_argument("--end-date", default=date.today().isoformat(),
                   help="action_date <= this (YYYY-MM-DD). Default today.")
    p.add_argument("--reuse-cached-zip",
                   help="Skip API submission, use this local zip path instead.")
    args = p.parse_args()

    if not PARQUET_PATH.exists():
        sys.exit(f"ERROR: {PARQUET_PATH} not found. Run parquement.py first.")

    today = date.today().isoformat()
    cache_name = f"dla_backfill_{args.start_date}_{args.end_date}_{today}.zip"
    zip_path = CACHE_DIR / cache_name

    if args.reuse_cached_zip:
        zip_path = Path(args.reuse_cached_zip)
        if not zip_path.exists():
            sys.exit(f"ERROR: --reuse-cached-zip {zip_path} not found")
        print(f"Reusing cached zip: {zip_path}")
    elif zip_path.exists():
        print(f"Cache hit for today: {zip_path.name}")
    else:
        job = submit_dla_job(args.start_date, args.end_date)
        st = poll_until_done(job["status_url"])
        if st.get("total_rows", 0) == 0:
            sys.exit("Job returned 0 rows; check filters before merging.")
        download_zip(st["file_url"], zip_path)

    print()
    print(f"Reading CSV from {zip_path.name}")
    new = read_backfill_csv(zip_path)

    print()
    print(f"Merging into {PARQUET_PATH.name}")
    stats = merge_into_parquet(new, PARQUET_PATH)
    print()
    print("Done.")
    print(f"  rows kept from existing parquet: {stats['kept']:,}")
    print(f"  rows replaced (stale snapshot):  {stats['replaced']:,}")
    print(f"  rows appended (from API):        {stats['appended']:,}")
    print(f"  parquet size: {PARQUET_PATH.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
