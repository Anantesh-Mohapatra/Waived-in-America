# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: kernelspec,jupytext
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: waived-in-america (3.13)
#     language: python
#     name: python3
# ---

# %% [markdown]
# This notebook cleans and merges the FPDS procurement data from usaspending.gov.

# %%
import gc
import sys
import zipfile
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq


def _repo_root() -> Path:
    try:
        start = Path(__file__).resolve().parent  # script execution
    except NameError:
        start = Path.cwd()  # notebook kernel
    p = start
    while not (p / "pyproject.toml").exists():
        if p.parent == p:
            raise FileNotFoundError(f"repo root (pyproject.toml) not found above {start}")
        p = p.parent
    return p


sys.path.insert(0, str(_repo_root() / "pipeline" / "lib"))
from paths import RAW_PROCUREMENT, PROCUREMENT_PARQUET

# %%
# For now, these are the columns I want to keep from the full dataset
columns_to_keep = [
    # --- IDs / keys ---
    'contract_transaction_unique_key',
    'contract_award_unique_key',
    'award_id_piid',
    'modification_number',
    'parent_award_agency_id',
    'parent_award_agency_name',
    'parent_award_id_piid',
    'parent_award_modification_number',

    # --- Money (outcomes) ---
    'federal_action_obligation',
    'total_dollars_obligated',
    'current_total_value_of_award',
    'base_and_all_options_value',

    # --- Timing ---
    'action_date',
    'action_date_fiscal_year',
    'period_of_performance_start_date',
    'period_of_performance_current_end_date',
    'solicitation_date',

    # --- Agencies / offices ---
    'awarding_agency_code',
    'awarding_agency_name',
    'awarding_sub_agency_code',
    'awarding_sub_agency_name',
    'awarding_office_code',
    'awarding_office_name',
    'funding_agency_code',
    'funding_agency_name',
    'funding_sub_agency_code',
    'funding_sub_agency_name',
    'funding_office_code',
    'funding_office_name',

    # --- Funding peculiarities / foreign funding ---
    'foreign_funding',
    'foreign_funding_description',

    # --- SAM registration exception (could matter for edge cases) ---
    'sam_exception',
    'sam_exception_description',

    # --- Recipient identity ---
    'recipient_uei',
    'recipient_duns',
    'recipient_name',
    'recipient_name_raw',
    'recipient_doing_business_as_name',
    'cage_code',
    'recipient_parent_uei',
    'recipient_parent_duns',
    'recipient_parent_name',
    'recipient_parent_name_raw',

    # --- Recipient location ---
    'recipient_country_code',
    'recipient_country_name',
    'recipient_address_line_1',
    'recipient_address_line_2',
    'recipient_city_name',
    'prime_award_transaction_recipient_county_fips_code',
    'recipient_county_name',
    'prime_award_transaction_recipient_state_fips_code',
    'recipient_state_code',
    'recipient_state_name',
    'recipient_zip_4_code',
    'prime_award_transaction_recipient_cd_original',
    'prime_award_transaction_recipient_cd_current',

    # --- Primary place of performance ---
    'primary_place_of_performance_country_code',
    'primary_place_of_performance_country_name',
    'primary_place_of_performance_city_name',
    'prime_award_transaction_place_of_performance_county_fips_code',
    'primary_place_of_performance_county_name',
    'prime_award_transaction_place_of_performance_state_fips_code',
    'primary_place_of_performance_state_code',
    'primary_place_of_performance_state_name',
    'primary_place_of_performance_zip_4',
    'prime_award_transaction_place_of_performance_cd_original',
    'prime_award_transaction_place_of_performance_cd_current',

    # --- Award / IDV structure ---
    'award_or_idv_flag',
    'award_type_code',
    'award_type',
    'idv_type_code',
    'idv_type',
    'multiple_or_single_award_idv_code',
    'multiple_or_single_award_idv',
    'type_of_idc_code',
    'type_of_idc',

    # --- Contract pricing ---
    'type_of_contract_pricing_code',
    'type_of_contract_pricing',

    # --- Descriptions ---
    'transaction_description',
    'prime_award_base_transaction_description',

    # --- Product / industry classification ---
    'product_or_service_code',
    'product_or_service_code_description',
    'naics_code',
    'naics_description',

    # --- Core domestic content / origin variables ---
    'domestic_or_foreign_entity_code',
    'domestic_or_foreign_entity',
    'country_of_product_or_service_origin_code',
    'country_of_product_or_service_origin',
    'place_of_manufacture_code',
    'place_of_manufacture',

    # --- Competition / procedures (super important controls) ---
    'extent_competed_code',
    'extent_competed',
    'solicitation_procedures_code',
    'solicitation_procedures',
    'type_of_set_aside_code',
    'type_of_set_aside',
    'evaluated_preference_code',
    'evaluated_preference',
    'fair_opportunity_limited_sources_code',
    'fair_opportunity_limited_sources',
    'other_than_full_and_open_competition_code',
    'other_than_full_and_open_competition',
    'number_of_offers_received',
    'commercial_item_acquisition_procedures_code',
    'commercial_item_acquisition_procedures',

    # --- A few high-level legal/process flags worth having as dummies ---
    'materials_supplies_articles_equipment_code',
    'materials_supplies_articles_equipment',
    'labor_standards_code',
    'labor_standards',
    'construction_wage_rate_requirements_code',
    'construction_wage_rate_requirements',

    # --- Purchase card / micro-purchase filter ---
    'purchase_card_as_payment_method_code',
    'purchase_card_as_payment_method',

    # --- Business size (compact way to capture small vs large) ---
    'contracting_officers_determination_of_business_size',
    'contracting_officers_determination_of_business_size_code',
]

# %%
# Where to read and write
zip_paths = sorted(RAW_PROCUREMENT.glob("FY*_All_Contracts_Full_*.zip"))
output = PROCUREMENT_PARQUET
output.parent.mkdir(parents=True, exist_ok=True)
if output.exists():
    output.unlink()  # remove old run so we can append cleanly

# %%
writer = None

for zp in zip_paths:
    print(f"\nProcessing zip: {zp}")

    with zipfile.ZipFile(zp) as zf:
        csv_files = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        total_csv = len(csv_files)

        for csv_idx, name in enumerate(csv_files, start=1):
            print(f"  ({csv_idx}/{total_csv}) Reading CSV in zip: {name}")

            # Read CSV normally
            df = pl.read_csv(
                zf.open(name),
                columns=columns_to_keep,
                infer_schema_length=0,
                ignore_errors=True,
                low_memory=True,
            )

            df = df.filter(pl.col("place_of_manufacture_code") != "C") # FILTER DECISION: drop rows with "C" in place_of_manufacture_code, which is "Not a manufactured end product"
            if df.is_empty():
                continue

            table = df.to_arrow()

            # Create writer on first non-empty table
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression="zstd")

            # Append next table
            writer.write_table(table)

            print(f"      wrote -> {df.height} rows")

            del df, table
            gc.collect()

# Close the writer (very important!)
if writer is not None:
    writer.close()

print("\nDone ->", output.resolve())
