# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
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
# # BidLink NSN Procurement History
#
# Combines all per-NSN BidLink procurement history line item CSVs into a
# single clean file.
#
# **Source**: `raw_data/bidlink/nsn/procurement_history_line_items/`
# Each file is a BidLink export for one NSN search. Filenames are generic
# (csv.csv, csv (1).csv, ...) — the NSN column within each file identifies
# the searched item.
#
# **Manual fix applied**:
# BidLink exports one row with NSN recorded as `-01-252-0367` (missing FSC).
# Corrected to `6115-01-252-0367` here.
#
# **Output**: `data/clean/bidlink_nsn.csv`

# %%
import sys
from pathlib import Path

import polars as pl


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
from paths import RAW_BIDLINK, BIDLINK_NSN_CSV, REPO_ROOT

# %%
in_dir = RAW_BIDLINK / "nsn" / "procurement_history_line_items"
out_file = BIDLINK_NSN_CSV
out_file.parent.mkdir(parents=True, exist_ok=True)

# %%
# Read and stack all CSVs
frames = [
    pl.read_csv(f, infer_schema=False)
    for f in sorted(in_dir.glob("*.csv"))
]
df = pl.concat(frames, how="diagonal_relaxed")

print(f"Loaded {len(frames)} files, {len(df)} rows")

# %%
# Manual fix: malformed NSN missing FSC prefix (BidLink export glitch)
df = df.with_columns(
    pl.col("NSN").replace({"-01-252-0367": "6115-01-252-0367"})
)

# %%
print("Unique NSNs:", df["NSN"].n_unique())
print(df["NSN"].value_counts().sort("count", descending=True))

# %%
df.write_csv(out_file)
print(f"Saved to {out_file.relative_to(REPO_ROOT)}")
