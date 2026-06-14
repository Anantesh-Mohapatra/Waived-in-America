# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:percent,ipynb
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
# # Procurement in Context Descriptives
#
# Reproducible numbers for the thesis's Data > Procurement in Context
# subsection, including DLA-specific shares cited inline (OQ3).
#
# **Outputs (in `pipeline/08_artifacts/output/`):**
# - `procurement_context_stats.txt` — federal-wide and DLA-specific shares
#   for transactions and dollar obligations, FY2017 through January 2026,
#   plus the foreign-share and nonavailability-share breakouts.
#
# **Run with:** `uv run python pipeline/08_artifacts/procurement_context_descriptives.py`
#
# **Inputs:**
# - `data/clean/procurement_data.parquet` (cleaned FPDS extract; rows with
#   `place_of_manufacture_code == "C"` already excluded upstream).
#
# **Place-of-manufacture coding (see `data_defs/place_of_manufacture_codes.json`):**
# - `D` = manufactured in US (domestic).
# - `J` = manufactured outside US, domestic non-availability (the nonavailability
#   code referenced throughout the thesis).
# - `A B E F G H I K L` = the remaining foreign-mfg codes (foreign share).
# - `C` = not a manufactured end product (already filtered out upstream).

# %%
from datetime import date, datetime
from pathlib import Path

import polars as pl

# Windows cp1252 terminal: ASCII tables, no Unicode in prints.
pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
pl.Config.set_tbl_rows(20)
pl.Config.set_tbl_cols(8)

# %%
# --- Resolve project root (script or notebook kernel) ---
import sys

try:
    _start = Path(__file__).resolve().parent
except NameError:
    _start = Path.cwd()
ROOT = _start
while not (ROOT / "pyproject.toml").exists():
    if ROOT.parent == ROOT:
        raise FileNotFoundError(f"repo root (pyproject.toml) not found above {_start}")
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT / "pipeline" / "lib"))
import paths as P

PROC_PATH = P.PROCUREMENT_PARQUET
OUT = P.DESCRIPTIVES_STATS
OUT.mkdir(parents=True, exist_ok=True)

# Cutoff matches the A5 / A6 / matched-controls / synth analysis window.
CUTOFF = date(2026, 1, 31)
DLA_NAME = "Defense Logistics Agency"

DOMESTIC_CODES = ["D"]
NONAVAIL_CODES = ["J"]

# %% [markdown]
# ## Apply the FY2017-through-Jan-2026 cutoff used elsewhere in the thesis

# %%
proc_lf = (
    pl.scan_parquet(PROC_PATH)
    .with_columns(
        pl.col("action_date").str.to_date("%Y-%m-%d").alias("action_date"),
        pl.col("federal_action_obligation").cast(pl.Float64, strict=False).alias("federal_action_obligation"),
    )
    .filter(pl.col("action_date") <= CUTOFF)
)

# %% [markdown]
# ## Federal-wide and DLA totals, by transaction and dollar obligation


# %%
def summarize(filter_expr: pl.Expr | None, label: str) -> dict:
    """Return n_transactions, total_dollars, mfg_obs counts, domestic / foreign / J splits."""
    q = proc_lf
    if filter_expr is not None:
        q = q.filter(filter_expr)
    row = q.select(
        pl.len().alias("n_transactions"),
        pl.col("federal_action_obligation").sum().alias("total_dollars"),
        # "any manufacture code recorded" denominator for the foreign-share calc.
        # Upstream already drops C; we further drop nulls (place_of_manufacture_code missing).
        pl.col("place_of_manufacture_code").is_not_null().sum().alias("n_mfg_obs"),
        pl.col("place_of_manufacture_code").is_in(DOMESTIC_CODES).sum().alias("n_domestic"),
        pl.col("place_of_manufacture_code").is_in(NONAVAIL_CODES).sum().alias("n_J"),
        # Dollar-weighted splits for the "share of dollars" sentences.
        pl.when(pl.col("place_of_manufacture_code").is_not_null())
          .then(pl.col("federal_action_obligation")).otherwise(0).sum().alias("dollars_mfg_obs"),
        pl.when(pl.col("place_of_manufacture_code").is_in(DOMESTIC_CODES))
          .then(pl.col("federal_action_obligation")).otherwise(0).sum().alias("dollars_domestic"),
        pl.when(pl.col("place_of_manufacture_code").is_in(NONAVAIL_CODES))
          .then(pl.col("federal_action_obligation")).otherwise(0).sum().alias("dollars_J"),
    ).collect(engine="streaming").row(0, named=True)
    row["label"] = label
    return row


fed = summarize(None, "Federal-wide")
dla = summarize(pl.col("awarding_sub_agency_name") == DLA_NAME, "DLA only")

# %% [markdown]
# ## Compose paragraph stats


# %%
def pct(num: float, den: float) -> str:
    return f"{100 * num / den:.2f}%" if den else "n/a"


def fmt_dollars(d: float) -> str:
    if d is None:
        return "n/a"
    if abs(d) >= 1e12:
        return f"${d / 1e12:.2f}T"
    if abs(d) >= 1e9:
        return f"${d / 1e9:.2f}B"
    if abs(d) >= 1e6:
        return f"${d / 1e6:.2f}M"
    return f"${d:,.0f}"


now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
lines = [
    f"=== Procurement in Context (OQ3) paragraph stats - generated {now} ===",
    f"Cutoff: action_date <= {CUTOFF.isoformat()}",
    f"Source: {PROC_PATH.relative_to(ROOT).as_posix()}",
    "",
    "# Federal-wide totals (FY2017 - Jan 2026, post-C-filter)",
    f"Transactions:            {fed['n_transactions']:>15,}",
    f"Dollar obligations:      {fmt_dollars(fed['total_dollars']):>15}",
    f"  Domestic (D):          {fed['n_domestic']:>15,} ({pct(fed['n_domestic'], fed['n_mfg_obs'])} of mfg-coded)",
    f"  Nonavailability (J):   {fed['n_J']:>15,} ({pct(fed['n_J'], fed['n_mfg_obs'])} of mfg-coded)",
    f"  Foreign (non-D):       {fed['n_mfg_obs'] - fed['n_domestic']:>15,} ({pct(fed['n_mfg_obs'] - fed['n_domestic'], fed['n_mfg_obs'])} of mfg-coded)",
    f"  Domestic $:            {fmt_dollars(fed['dollars_domestic']):>15} ({pct(fed['dollars_domestic'], fed['dollars_mfg_obs'])} of mfg-coded $)",
    f"  Nonavail $:            {fmt_dollars(fed['dollars_J']):>15} ({pct(fed['dollars_J'], fed['dollars_mfg_obs'])} of mfg-coded $)",
    "",
    "# DLA-specific totals (awarding_sub_agency_name == 'Defense Logistics Agency')",
    f"Transactions:            {dla['n_transactions']:>15,}",
    f"Dollar obligations:      {fmt_dollars(dla['total_dollars']):>15}",
    f"  Domestic (D):          {dla['n_domestic']:>15,} ({pct(dla['n_domestic'], dla['n_mfg_obs'])} of DLA mfg-coded)",
    f"  Nonavailability (J):   {dla['n_J']:>15,} ({pct(dla['n_J'], dla['n_mfg_obs'])} of DLA mfg-coded)",
    f"  Foreign (non-D):       {dla['n_mfg_obs'] - dla['n_domestic']:>15,} ({pct(dla['n_mfg_obs'] - dla['n_domestic'], dla['n_mfg_obs'])} of DLA mfg-coded)",
    f"  Domestic $:            {fmt_dollars(dla['dollars_domestic']):>15} ({pct(dla['dollars_domestic'], dla['dollars_mfg_obs'])} of DLA mfg-coded $)",
    f"  Nonavail $:            {fmt_dollars(dla['dollars_J']):>15} ({pct(dla['dollars_J'], dla['dollars_mfg_obs'])} of DLA mfg-coded $)",
    "",
    "# DLA share of federal-wide totals",
    f"DLA transaction share:   {pct(dla['n_transactions'], fed['n_transactions'])}",
    f"DLA dollar share:        {pct(dla['total_dollars'], fed['total_dollars'])}",
]

txt = "\n".join(lines)
print(txt)
(OUT / "procurement_context_stats.txt").write_text(txt, encoding="utf-8", newline="\n")
print("\n  wrote procurement_context_stats.txt")
