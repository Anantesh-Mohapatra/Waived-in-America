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
# # Panel Overview Descriptives
#
# Builds the figures and tables that support the thesis's Panel Overview
# subsection (Data section).
#
# **Outputs (in `results/descriptives/`):**
# - `F5_treatment_ds_by_event_year.png` — treatment-panel domestic
#   share by event year.
# - `F4a_ds_by_fiscal_year.png` — domestic share by fiscal year for
#   three groups (full DLA universe, filtered comparator subset, treatment
#   panel aggregate).
# - `paragraph_stats.txt` — inline numbers cited in the prose.
#
# **Run with:** `uv run python pipeline/08_artifacts/panel_overview_descriptives.py`
#
# **Memory note:** uses lazy scans throughout. Only the per-query aggregated
# results are materialized; the full panel is never held in memory.
#
# **Inputs:** the enriched panel, the donor universe, and the treated-NSN
# first-waiver dates (resolved via `pipeline/lib/paths.py`).

# %%
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import polars as pl
import seaborn as sns

# Windows cp1252 terminal: ASCII tables, no Unicode in prints.
pl.Config.set_tbl_formatting("ASCII_MARKDOWN")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(12)

# Seaborn paper-style defaults for the figures.
sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)

# Times New Roman body, STIX math (same font setup as build_thesis_figures.py).
import matplotlib as mpl  # noqa: E402
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
mpl.rcParams["mathtext.fontset"] = "stix"

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

PANEL_PATH = P.PANEL_ENRICHED
DONORS_PATH = P.donor_universe_path("main")
TREATED_PATH = P.TREATMENT_DATES
FIG_OUT   = P.DESCRIPTIVES_FIGURES
TBL_OUT   = P.DESCRIPTIVES_TABLES
STATS_OUT = P.DESCRIPTIVES_STATS
for d in (FIG_OUT, TBL_OUT, STATS_OUT):
    d.mkdir(parents=True, exist_ok=True)


def collect_streaming(lf: pl.LazyFrame) -> pl.DataFrame:
    """Collect with streaming engine to keep peak memory down."""
    return lf.collect(engine="streaming")


# %% [markdown]
# ## Load (small files eager; panel lazy)

# %%
panel_lf = pl.scan_parquet(PANEL_PATH)

donors = pl.read_csv(DONORS_PATH, schema_overrides={"nsn": pl.Utf8, "fsg": pl.Utf8, "fsc": pl.Utf8})
donor_nsns = donors["nsn"].to_list()
print(f"donor universe NSNs: {len(donor_nsns):,}")

treated = pl.read_csv(TREATED_PATH)
treated_nsns_all = treated["nsn"].str.replace_all("-", "", literal=True).to_list()
# Pre-waiver presence filter -> the 27-NSN analysis sample (matches the event
# study, matched controls, synthetic controls, and the origin figures): keep
# treated NSNs with at least one panel cell in event years [-5, -1].
_pre_window = set(
    collect_streaming(
        panel_lf.filter(pl.col("nsn").is_in(treated_nsns_all)
                        & pl.col("event_year").is_between(-5, -1))
        .select("nsn").unique()
    )["nsn"].to_list()
)
treated = treated.filter(
    pl.col("nsn").str.replace_all("-", "", literal=True).is_in(_pre_window))
treated_nsns = treated["nsn"].str.replace_all("-", "", literal=True).to_list()
print(f"treated NSNs: {len(treated_nsns)} (pre-window-present)")

# %% [markdown]
# ## F5 — Treatment domestic share by event year (thesis figure F5)

# %%
# Equal weight per NSN: take the cell-level domestic_share within each
# treated NSN at each event year (= n_domestic / n_mfg_obs in that NSN-FY
# cell), then average those per-NSN shares across NSNs at each event year.
# Matches the aggregation convention used in matched_controls
# (ATT = mean(tau_i)) and in the synth event-time aggregate.
f1 = collect_streaming(
    panel_lf
    .filter(pl.col("nsn").is_in(treated_nsns))
    # Drop NSN-FY cells with no observable manufacturing data: n_mfg_obs == 0
    # yields a NaN cell share (0/0), which would propagate through mean().
    .filter(pl.col("domestic_share").is_not_null()
            & pl.col("domestic_share").is_not_nan())
    .group_by("event_year")
    .agg(
        pl.col("domestic_share").mean().alias("domestic_share"),
        pl.col("nsn").n_unique().alias("n_treated_NSNs"),
    )
    .sort("event_year")
)
# Restrict the plot to event years where >= 3 treated NSNs contribute.
f1_plot = f1.filter(pl.col("n_treated_NSNs") >= 3)
print(f1_plot)

f1_pd = f1_plot.to_pandas()
fig, ax = plt.subplots(figsize=(7, 4))
sns.lineplot(data=f1_pd, x="event_year", y="domestic_share", marker="o", ax=ax)
ax.axvline(-0.5, linestyle="--", color="0.4", linewidth=1.0)
ax.text(-0.5, 1.02, "  waiver", color="0.4", fontsize=9, ha="left", va="bottom")
ax.set_xlabel("Event year (years relative to waiver date)")
ax.set_ylabel("Domestic sourcing share")
ax.set_title("Treatment Panel Domestic Sourcing Share by Event Year")
ax.set_ylim(0, 1.08)
# Percent y-axis (0-100%), with minor gridlines at 10% so low post-waiver
# values are legible. Matches the percent formatting used in F4a/F4b/A0.
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
ax.grid(axis="y", which="minor", linewidth=0.5, alpha=0.4)
sns.despine(ax=ax)
fig.tight_layout()
fig.savefig(FIG_OUT / "F5_treatment_ds_by_event_year.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  wrote F5.png")

# %% [markdown]
# ## F4 — Domestic share by fiscal year: three groups (thesis figure F4)

# %%
def fy_share_lf(filter_expr: pl.Expr | None, label: str) -> pl.DataFrame:
    """Compute weighted domestic share by fy, with the panel filter applied lazily."""
    q = panel_lf
    if filter_expr is not None:
        q = q.filter(filter_expr)
    return collect_streaming(
        q.group_by("fy")
         .agg(
             pl.col("n_domestic").cast(pl.Float64).sum().alias("n_domestic"),
             pl.col("n_mfg_obs").cast(pl.Float64).sum().alias("n_mfg_obs"),
             pl.col("nsn").n_unique().alias("n_NSNs"),
         )
         .with_columns(
             (pl.col("n_domestic") / pl.col("n_mfg_obs")).alias("domestic_share"),
             pl.lit(label).alias("group"),
         )
         .sort("fy")
    )


# Same-FSC comparator pool: NSNs in the FSCs the treatment panel spans,
# excluding the treated NSNs themselves.
treated_fscs = collect_streaming(
    panel_lf
    .filter(pl.col("nsn").is_in(treated_nsns))
    .select(pl.col("fsc").unique())
)["fsc"].to_list()
print(f"Treatment panel FSCs ({len(treated_fscs)}): {sorted(treated_fscs)}")

f2_full = fy_share_lf(None, "Full DLA universe")
f2_filtered = fy_share_lf(pl.col("nsn").is_in(donor_nsns), "Filtered comparison subset")
f2_same_fsc = fy_share_lf(
    pl.col("fsc").is_in(treated_fscs) & ~pl.col("nsn").is_in(treated_nsns),
    "Same-FSC items",
)
f2_treated = fy_share_lf(pl.col("nsn").is_in(treated_nsns), "Treatment panel")
f2 = pl.concat([f2_full, f2_filtered, f2_same_fsc, f2_treated], how="vertical")
print(f2)

# %%
f2_pd = f2.to_pandas()
group_order = [
    "Full DLA universe",
    "Filtered comparison subset",
    "Same-FSC items",
    "Treatment panel",
]
fig, ax = plt.subplots(figsize=(7, 4))
sns.lineplot(
    data=f2_pd,
    x="fy",
    y="domestic_share",
    hue="group",
    hue_order=group_order,
    style="group",
    style_order=group_order,
    markers=True,
    dashes=False,
    ax=ax,
)
ax.set_xlabel("Fiscal year")
ax.set_ylabel("Domestic sourcing share")
ax.set_title("Domestic Sourcing Share by Fiscal Year")
ax.set_ylim(0, 1.08)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
ax.grid(axis="y", which="minor", linewidth=0.5, alpha=0.4)
ax.legend(title=None, loc="lower left", framealpha=0.9)
sns.despine(ax=ax)
fig.tight_layout()
fig.savefig(FIG_OUT / "F4a_ds_by_fiscal_year.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  wrote F4a.png")

# %% [markdown]
# ## T1 — Per-NSN summary stats (appendix table)

# %% [markdown]
# ## Paragraph stats

# %%
# Treatment-panel summary (streaming aggregate).
treated_summary = collect_streaming(
    panel_lf
    .filter(pl.col("nsn").is_in(treated_nsns))
    .select(
        pl.len().alias("n_cells"),
        pl.col("n_transactions").sum().alias("n_transactions"),
        pl.col("fy").min().alias("fy_min"),
        pl.col("fy").max().alias("fy_max"),
        pl.col("event_year").min().alias("ey_min"),
        pl.col("event_year").max().alias("ey_max"),
        pl.col("nsn").n_unique().alias("n_nsns"),
    )
).row(0, named=True)

filtered_summary = collect_streaming(
    panel_lf
    .filter(pl.col("nsn").is_in(donor_nsns))
    .select(
        pl.col("n_transactions").sum().alias("n_transactions"),
        pl.col("nsn").n_unique().alias("n_nsns"),
    )
).row(0, named=True)

full_summary = collect_streaming(
    panel_lf.select(
        pl.col("n_transactions").sum().alias("n_transactions"),
        pl.col("nsn").n_unique().alias("n_nsns"),
    )
).row(0, named=True)

container_count = sum(1 for n in treated_nsns if n.startswith("8150"))
non_container_count = len(treated_nsns) - container_count

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
lines = [
    f"=== Panel Overview paragraph stats — generated {now} ===",
    "",
    "# Treatment panel",
    f"NSNs:                {treated_summary['n_nsns']}",
    f"  Containers (8150): {container_count}",
    f"  Non-containers:    {non_container_count}",
    f"Cells:               {treated_summary['n_cells']:,}",
    f"Transactions:        {treated_summary['n_transactions']:,}",
    f"FY range:            {treated_summary['fy_min']} - {treated_summary['fy_max']}",
    f"Event-year range:    {treated_summary['ey_min']} - {treated_summary['ey_max']}",
    "",
    "# Comparison universe",
    f"Full DLA universe NSNs:            {full_summary['n_nsns']:,}",
    f"Filtered comparison subset NSNs:   {filtered_summary['n_nsns']:,}",
    f"  Filter drops:                    "
    f"{full_summary['n_nsns'] - filtered_summary['n_nsns']:,} "
    f"({100 * (full_summary['n_nsns'] - filtered_summary['n_nsns']) / full_summary['n_nsns']:.1f}%)",
    "",
    "# Transaction totals",
    f"Full DLA universe transactions:            {full_summary['n_transactions']:,}",
    f"Filtered comparison subset transactions:   {filtered_summary['n_transactions']:,}",
    f"Treatment panel transactions:              {treated_summary['n_transactions']:,}",
]

txt = "\n".join(lines)
print(txt)
(STATS_OUT / "paragraph_stats.txt").write_text(txt, encoding="utf-8", newline="\n")
print("\n  wrote paragraph_stats.txt")
