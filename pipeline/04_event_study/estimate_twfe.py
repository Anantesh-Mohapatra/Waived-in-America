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
# # Event Study TWFE — Headline Regressions
#
# Naive TWFE event study with NSN + FY fixed effects. NSN-clustered SE (CRV1).
#
# **Outcomes** (3):
# - `max_log_unit_price` — log of cell-level max unit price
# - `mean_offers` — mean number of offers received per cell
# - `domestic_share` — share of cell's transactions with `place_of_manufacture_code = 'D'`
#
# **Specs** (stargazer-style, per primary outcome — 4 nested columns):
# - **(1) OLS (no FE)** — `outcome ~ i(event_year, ref=-1)`
# - **(2) + NSN FE** — adds `| nsn`
# - **(3) + FY FE** — adds `+ fy` (the headline "naive event study")
# - **(4) + extent controls** — adds `competitive_share + sole_source_share` as covariates
#
# All four specs run on the **common sample** (cells where outcome AND both extent
# covariates are non-null) so the side-by-side N's match. Significance stars:
# *** p<0.01, ** p<0.05, * p<0.1.
#
# **Panels** (2): `dla_only` (every DLA NSN) and `enriched` (DLA + BidLink overlay
# for the 91 BidLink-covered NSNs).
#
# **Control universe**: drop NSNs with no FY≥2022 activity (those can't help
# identify post-treatment FY FE). Replaces the FSC restriction. Also applies
# `MIN_OBS_PER_NSN ≥ 2`.
#
# **Plot truncation**: x-axis shown for `ey ∈ [-4, +3]` (full range estimated;
# β_{-8}…β_{-5} identified off ≤9 NSNs each, so they're shown in the table but
# truncated from the plot). N treated NSNs per event year shown as faint bars.
#
# **Architecture**: regressions run in parallel via `joblib.Parallel(n_jobs=4)`
# with disk-cached fitted models. Iterating on plots/tables is instant once warm.
# Set `FORCE_REFRESH=True` to refit everything.

# %%
import sys
import time
from pathlib import Path

import polars as pl
from joblib import Parallel, delayed

# Locate repo root (script or notebook kernel) and make pipeline/lib importable.
_HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_ROOT = _HERE
while not (_ROOT / "pyproject.toml").exists():
    if _ROOT.parent == _ROOT:
        raise FileNotFoundError(f"repo root (pyproject.toml) not found above {_HERE}")
    _ROOT = _ROOT.parent
if str(_ROOT / "pipeline" / "lib") not in sys.path:
    sys.path.insert(0, str(_ROOT / "pipeline" / "lib"))

import twfe_helpers as h
from paths import REPO_ROOT, panel_logs

# %% [markdown]
# ## Configuration

# %%
FORCE_REFRESH = False  # toggle True to invalidate all caches and refit
N_JOBS = 4              # parallel regression workers
PRIMARY_OUTCOMES = ["max_log_unit_price", "mean_offers", "domestic_share"]
EXTENT_CTRLS = "competitive_share + sole_source_share"
COMMON_NONNULL_COLS = ["competitive_share", "sole_source_share"]

# %% [markdown]
# ## Load + filter panels

# %%
panels = h.load_panels()
print("Raw panels:")
for label, p in panels.items():
    print(f"  {label}: {len(p):,} cells, {p['nsn'].n_unique():,} NSNs, "
          f"{p.filter(pl.col('treated')==1)['nsn'].n_unique()} treated")

panels = {k: h.apply_control_filter(v, k) for k, v in panels.items()}

# Drop treated NSNs lacking pre-window coverage. Aligns the treated sample with
# matched_controls and synth_analysis (Stage 2 in both); see twfe_helpers.
_filtered, _exclusion_rows = {}, []
for _label, _p in panels.items():
    _fp, _dropped = h.apply_pre_window_filter(_p, label=_label)
    _filtered[_label] = _fp
    _exclusion_rows.extend(_dropped)
panels = _filtered
if _exclusion_rows:
    _logs_dir = panel_logs("main")
    _logs_dir.mkdir(parents=True, exist_ok=True)
    _log_path = _logs_dir / "treated_exclusion_log.csv"
    pl.DataFrame(_exclusion_rows).write_csv(_log_path)
    print(f"  Wrote {_log_path.relative_to(REPO_ROOT)} ({len(_exclusion_rows)} rows)")

EVENT_YEARS = h.derive_event_years(panels)
print(f"\nEvent year range (treated NSNs): {EVENT_YEARS}")

# %% [markdown]
# ## Build spec list and prep data

# %%
prepped = {}  # (outcome, panel_label) -> prepared pandas DataFrame
for outcome in PRIMARY_OUTCOMES:
    for panel_label, panel in panels.items():
        prepped[(outcome, panel_label)] = h.prep(
            panel, outcome, common_sample_cols=COMMON_NONNULL_COLS
        )
        n = len(prepped[(outcome, panel_label)])
        n_nsn = prepped[(outcome, panel_label)]["nsn"].nunique()
        print(f"  prep[{outcome}, {panel_label}]: N={n:,}, NSNs={n_nsn:,}")

# Spec list: stargazer-style nested specs.
# (1) OLS (no FE, no controls): naive pooled regression
# (2) + NSN FE: absorbs unit baseline
# (3) + FY FE: + calendar-time shocks (the headline naive event study)
# (4) + extent controls: + cell-level extent_competed_code distribution
SPEC_DEFS = [
    # (spec_name, fe, controls, header_label)
    ("ols",         None,        None,         "OLS (no FE)"),
    ("nsn_fe",      "nsn",       None,         "+ NSN FE"),
    ("nsn_fy_fe",   "nsn + fy",  None,         "+ FY FE"),
    ("with_ctrl",   "nsn + fy",  EXTENT_CTRLS, "+ extent controls"),
]
specs = []
for outcome in PRIMARY_OUTCOMES:
    for panel_label in panels:
        for spec_name, fe, ctrls, _ in SPEC_DEFS:
            specs.append((outcome, panel_label, spec_name, fe, ctrls))
print(f"\nTotal primary specs to fit: {len(specs)} "
      f"({len(PRIMARY_OUTCOMES)} outcomes × {len(panels)} panels × {len(SPEC_DEFS)} specs)")

# %% [markdown]
# ## Fit primary regressions in parallel
#
# Each worker checks the disk cache; only refits if cache is missing or stale
# (panel parquet newer than cache). Set `FORCE_REFRESH=True` above to override.

# %%
def _fit_one(outcome, panel_label, spec_name, fe, controls):
    cache_key = f"{outcome}__{panel_label}__{spec_name}"
    data = prepped[(outcome, panel_label)]
    m = h.run_spec(
        data, outcome, controls,
        cache_key=cache_key, panel_label=panel_label,
        fe=fe,
        force_refresh=FORCE_REFRESH,
    )
    return cache_key, m

t0 = time.time()
fitted = dict(Parallel(n_jobs=N_JOBS, verbose=10)(
    delayed(_fit_one)(*s) for s in specs
))
print(f"\nAll primary specs done in {time.time()-t0:.1f}s")

# %% [markdown]
# ## Display tables and save plots
#
# 6 tables (3 outcomes × 2 panels), 6 plots truncated to ey ∈ [-4, +3].

# %%
TABLE_HEADS = [h_label for _, _, _, h_label in SPEC_DEFS]   # all 4 in table
PLOT_HEADS = TABLE_HEADS[-2:]                                # last 2 in plot (FE-saturated)
PLOT_RANGE = (-4, 3)

for outcome in PRIMARY_OUTCOMES:
    for panel_label, panel in panels.items():
        models_all = [
            fitted[f"{outcome}__{panel_label}__{spec_name}"]
            for spec_name, _, _, _ in SPEC_DEFS
        ]
        n_treated = h.compute_n_treated_per_ey(panel, EVENT_YEARS)
        h.show_table(models_all, TABLE_HEADS,
                     title=f"{outcome} — {panel_label}")
        slug = f"{outcome.split('_')[0]}_{panel_label}"
        # Plot only the FE-saturated specs (= last 2 columns) so the figure stays readable.
        h.make_plot(models_all[-2:], PLOT_HEADS, n_treated,
                    title=f"Effect on {outcome} — {panel_label}",
                    slug=slug, ey_range=PLOT_RANGE)
        print(f"  -> wrote results/main/event_study/figures/{slug}.png")

# %% [markdown]
# ## Done
#
# - Tables rendered inline above; not written to disk.
# - Plots written to `results/main/event_study/figures/`.
# - Fitted models cached at `data/cache/main/`.
# - To refit everything: set `FORCE_REFRESH = True` and re-run.

# %%
print("\nDONE.")
