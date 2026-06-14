"""
twfe_helpers.py

Reusable functions for the TWFE event study. Imported by
pipeline/04_event_study/estimate_twfe.py (headline) and
pipeline/07_robustness/run_event_study_refit.py (variant refits).

Architecture:
- `load_panels` and `apply_control_filter` produce the regression-ready polars panel.
- `prep` converts a polars panel to a pandas DataFrame with outcome-specific
  null filtering and the MIN_OBS_PER_NSN >= 2 filter.
- `run_spec` is `feols(...)` with disk-cache: pickled fitted model invalidates
  when the relevant panel parquet is newer than the cache file.
- `make_plot` and `show_table` render outputs (plots saved to disk, tables inline).

Cache directory: data/cache/main/ (variant refits override CACHE_DIR).
Regenerable; .gitignore'd.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
from joblib.externals.cloudpickle import cloudpickle
from pyfixest.estimation import feols
from pyfixest.report import etable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PANEL_DLA_ONLY, PANEL_ENRICHED, cache_dir, event_study_figures

# ---------- Paths and constants ----------
FIGURES_DIR = event_study_figures("main")
CACHE_DIR = cache_dir("main")  # default for run_spec; refits pass cache_dir=
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PANEL_PARQUETS = {
    "dla_only": PANEL_DLA_ONLY,
    "enriched": PANEL_ENRICHED,
}

# Sentinel for control NSNs' event_year (matches build_event_panel.py).
# The dummy `event_year::1000` is collinear with NSN FE (every control row
# has event_year=1000) so pyfixest drops it.
EVENT_YEAR_SENTINEL = 1000

# NSN must have at least this many non-null-outcome cells to enter the regression.
# pyfixest auto-drops singletons anyway; pre-filtering halves the demean work.
MIN_OBS_PER_NSN = 2

# Treated NSN must have at least one cell with event_year in this inclusive
# range (relative to its first_waiver_date) to enter the regression. See
# `apply_pre_window_filter` for the rationale.
PRE_WINDOW = (-5, -1)

# Bump when a pipeline change shifts the regression sample (e.g.,
# new/changed filter in `apply_*` or `prep`). Caches whose embedded version
# does not match are treated as stale and refit. Old cache files written
# before this mechanism existed are also rejected (no embedded version).
_CACHE_VERSION = "v2-pre-window"


# ---------- Panel loading + filtering ----------
def load_panels() -> dict[str, pl.DataFrame]:
    """Load both panel parquets. Returns dict keyed by panel label."""
    return {label: pl.read_parquet(path) for label, path in PANEL_PARQUETS.items()}


def apply_control_filter(panel: pl.DataFrame, label: str | None = None) -> pl.DataFrame:
    """Drop NSNs with no FY>=2022 activity (controls who can't help identify
    post-treatment FY FE). Treated NSNs satisfy automatically (waivers 2022-2025).

    Why: a control NSN with data only in pre-2022 contributes only to
    pre-period FY FE, not to identifying β_k for k>=0.
    """
    n_before_nsns = panel["nsn"].n_unique()
    n_before_cells = len(panel)
    nsns_with_post_2022 = panel.filter(pl.col("fy") >= 2022)["nsn"].unique()
    out = panel.filter(pl.col("nsn").is_in(nsns_with_post_2022))
    n_after_nsns = out["nsn"].n_unique()
    n_after_cells = len(out)
    tag = f" [{label}]" if label else ""
    print(f"  Control filter{tag}: "
          f"{n_before_nsns:,} -> {n_after_nsns:,} NSNs "
          f"({n_before_cells:,} -> {n_after_cells:,} cells)")
    return out


def apply_pre_window_filter(
    panel: pl.DataFrame,
    label: str | None = None,
    pre_window: tuple[int, int] = PRE_WINDOW,
) -> tuple[pl.DataFrame, list[dict]]:
    """Drop treated NSNs lacking any cell with event_year in `pre_window`.

    Why: identification of post-treatment β_k against the k=-1 reference is
    only credible when each treated NSN's pre-period sits in the neighbourhood
    of that reference. A treated NSN whose only pre-period cells are far
    earlier (e.g., 6+ years pre-waiver) anchors β_0 to a stale baseline and
    contributes no within-NSN variation to the event-time coefficients inside
    the headline plot range.

    Aligns the event-study treated sample with the matched_controls and
    synth_analysis Stage 2 sample rules. Those pipelines layer additional
    constraints (donor coverage; per-outcome minimum-observation counts in the
    pre-window), so the resulting samples are similar but not byte-identical.

    Returns (filtered_panel, dropped_info) where `dropped_info` lists one dict
    per excluded NSN — the caller is expected to persist these for audit.
    """
    lo, hi = pre_window
    treated = panel.filter(pl.col("treated") == 1)
    treated_with_pre = set(
        treated
        .filter(pl.col("event_year").is_between(lo, hi, closed="both"))
        ["nsn"].unique()
        .to_list()
    )
    all_treated = set(treated["nsn"].unique().to_list())
    dropped = sorted(all_treated - treated_with_pre)

    dropped_info: list[dict] = []
    for nsn in dropped:
        sub = treated.filter(pl.col("nsn") == nsn)
        eys = sorted({int(e) for e in sub["event_year"].to_list() if e != EVENT_YEAR_SENTINEL})
        dropped_info.append({
            "nsn": nsn,
            "panel_label": label or "",
            "reason": f"no observation in event_year [{lo}, {hi}]",
            "n_cells_total": len(sub),
            "event_years_observed": ",".join(str(e) for e in eys),
        })

    out = panel.filter(~pl.col("nsn").is_in(dropped))
    tag = f" [{label}]" if label else ""
    print(f"  Pre-window filter{tag}: "
          f"{len(all_treated)} -> {len(all_treated) - len(dropped)} treated NSNs "
          f"({len(panel):,} -> {len(out):,} cells)")
    return out, dropped_info


# ---------- prep: polars panel -> pandas regression frame ----------
def prep(panel: pl.DataFrame, outcome: str,
         common_sample_cols: list[str] | None = None) -> pd.DataFrame:
    """Drop rows where outcome (or any common_sample_cols) is null/NaN.
    Apply MIN_OBS_PER_NSN >= 2. Cast to pandas for pyfixest.

    common_sample_cols: covariates whose nulls should also drop rows. Pass the
    union of all controls used across nested specs (e.g., for the primary
    regressions, ['competitive_share', 'sole_source_share']) so spec 1 (no
    controls) and spec 2 (with controls) share the same N.
    """
    keep = panel.filter(pl.col(outcome).is_not_null() & ~pl.col(outcome).is_nan())
    if common_sample_cols:
        for col in common_sample_cols:
            keep = keep.filter(pl.col(col).is_not_null() & ~pl.col(col).is_nan())
    nsn_counts = keep.group_by("nsn").len()
    keep_nsns = nsn_counts.filter(pl.col("len") >= MIN_OBS_PER_NSN)["nsn"]
    keep = keep.filter(pl.col("nsn").is_in(keep_nsns))
    df = keep.to_pandas()
    df["nsn"] = df["nsn"].astype("category")
    df["fy"] = df["fy"].astype("int64")
    # event_year is already populated by build_event_panel.py (treated -> day-precise,
    # untreated -> EVENT_YEAR_SENTINEL). Just ensure int64 for pyfixest.
    df["event_year"] = df["event_year"].astype("int64")
    return df


# ---------- run_spec: cached feols ----------
def _cache_is_fresh(cache_path: Path, panel_label: str) -> bool:
    """True iff cache exists and is newer than the relevant panel parquet."""
    if not cache_path.exists():
        return False
    panel_path = PANEL_PARQUETS[panel_label]
    if not panel_path.exists():
        return False
    return cache_path.stat().st_mtime > panel_path.stat().st_mtime


def _load_cached_model(cache_path: Path):
    """Load a cached fit. Returns the model iff its embedded version matches
    `_CACHE_VERSION`; otherwise returns None so the caller refits. Cache files
    written before the version field existed unpickle to a non-dict and also
    return None."""
    try:
        with open(cache_path, "rb") as f:
            obj = cloudpickle.load(f)
    except Exception:
        return None
    if isinstance(obj, dict) and obj.get("_version") == _CACHE_VERSION:
        return obj.get("model")
    return None


def run_spec(data: pd.DataFrame, outcome: str, controls: str | None,
             cache_key: str, panel_label: str,
             fe: str | None = "nsn + fy",
             vcov: dict | None = None,
             force_refresh: bool = False,
             cache_dir: Path | None = None):
    """Fit a TWFE event study with disk cache.

    Args:
        data: prepared pandas DataFrame from prep().
        outcome: LHS variable name.
        controls: extra RHS covariates, e.g. "competitive_share + sole_source_share". None for naive spec.
        cache_key: unique identifier per spec; pickle saved to cache/{cache_key}.joblib.
        panel_label: "dla_only" or "enriched"; used for cache invalidation.
        fe: fixed-effects spec (after `|`). Pass "nsn", "nsn + fy", or None
            (no FE — pure OLS). Default "nsn + fy".
        vcov: passed to feols. Defaults to {"CRV1": "nsn"}.
        force_refresh: if True, ignore cache and refit.
        cache_dir: cache directory; defaults to the main-variant CACHE_DIR
            (variant refits pass their own).

    Returns the fitted pyfixest model.
    """
    if cache_dir is None:
        cache_dir = CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.joblib"
    if not force_refresh and _cache_is_fresh(cache_path, panel_label):
        cached = _load_cached_model(cache_path)
        if cached is not None:
            return cached
    if vcov is None:
        vcov = {"CRV1": "nsn"}
    formula = f"{outcome} ~ i(event_year, ref=-1)"
    if controls:
        formula += f" + {controls}"
    if fe:
        formula += f" | {fe}"
    print(f"  fitting {cache_key}: {formula}", flush=True)
    m = feols(formula, data=data, vcov=vcov)
    # Use cloudpickle (handles pyfixest's nested-closure attributes that
    # plain pickle/joblib.dump fail on). Embed _CACHE_VERSION so that future
    # pipeline changes can invalidate the cache without touching parquet mtimes.
    try:
        with open(cache_path, "wb") as f:
            cloudpickle.dump({"_version": _CACHE_VERSION, "model": m}, f)
    except Exception as e:
        # Clean up partial file so cache_is_fresh doesn't return True next time.
        try:
            cache_path.unlink(missing_ok=True)
        except Exception:
            pass
        print(f"  WARNING: cache write failed for {cache_key}: {e}", flush=True)
    return m


# ---------- Coefficient extraction ----------
_EVENT_YEAR_RE = re.compile(
    r"^C\(event_year\)\[T\.(-?\d+)\]$"
    r"|^event_year::(-?\d+)$"
    r"|^event_year\[T\.(-?\d+)\]$"
    r"|^event_year\s*[×x]\s*[×x]?\s*(-?\d+)$"
)


def extract_coefs(model) -> list[tuple[int, float, float, float]]:
    """Pull (event_year, coef, ci_low, ci_high) from a fitted model.
    Filters to event_year terms; skips sentinel and any non-event covariates."""
    coefs = model.coef()
    ci = model.confint()
    rows = []
    for name, b in coefs.items():
        m = _EVENT_YEAR_RE.match(str(name).strip())
        if not m:
            continue
        ey = int(next(g for g in m.groups() if g is not None))
        if ey == EVENT_YEAR_SENTINEL:
            continue
        lo = ci.loc[name, "2.5%"]
        hi = ci.loc[name, "97.5%"]
        rows.append((ey, b, lo, hi))
    return sorted(rows, key=lambda r: r[0])


def compute_n_treated_per_ey(panel: pl.DataFrame, event_years: list[int]) -> dict[int, int]:
    """Return {ey: n_unique_treated_NSNs} for plot annotation."""
    out = {}
    treated = panel.filter(pl.col("treated") == 1)
    for ey in event_years:
        out[ey] = treated.filter(pl.col("event_year") == ey)["nsn"].n_unique()
    return out


# ---------- Plotting ----------
# Shared seaborn theme. Mirrors pipeline/06_synth/plot.py so
# figures across the three analyses (event_study, matched_controls,
# synth_analysis) share a visual language. Applied once on first plot call to
# avoid surprising any caller that imports twfe_helpers for non-plot uses.
_THEME_APPLIED = False
_THEME_RC = {
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10,
    "axes.labelweight": "regular",
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "figure.dpi": 130,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
}

# Semantic colors — match pipeline/06_synth/plot.py.
COLOR_NEUTRAL = "#7a7a7a"
COLOR_ZERO_REF = "#222222"


def _ensure_theme() -> None:
    global _THEME_APPLIED
    if _THEME_APPLIED:
        return
    sns.set_theme(
        context="paper", style="whitegrid", palette="deep",
        font="DejaVu Sans", font_scale=1.05, rc=_THEME_RC,
    )
    _THEME_APPLIED = True


def _spec_palette(n: int) -> list[str]:
    """Categorical colors for `n` model columns, drawn from seaborn 'deep'."""
    _ensure_theme()
    return sns.color_palette("deep", n_colors=max(n, 1)).as_hex()


def make_plot(models, headers: list[str], n_treated_per_ey: dict[int, int],
              title: str, slug: str, ey_range: tuple[int, int] = (-4, 3),
              out_dir: Path | None = None) -> Path:
    """Coefplot of one or more fitted models, truncated to `ey_range`.

    Per-event-year treated-NSN counts appear as small `n=N` annotations at the
    top of the axis, matching the convention used in `pipeline/06_synth/plot.py`.
    """
    _ensure_theme()
    out_dir = out_dir or FIGURES_DIR
    fig, ax = plt.subplots(figsize=(8.5, 5))

    n_models = len(models)
    offsets = (
        [0.0] if n_models <= 1
        else [(i - (n_models - 1) / 2) * 0.18 for i in range(n_models)]
    )
    palette = _spec_palette(n_models)

    for i, (model, head) in enumerate(zip(models, headers)):
        rows = [r for r in extract_coefs(model) if ey_range[0] <= r[0] <= ey_range[1]]
        if not rows:
            continue
        xs = [r[0] + offsets[i] for r in rows]
        ys = [r[1] for r in rows]
        lo = [r[1] - r[2] for r in rows]
        hi = [r[3] - r[1] for r in rows]
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", capsize=3,
                    color=palette[i % len(palette)], label=head,
                    markersize=5, linewidth=1.4, zorder=3)

    # Reference category marker
    if ey_range[0] <= -1 <= ey_range[1]:
        ax.scatter([-1], [0], color=COLOR_ZERO_REF, marker="s", s=42,
                   zorder=5, label="ref (k=-1)")

    # Zero line and treatment boundary
    ax.axhline(0, linestyle="--", color=COLOR_NEUTRAL, linewidth=0.8, zorder=2)
    ax.axvline(-0.5, linestyle="--", color=COLOR_NEUTRAL, linewidth=0.8, zorder=2)

    xticks = list(range(ey_range[0], ey_range[1] + 1))
    ax.set_xticks(xticks)
    ax.set_xlim(ey_range[0] - 0.5, ey_range[1] + 0.5)
    ax.set_xlabel("Event year (years from waiver)")
    ax.set_ylabel("Coefficient (95% CI)")
    ax.set_title(title)
    ax.legend(loc="best")

    # N-treated annotations at the top of the axis. Set after the errorbar
    # layer so get_ylim reflects the data extent.
    eys_in_range = [ey for ey in sorted(n_treated_per_ey)
                    if ey_range[0] <= ey <= ey_range[1]]
    if eys_in_range:
        y_top = ax.get_ylim()[1]
        for ey in eys_in_range:
            ax.annotate(f"n={n_treated_per_ey[ey]}", xy=(ey, y_top),
                        xytext=(0, -4), textcoords="offset points",
                        ha="center", va="top",
                        fontsize=7, color=COLOR_NEUTRAL, alpha=0.9)

    sns.despine(ax=ax)
    fig_path = out_dir / f"{slug}.png"
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


# ---------- Table rendering ----------
def _clean_label(label: str) -> str:
    """Convert pyfixest's mangled event_year labels (e.g. 'event_year × × -4',
    'event_year::-4') into clean 'k=-4' / 'k=+0' format."""
    s = str(label)
    if "event_year" in s:
        m = re.search(r"-?\d+", s)
        if m:
            ey = int(m.group())
            return f"k={ey:+d}"
    return label


def show_table(models, headers: list[str], title: str | None = None) -> pd.DataFrame:
    """Render etable inline (notebook display); return the DataFrame.
    Sentinel-event-year row dropped. event_year row labels rewritten to k=±N.
    Significance stars: *** p<0.01, ** p<0.05, * p<0.1.
    No disk writes."""
    # `b*` in coef_fmt marks where significance stars go; signif_code thresholds
    # map to (***, **, *) in increasing order: p<0.01 ***, p<0.05 **, p<0.1 *.
    df = etable(models, model_heads=headers, type="df",
                signif_code=[0.01, 0.05, 0.1],
                coef_fmt="b* \n (se)")
    # Drop sentinel rows (defensive — pyfixest should drop the dummy via
    # collinearity, but if any row leaks through, strip it).
    keep_mask = []
    for tup in df.index:
        name = tup[1] if isinstance(tup, tuple) and len(tup) == 2 else tup
        keep_mask.append(str(EVENT_YEAR_SENTINEL) not in str(name))
    df = df[keep_mask]
    # Rewrite event_year labels in the index.
    new_tuples = []
    for tup in df.index:
        if isinstance(tup, tuple) and len(tup) == 2:
            kind, name = tup
            new_tuples.append((kind, _clean_label(name)))
        else:
            new_tuples.append(_clean_label(tup))
    if new_tuples and isinstance(new_tuples[0], tuple):
        df.index = pd.MultiIndex.from_tuples(new_tuples, names=df.index.names)
    else:
        df.index = pd.Index(new_tuples, name=df.index.name)
    if title:
        print(f"\n## {title}\n")
    try:
        from IPython.display import display
        display(df)
    except ImportError:
        print(df.to_string())
    return df


# ---------- Convenience: derive event years from data ----------
def derive_event_years(panels: dict[str, pl.DataFrame]) -> list[int]:
    """Union of all treated event_year values across panels (sentinel excluded)."""
    eys: set[int] = set()
    for p in panels.values():
        treated_eys = (
            p.filter((pl.col("treated") == 1) & (pl.col("event_year") != EVENT_YEAR_SENTINEL))
            ["event_year"].to_list()
        )
        eys.update(treated_eys)
    return sorted(eys)
