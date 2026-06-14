"""Stage 5 - Aggregate plots for synth fits (seaborn).

Reads:
  results/tables/synth_att.csv      (per-fit summary)
  results/tables/effect_path/*.csv

Writes per outcome into results/figures/aggregate/:
  forest__<outcome>.png                  ATT ± 95% CI for each treated
  event_study_aggregate__<outcome>.png   mean per-EY effect across treated (+demeaned)
  event_study_grid__<outcome>.png        per-NSN event-study subplot grid (+demeaned)

Trial mode (--trial): reads from results/trial/ instead of results/tables/.

TIME DIMENSION = day-precise event_year throughout.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

# ---------- seaborn theme ----------
sns.set_theme(
    context="paper",
    style="whitegrid",
    palette="deep",
    font="DejaVu Sans",
    font_scale=1.05,
    rc={
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
    },
)

# Semantic colors used across plots.
COLOR_EFFECT = "#1b6f4a"    # olive green
COLOR_NEUTRAL = "#7a7a7a"   # muted gray
COLOR_OBSERVED = "#c0392b"  # warm red, significant estimates
COLOR_ZERO_REF = "#222222"  # near-black for zero/reference lines

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths as P

OUTCOMES = ("max_log_unit_price", "mean_offers", "domestic_share")
OUTCOME_LABELS = {
    "max_log_unit_price": "max log unit price",
    "mean_offers": "mean offers",
    "domestic_share": "domestic share",
}


def get_paths(trial: bool, variant: str = "main"):
    trial_base = P.DATA / "synth_trial" / variant
    tables = trial_base if trial else P.synth_tables(variant)
    figs_root = (trial_base / "figures") if trial else P.synth_figures(variant)
    figs_aggregate = figs_root / "aggregate"
    figs_aggregate.mkdir(parents=True, exist_ok=True)
    return {
        "att": tables / "synth_att.csv",
        "effect": tables / "effect_path",
        "figs_aggregate": figs_aggregate,
    }


def label(outcome: str) -> str:
    return OUTCOME_LABELS.get(outcome, outcome)


def add_treatment_marker(ax, label_text: str = "treatment"):
    """Vertical dashed line at EY=0 boundary, with subtle label."""
    ax.axvline(-0.5, color=COLOR_NEUTRAL, linestyle="--", linewidth=0.9, alpha=0.7)
    ax.annotate(
        label_text, xy=(-0.5, ax.get_ylim()[1]), xytext=(2, -10),
        textcoords="offset points", ha="left", va="top",
        fontsize=8, color=COLOR_NEUTRAL, style="italic",
    )


def plot_forest(att_df: pl.DataFrame, outcome: str, out: Path):
    sub = att_df.filter(pl.col("outcome") == outcome).sort("att")
    if not len(sub):
        return
    pdf = sub.to_pandas()
    pdf["sig"] = pdf["empirical_p"].fillna(1) < 0.05
    y = list(range(len(pdf)))
    fig, ax = plt.subplots(figsize=(8, max(3.5, len(pdf) * 0.28)))
    xerr_low = pdf["att"] - pdf["ci_lo"]
    xerr_high = pdf["ci_hi"] - pdf["att"]
    for i, (att, lo, hi, sig) in enumerate(zip(pdf["att"], xerr_low, xerr_high, pdf["sig"])):
        c = COLOR_OBSERVED if sig else COLOR_NEUTRAL
        ax.errorbar(att, i, xerr=[[lo], [hi]], fmt="o",
                    color=c, ecolor=c, capsize=3, markersize=6,
                    elinewidth=1.5, alpha=0.95)
    ax.axvline(0, color=COLOR_ZERO_REF, linewidth=0.9, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(pdf["treated_nsn"], fontsize=8)
    ax.set_xlabel(f"ATT ({label(outcome)})")
    ax.set_title(
        f"Per-treated ATT $\\pm$ 95% CI: {label(outcome)}\n"
        f"red = empirical $p$ < 0.05, gray = not significant"
    )
    sns.despine(ax=ax)
    fig.savefig(out)
    plt.close(fig)


def load_outcome_event_paths(att_df: pl.DataFrame, effect_dir: Path, outcome: str):
    """Read all effect_path CSVs for fits of this outcome; return list of (tnsn, df)."""
    sub = att_df.filter((pl.col("outcome") == outcome)
                       & (pl.col("error").is_null() | (pl.col("error") == "NA")))
    out = []
    for r in sub.iter_rows(named=True):
        tnsn = r["treated_nsn"]
        p = effect_dir / f"{tnsn}__{outcome}.csv"
        if not p.exists():
            continue
        df = pl.read_csv(p)
        out.append((tnsn, df.sort("event_year")))
    return out


def plot_aggregate_event_study(att_df: pl.DataFrame, effect_dir: Path,
                               outcome: str, out: Path,
                               column: str = "effect",
                               title_suffix: str = ""):
    """Average per-EY effect across treated NSNs.

    column: which effect column to plot.
      "effect"          = raw (treated - synthetic); includes any constant
                          pre-period level gap that synthdid's DID absorbs.
      "effect_demeaned" = effect minus each NSN's pre-period mean of effect.
                          Pre-period clusters near 0; post-period values are
                          the change relative to baseline (the DID-relevant
                          quantity).
    """
    paths = load_outcome_event_paths(att_df, effect_dir, outcome)
    if not paths:
        return

    rows = []
    for tnsn, df in paths:
        for r in df.iter_rows(named=True):
            rows.append({"treated_nsn": tnsn, "event_year": r["event_year"],
                         "y": r[column]})
    long = pl.DataFrame(rows)

    summary = (
        long.group_by("event_year")
        .agg(
            pl.col("y").mean().alias("mean"),
            pl.col("y").quantile(0.25).alias("q25"),
            pl.col("y").quantile(0.75).alias("q75"),
            pl.col("treated_nsn").n_unique().alias("n_treated"),
        )
        .sort("event_year")
        .filter(pl.col("n_treated") >= 2)
    )

    fig, ax = plt.subplots(figsize=(8.5, 5))

    for tnsn, df in paths:
        ax.plot(df["event_year"], df[column], color=COLOR_NEUTRAL,
                linewidth=0.7, alpha=0.25)

    ax.fill_between(
        summary["event_year"], summary["q25"], summary["q75"],
        color=COLOR_EFFECT, alpha=0.18, label="Inter-quartile range",
    )
    ax.plot(summary["event_year"], summary["mean"], color=COLOR_EFFECT,
            linewidth=2.4, label="Mean effect", marker="o", markersize=6)

    ax.axhline(0, color=COLOR_ZERO_REF, linewidth=0.9, alpha=0.7)
    add_treatment_marker(ax)

    y_max = ax.get_ylim()[1]
    for ey, n in zip(summary["event_year"], summary["n_treated"]):
        ax.annotate(f"n={n}", xy=(ey, y_max), xytext=(0, -4),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=7, color=COLOR_NEUTRAL, alpha=0.85)

    ylab = (f"{label(outcome)}: treated $-$ synthetic" if column == "effect"
            else f"{label(outcome)}: change vs pre-period baseline")
    ax.set_xlabel("Event year (day-precise, relative to waiver date)")
    ax.set_ylabel(ylab)
    ax.set_title(
        f"Aggregate event study{title_suffix}: {label(outcome)}\n"
        f"{len(paths)} treated NSNs (unbalanced post-period; $n$ shown per EY)"
    )
    ax.legend(loc="best")
    sns.despine(ax=ax)
    fig.savefig(out)
    plt.close(fig)


def plot_event_study_grid(att_df: pl.DataFrame, effect_dir: Path,
                          outcome: str, out: Path,
                          column: str = "effect",
                          title_suffix: str = ""):
    """Per-NSN event-study subplots in one figure (grid).

    column: same semantics as plot_aggregate_event_study.
    """
    paths = load_outcome_event_paths(att_df, effect_dir, outcome)
    if not paths:
        return

    n = len(paths)
    ncols = 5 if n > 16 else 4
    nrows = (n + ncols - 1) // ncols

    all_y = [v for _, df in paths for v in df[column].to_list()]
    y_lo, y_hi = min(all_y), max(all_y)
    pad = (y_hi - y_lo) * 0.08
    y_lim = (y_lo - pad, y_hi + pad)

    all_eys = sorted({ey for _, df in paths for ey in df["event_year"].to_list()})
    x_lo, x_hi = min(all_eys) - 0.3, max(all_eys) + 0.3

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 2.6, nrows * 2.1),
        sharex=True, sharey=True,
    )
    axes_flat = axes.flatten() if n > 1 else [axes]

    for i, (tnsn, df) in enumerate(paths):
        ax = axes_flat[i]
        pre = df.filter(pl.col("is_pre") == True)
        post = df.filter(pl.col("is_pre") == False)
        ax.plot(df["event_year"], df[column], color=COLOR_EFFECT,
                linewidth=1.4, alpha=0.95)
        ax.scatter(pre["event_year"], pre[column], facecolors="white",
                   edgecolors=COLOR_EFFECT, s=22, linewidth=1.0, zorder=3)
        ax.scatter(post["event_year"], post[column], color=COLOR_EFFECT,
                   s=22, zorder=3)
        ax.axhline(0, color=COLOR_ZERO_REF, linewidth=0.6, alpha=0.6)
        ax.axvline(-0.5, color=COLOR_NEUTRAL, linestyle="--",
                   linewidth=0.6, alpha=0.6)
        ax.set_title(tnsn, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lim)

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    ylab = (f"{label(outcome)}: treated $-$ synthetic" if column == "effect"
            else f"{label(outcome)}: change vs pre-period baseline")
    fig.suptitle(
        f"Per-NSN event studies{title_suffix}: {label(outcome)} ($n$ = {n})",
        fontsize=13, fontweight="semibold", y=0.995,
    )
    fig.supxlabel("Event year", fontsize=10)
    fig.supylabel(ylab, fontsize=10)
    sns.despine(fig=fig)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.97))
    fig.savefig(out)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trial", action="store_true")
    p.add_argument("--variant", choices=["main", "dla_only"], default="main")
    args = p.parse_args()

    paths = get_paths(args.trial, args.variant)
    if not paths["att"].exists():
        raise SystemExit(f"Missing {paths['att']}")

    att = pl.read_csv(paths["att"], schema_overrides={"treated_nsn": pl.Utf8})
    print(f"Loaded {len(att)} fit rows from {paths['att']}")

    fa = paths["figs_aggregate"]
    n_made = 0

    for o in OUTCOMES:
        plot_forest(att, o, fa / f"forest__{o}.png")
        n_made += 1
        plot_aggregate_event_study(att, paths["effect"], o,
                                   fa / f"event_study_aggregate__{o}.png",
                                   column="effect")
        plot_aggregate_event_study(att, paths["effect"], o,
                                   fa / f"event_study_aggregate__demeaned__{o}.png",
                                   column="effect_demeaned",
                                   title_suffix=" (demeaned)")
        plot_event_study_grid(att, paths["effect"], o,
                              fa / f"event_study_grid__{o}.png",
                              column="effect")
        plot_event_study_grid(att, paths["effect"], o,
                              fa / f"event_study_grid__demeaned__{o}.png",
                              column="effect_demeaned",
                              title_suffix=" (demeaned)")
        n_made += 4

    print(f"Wrote {n_made} figures: aggregate -> {fa}")


if __name__ == "__main__":
    main()
