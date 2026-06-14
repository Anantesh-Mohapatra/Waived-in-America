"""Generate seaborn figures for the thesis Analysis section.

Reads canonical result CSVs from the three analysis pipelines and emits
PNGs to results/descriptives/figures/. Reads the event-study Spec 3
coefficients from `event_study.json` (the regenerated intermediate), so the
figures stay in lock-step with the regression output.

Re-run after any pipeline change with:
    uv run python pipeline/08_artifacts/build_thesis_figures.py
"""
from __future__ import annotations

from pathlib import Path

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from scipy import stats as _stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths as P

REPO = P.REPO_ROOT
OUT = P.DESCRIPTIVES_FIGURES
OUT.mkdir(parents=True, exist_ok=True)

WAIVERS_CSV = P.WAIVERS_CLEANED

# Import the canonical 3-spec ladder data from the table generator so both
# the figures and the tables draw from the same source of truth.
sys.path.insert(0, str(Path(__file__).parent))
from build_thesis_tables import LADDER_DATA, _compute_foreign_share  # noqa: E402
import json
import es_stats
# ES_COEFS (spec-3 coefplot data) is read from the main event_study.json via
# the same transform build_thesis_tables uses, so the figure can never diverge
# from the regression output.
_ES_ORDER = ["domestic_share", "max_log_unit_price", "mean_offers"]
_ES_JSON = json.loads(P.event_study_json("main").read_text(encoding="utf-8"))

sns.set_theme(context="paper", style="whitegrid", palette="deep")

# Times New Roman body, STIX math (designed as a Times-compatible math font).
import matplotlib as mpl  # noqa: E402
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
mpl.rcParams["mathtext.fontset"] = "stix"

EN = "–"   # en dash for ranges in axis text


# ---------------------------------------------------------------------------
# 1. Event-study coefficient plots (enriched panel, Spec 3 = NSN + FY FE)
# ---------------------------------------------------------------------------
# Spec-3 (+NSN+FY FE) coefficients from the main event_study.json.
ES_COEFS = {o: es_stats.es_spec_from_json(_ES_JSON[o]["es_coefs"]) for o in _ES_ORDER}


def plot_event_study(outcome: str, spec: dict, *, out_dir: Path = OUT) -> Path:
    df = pd.DataFrame(spec["coefs"], columns=["k", "beta", "se"])
    df["ci_lo"] = df["beta"] - 1.96 * df["se"]
    df["ci_hi"] = df["beta"] + 1.96 * df["se"]
    # Truncate the noisy endpoint window so the plot reads cleanly.
    df = df[df["k"].between(-4, 3)].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=150)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(-0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.errorbar(
        df["k"], df["beta"],
        yerr=1.96 * df["se"],
        fmt="o", color=sns.color_palette("deep")[0],
        ecolor=sns.color_palette("deep")[0], elinewidth=1.1, capsize=3,
        markersize=5,
    )
    # Mark the reference period with an open circle at zero.
    ref_k = df.loc[df["k"] == -1, "k"].iloc[0]
    ax.scatter([ref_k], [0], facecolors="white",
               edgecolors=sns.color_palette("deep")[0], s=60, zorder=3,
               label="Reference (k = -1)")

    ax.set_xlabel("Event year (years since first waiver)")
    ax.set_ylabel(spec["ylabel"])
    ax.set_title(spec["title"])
    ax.set_xticks(range(-4, 4))
    ax.legend(loc="best", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#cccccc", fontsize=9)

    # N treated NSNs per event year, as small labels at the top of the plot.
    n_per_ey = spec.get("n_treated_per_ey", {})
    if n_per_ey:
        y_top = ax.get_ylim()[1]
        for ey in range(-4, 4):
            if ey in n_per_ey:
                ax.annotate(f"n={n_per_ey[ey]}", xy=(ey, y_top),
                            xytext=(0, -4), textcoords="offset points",
                            ha="center", va="top",
                            fontsize=7, color="gray", alpha=0.9)

    p_txt = "<0.001" if spec["pretrend_p"] < 0.001 else f"{spec['pretrend_p']:.3f}"
    note = (
        f"NSN + FY FE; error bars are 95% CIs with CRV1 SE clustered on NSN  "
        f"N = {spec['n_obs']:,}; clusters = {spec['n_clusters']:,}  "
        f"joint pre-trend p = {p_txt}"
    )
    fig.text(0.5, -0.02, note, ha="center", fontsize=8, color="gray")
    fig.tight_layout()
    path = out_dir / f"F{6 + ['domestic_share','max_log_unit_price','mean_offers'].index(outcome)}_event_study_coefplot_{outcome}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 2. Matched controls per-NSN tau dotplots
# ---------------------------------------------------------------------------
NN_DID_TAU = P.matched_tables("main") / "nn_did_tau_per_nsn_fsg_off.csv"
NN_DID_SUMMARY = P.matched_tables("main") / "nn_did_summary_fsg_off.csv"

OUTCOME_LABEL = {
    "domestic_share": "Domestic Share",
    "max_log_unit_price": "Max Log Unit Price",
    "mean_offers": "Mean Offers",
}


def plot_nn_did_dotplot(outcome: str, post_label: str = "headline", *,
                        tau_path: Path = NN_DID_TAU,
                        summary_path: Path = NN_DID_SUMMARY,
                        out_dir: Path = OUT) -> Path:
    tau = pd.read_csv(tau_path)
    tau = tau[(tau["outcome"] == outcome) & (tau["post_label"] == post_label)].copy()
    tau = tau.sort_values("tau").reset_index(drop=True)
    tau["label"] = tau["treated_nsn"].astype(str).str[-4:]

    summary = pd.read_csv(summary_path)
    row = summary[(summary["outcome"] == outcome) & (summary["post_label"] == post_label)].iloc[0]
    att, se = row["att"], row["se"]
    n_treated = int(row["n_treated"])
    t_crit = _stats.t.ppf(0.975, df=n_treated - 1) if n_treated > 1 else float("nan")
    ci_lo, ci_hi = att - t_crit * se, att + t_crit * se
    window_label = {"headline": "EY ≥ 0", "ey2plus": "EY ≥ 2"}[post_label]

    height = max(4.0, 0.18 * len(tau) + 1.0)
    fig, ax = plt.subplots(figsize=(6.5, height), dpi=150)
    ax.axvline(0, color="black", linewidth=0.8)

    # 95% CI band around the pooled ATT — answers "is this significant?"
    # at a glance. Per-NSN points have no individual CIs (see write-up).
    ax.axvspan(ci_lo, ci_hi,
               color=sns.color_palette("deep")[3], alpha=0.15,
               label="ATT 95% CI")
    ax.axvline(att, color=sns.color_palette("deep")[3], linestyle="--",
               linewidth=1.2, label=f"ATT = {att:+.3f}")

    ax.scatter(tau["tau"], range(len(tau)),
               color=sns.color_palette("deep")[0], s=35, zorder=3)
    ax.set_yticks(range(len(tau)))
    ax.set_yticklabels(tau["label"], fontsize=8)
    ax.set_ylabel("Treated NSN (last 4 digits)")
    ax.set_xlabel(rf"Per-NSN $\tau_i$ ({OUTCOME_LABEL[outcome]})")
    ax.set_title(rf"Matched Controls: Per-NSN $\tau_i$ Distribution, {OUTCOME_LABEL[outcome]} ({window_label})")
    ax.legend(loc="best", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#cccccc", fontsize=9)

    note = (f"N treated = {int(row['n_treated'])}; "
            f"ATT = {att:+.3f} (SE {se:.3f}); shaded band is the ATT 95% CI. "
            r"Per-NSN $\tau_i$ have no individual CIs.")
    fig.text(0.5, -0.01, note, ha="center", fontsize=7.5, color="gray")
    fig.tight_layout()
    # Appendix figures A4-A9: 2 windows per outcome, headline then ey2plus.
    _oi = ["domestic_share", "max_log_unit_price", "mean_offers"].index(outcome)
    _fig_num = 4 + 2 * _oi + (0 if post_label == "headline" else 1)
    suffix = "" if post_label == "headline" else f"_{post_label}"
    path = out_dir / f"A{_fig_num}_matched_dotplot_{outcome}{suffix}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 3. Synthetic controls forest plots
# ---------------------------------------------------------------------------
SYNTH_ATT = P.synth_tables("main") / "synth_att.csv"
SYNTH_SUMMARY = P.synth_tables("main") / "synth_summary.csv"


def plot_synth_forest(outcome: str, *,
                      att_path: Path = SYNTH_ATT,
                      summary_path: Path = SYNTH_SUMMARY,
                      out_dir: Path = OUT) -> Path:
    att = pd.read_csv(att_path)
    att = att[att["outcome"] == outcome].dropna(subset=["att"]).copy()
    att = att.sort_values("att").reset_index(drop=True)
    att["label"] = att["treated_nsn"].astype(str).str[-4:]
    att["sig"] = att["empirical_p"] < 0.05

    summary = pd.read_csv(summary_path)
    srow = summary[summary["outcome"] == outcome].iloc[0]

    height = max(4.0, 0.20 * len(att) + 1.0)
    fig, ax = plt.subplots(figsize=(6.5, height), dpi=150)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(srow["median_att"], color=sns.color_palette("deep")[3],
               linestyle="--", linewidth=1.0,
               label=f"median ATT = {srow['median_att']:.3f}")

    colors = [sns.color_palette("deep")[3] if s else sns.color_palette("deep")[0]
              for s in att["sig"]]
    for i, (_, r) in enumerate(att.iterrows()):
        ax.plot([r["ci_lo"], r["ci_hi"]], [i, i],
                color=colors[i], linewidth=1.0, alpha=0.7)
    ax.scatter(att["att"], range(len(att)), c=colors, s=35, zorder=3)

    ax.set_yticks(range(len(att)))
    ax.set_yticklabels(att["label"], fontsize=8)
    ax.set_ylabel("Treated NSN (last 4 digits)")
    ax.set_xlabel(f"Synth ATT ({OUTCOME_LABEL[outcome]})")
    ax.set_title(f"Synthetic Controls: Per-NSN ATT, {OUTCOME_LABEL[outcome]}")

    sig_handle = plt.Line2D([0], [0], marker="o", color="w",
                            markerfacecolor=sns.color_palette("deep")[3],
                            markersize=7, label="Empirical p < 0.05")
    nonsig_handle = plt.Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=sns.color_palette("deep")[0],
                               markersize=7, label="Not significant")
    handles = [sig_handle, nonsig_handle]
    handles.append(ax.lines[1])   # median ATT line, already constructed
    ax.legend(handles=handles, loc="best", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#cccccc", fontsize=9)

    n_sig = int(srow["n_empirical_p_lt_05"])
    n_fits = int(srow["n_fits"])
    note = (
        f"N fits = {n_fits}; significant (empirical p < 0.05): {n_sig}/{n_fits}  "
        f"placebo 95% CIs; subsample reps per fit"
    )
    fig.text(0.5, -0.01, note, ha="center", fontsize=8, color="gray")
    fig.tight_layout()
    # Appendix figures A10-A12.
    path = out_dir / f"A{10 + ['domestic_share', 'max_log_unit_price', 'mean_offers'].index(outcome)}_synth_forest_{outcome}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 4. Synthetic Controls — aggregate event-time path
# ---------------------------------------------------------------------------
# Per-fit effect paths live as one CSV per (treated NSN, outcome) in
# results/main/synth/tables/effect_path/. We aggregate them
# across treated NSNs at each event year to show how the mean synth
# effect evolves over time, with an IQR band for the spread.
EFFECT_PATH_DIR = P.synth_tables("main") / "effect_path"


def plot_synth_event_time(outcome: str, *,
                          effect_path_dir: Path = EFFECT_PATH_DIR,
                          out_dir: Path = OUT) -> Path:
    # Use effect_demeaned so the pre-period centers on zero by construction
    # and the post-period values represent the treatment effect itself,
    # not the pre-existing fit gap.
    rows: list[pd.DataFrame] = []
    for csv_path in sorted(effect_path_dir.glob(f"*__{outcome}.csv")):
        df = pd.read_csv(csv_path)
        df["treated_nsn"] = csv_path.stem.split("__")[0]
        rows.append(df[["event_year", "effect_demeaned", "treated_nsn"]])
    long = pd.concat(rows, ignore_index=True).rename(columns={"effect_demeaned": "effect"})

    # Aggregate across NSNs at each event year, with a 95% CI on the mean
    # (t-distribution with df = n - 1). The CI is what determines whether
    # an event-year aggregate is statistically distinguishable from zero.
    def _agg(g: pd.Series) -> pd.Series:
        n = len(g)
        m = g.mean()
        se = g.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
        t_crit = _stats.t.ppf(0.975, df=n - 1) if n > 1 else float("nan")
        half = t_crit * se if n > 1 else float("nan")
        return pd.Series({"mean": m, "ci_lo": m - half, "ci_hi": m + half, "count": n})

    agg = (long.groupby("event_year")["effect"].apply(_agg)
           .unstack().reset_index())

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=150)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(-0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    pre = agg[agg["event_year"] < 0]
    post = agg[agg["event_year"] >= 0]

    for sub in (pre, post):
        ax.fill_between(sub["event_year"], sub["ci_lo"], sub["ci_hi"],
                        color=sns.color_palette("deep")[2], alpha=0.2,
                        label="95% CI on mean" if sub is pre else None)
        ax.plot(sub["event_year"], sub["mean"],
                color=sns.color_palette("deep")[2], linewidth=2.0,
                marker="o", markersize=5,
                label="Mean effect" if sub is pre else None)

    ax.set_xlabel("Event year (years since waiver)")
    ax.set_ylabel(f"Synth effect: treated minus synthetic ({OUTCOME_LABEL[outcome]})")
    ax.set_title(f"Synthetic Controls Event-Time Path: {OUTCOME_LABEL[outcome]}")
    ax.legend(loc="best", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#cccccc", fontsize=9)

    # N treated NSNs per event year, as small labels at the top of the plot.
    y_top = ax.get_ylim()[1]
    for _, row in agg.iterrows():
        ax.annotate(f"n={int(row['count'])}", xy=(row["event_year"], y_top),
                    xytext=(0, -4), textcoords="offset points",
                    ha="center", va="top",
                    fontsize=7, color="gray", alpha=0.9)
    fig.tight_layout()
    path = out_dir / f"F{9 + ['domestic_share','max_log_unit_price','mean_offers'].index(outcome)}_synth_event_time_{outcome}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 5. Appendix figure: 3-spec event-study coefplot per outcome
# ---------------------------------------------------------------------------
# Per-outcome visualization of the same ladder shown in the appendix HTML
# tables: Spec 1 (OLS), Spec 2 (+NSN FE), Spec 3 (+NSN+FY FE) overlaid
# with horizontal offsets so the three series are visually separable at
# each event year.
SPEC_LABELS_SHORT = ["(1) OLS", "(2) + NSN FE", "(3) + NSN + FY FE"]


def plot_event_study_ladder(outcome: str, *,
                            ladder_data: dict | None = None,
                            out_dir: Path = OUT) -> Path:
    data = (ladder_data if ladder_data is not None else LADDER_DATA)[outcome]
    coefs = data["coefs"]
    # Match the main coefplot range so the appendix figure reads alongside it.
    xs_base = [k for k in sorted(coefs.keys()) if -4 <= k <= 3]

    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=150)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(-0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    palette = sns.color_palette("deep", n_colors=3)
    offsets = [-0.18, 0.0, 0.18]

    for spec_i in range(3):
        ys = [coefs[k][spec_i][0] for k in xs_base]
        ses = [coefs[k][spec_i][1] for k in xs_base]
        xs = [k + offsets[spec_i] for k in xs_base]
        ax.errorbar(xs, ys,
                    yerr=[1.96 * s for s in ses],
                    fmt="o", color=palette[spec_i],
                    ecolor=palette[spec_i], elinewidth=1.1,
                    capsize=3, markersize=4.5,
                    label=SPEC_LABELS_SHORT[spec_i])

    # Open marker at the omitted reference period.
    ax.scatter([-1], [0], facecolors="white", edgecolors="gray",
               s=55, zorder=4, label="Reference (k = -1)")

    ax.set_xlabel("Event year (years since first waiver)")
    ax.set_ylabel("Coefficient on event-year dummy")
    ax.set_title(f"Event Study Specification Ladder: {data['title']}")
    ax.set_xticks(range(-4, 4))
    ax.legend(loc="best", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#cccccc", fontsize=9)

    note = (
        "Three specifications overlaid (small x-offsets for legibility). "
        "Error bars are 95% CIs with NSN-clustered standard errors."
    )
    fig.text(0.5, -0.02, note, ha="center", fontsize=8, color="gray")
    fig.tight_layout()
    # Appendix figures A1-A3.
    path = out_dir / f"A{1 + ['domestic_share', 'max_log_unit_price', 'mean_offers'].index(outcome)}_event_study_ladder_{outcome}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 6. Waiver universe descriptives (Data section)
# ---------------------------------------------------------------------------
# Three stacked-bar figures: waiver counts by fiscal year, with top
# agencies / duration groups / top PSC codes broken out.
# Source: output/procurement-waivers-cleaned.csv (the cleaned waiver portal
# extract). FY 2026 is in-progress and is dropped so the bars compare apples
# to apples across complete fiscal years.

TOP_AGENCIES_FULL = [
    "NATIONAL INSTITUTE OF STANDARDS AND TECHNOLOGY",
    "VETERANS AFFAIRS, DEPARTMENT OF",
    "NATIONAL INSTITUTES OF HEALTH",
    "NATIONAL OCEANIC AND ATMOSPHERIC ADMINISTRATION",
    "AGRICULTURAL RESEARCH SERVICE",
    "CENTERS FOR DISEASE CONTROL AND PREVENTION",
]
AGENCY_SHORT = {
    "NATIONAL INSTITUTE OF STANDARDS AND TECHNOLOGY": "NIST",
    "VETERANS AFFAIRS, DEPARTMENT OF": "VA",
    "NATIONAL INSTITUTES OF HEALTH": "NIH",
    "NATIONAL OCEANIC AND ATMOSPHERIC ADMINISTRATION": "NOAA",
    "AGRICULTURAL RESEARCH SERVICE": "ARS",
    "CENTERS FOR DISEASE CONTROL AND PREVENTION": "CDC",
}
AGENCY_ORDER = ["NIST", "VA", "NIH", "NOAA", "ARS", "CDC", "Other"]

DURATION_MAP = {
    "Instant Delivery Only":       "Instant delivery",
    "0 - 6 months":                "0-6 months",
    "Between 6 months and 1 year": "6-12 months",
    "Between 1 and 2 years":       "1-2 years",
    "Between 2 and 3 years":       "2+ years",
    "Between 3 and 5 years":       "2+ years",
    "More than 5 years":           "2+ years",
}
DURATION_ORDER = ["Instant delivery", "0-6 months", "6-12 months",
                  "1-2 years", "2+ years", "Unknown"]

PSC_LABELS = {
    "6640": "6640 Lab equipment",
    "6515": "6515 Medical/surgical",
    "6505": "6505 Drugs/biologicals",
    "8150": "8150 Freight containers",
}
PSC_ORDER = ["6640 Lab equipment", "6515 Medical/surgical",
             "6505 Drugs/biologicals", "8150 Freight containers", "Other PSCs"]


def _load_waivers_for_counts() -> pl.DataFrame:
    """Cleaned waivers, FY < 2026 (drop the in-progress year)."""
    return (pl.read_csv(WAIVERS_CSV, infer_schema_length=2000)
            .filter(pl.col("fiscal_year") < 2026))


def _stacked_bar(pivot: pl.DataFrame, fys: list[int], series_order: list[str],
                 colors: list, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=150)
    bottoms = np.zeros(len(fys))
    for i, name in enumerate(series_order):
        vals = (pivot[name].to_numpy() if name in pivot.columns
                else np.zeros(len(fys)))
        ax.bar(fys, vals, bottom=bottoms, label=name, color=colors[i],
               edgecolor="white", linewidth=0.5)
        bottoms = bottoms + vals
    for i, fy in enumerate(fys):
        # White bbox so horizontal gridlines pass behind the n-count text,
        # not through it (the 507 label otherwise sat on the y=500 gridline).
        ax.text(fy, bottoms[i] + max(bottoms) * 0.015,
                f"{int(bottoms[i]):,}", ha="center", va="bottom",
                fontsize=8, color="#333",
                bbox=dict(facecolor="white", edgecolor="none",
                          boxstyle="round,pad=0.15"))
    ax.set_xlabel("Fiscal year")
    ax.set_ylabel("Number of waivers")
    ax.set_title(title)
    ax.set_xticks(fys)
    ax.set_ylim(0, max(bottoms) * 1.08)
    # Bar charts: horizontal gridlines aid value reading; vertical ones
    # just cut through the n-count labels above each bar.
    ax.xaxis.grid(False)
    # Legend below the plot so it cannot collide with the data labels above
    # the bars regardless of fiscal year ordering or bar height.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              frameon=True, facecolor="white", framealpha=0.92,
              edgecolor="#cccccc", fontsize=9,
              ncol=min(len(series_order), 4))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_waiver_counts_by_agency() -> Path:
    w = _load_waivers_for_counts().with_columns(
        pl.when(pl.col("Contracting_Office_Agency_Name").is_in(TOP_AGENCIES_FULL))
        .then(pl.col("Contracting_Office_Agency_Name").replace(AGENCY_SHORT))
        .otherwise(pl.lit("Other"))
        .alias("agency_group")
    )
    pivot = (w.group_by(["fiscal_year", "agency_group"]).len().rename({"len": "n"})
             .pivot(on="agency_group", index="fiscal_year", values="n")
             .fill_null(0).sort("fiscal_year"))
    fys = pivot["fiscal_year"].to_list()
    colors = sns.color_palette("deep", n_colors=len(AGENCY_ORDER))
    # Push "Other" to a neutral gray.
    colors = list(colors)
    colors[AGENCY_ORDER.index("Other")] = (0.6, 0.6, 0.6)
    return _stacked_bar(
        pivot, fys, AGENCY_ORDER, colors,
        "Nonavailability Waivers by Fiscal Year and Contracting Agency",
        OUT / "F1_waiver_counts_by_fy_agency.png",
    )


def plot_waiver_counts_by_duration() -> Path:
    w = _load_waivers_for_counts().with_columns(
        pl.col("Expected_Maximum_Duration_of_the_Proposed_Waiver")
        .replace(DURATION_MAP).fill_null("Unknown").alias("duration_group")
    )
    pivot = (w.group_by(["fiscal_year", "duration_group"]).len().rename({"len": "n"})
             .pivot(on="duration_group", index="fiscal_year", values="n")
             .fill_null(0).sort("fiscal_year"))
    fys = pivot["fiscal_year"].to_list()
    colors = list(sns.color_palette("deep", n_colors=len(DURATION_ORDER)))
    colors[DURATION_ORDER.index("Unknown")] = (0.6, 0.6, 0.6)
    return _stacked_bar(
        pivot, fys, DURATION_ORDER, colors,
        "Nonavailability Waivers by Fiscal Year and Expected Duration",
        OUT / "F2_waiver_counts_by_fy_duration.png",
    )


def plot_waiver_counts_by_psc() -> Path:
    w = _load_waivers_for_counts().with_columns(
        pl.col("Product_Service_Code_(PSC)").str.slice(0, 4).alias("psc4")
    ).with_columns(
        pl.when(pl.col("psc4").is_in(list(PSC_LABELS.keys())))
        .then(pl.col("psc4").replace(PSC_LABELS))
        .otherwise(pl.lit("Other PSCs"))
        .alias("psc_group")
    )
    pivot = (w.group_by(["fiscal_year", "psc_group"]).len().rename({"len": "n"})
             .pivot(on="psc_group", index="fiscal_year", values="n")
             .fill_null(0).sort("fiscal_year"))
    fys = pivot["fiscal_year"].to_list()
    colors = list(sns.color_palette("deep", n_colors=len(PSC_ORDER)))
    colors[PSC_ORDER.index("Other PSCs")] = (0.6, 0.6, 0.6)
    return _stacked_bar(
        pivot, fys, PSC_ORDER, colors,
        "Nonavailability Waivers by Fiscal Year and Top PSC Codes",
        OUT / "F3_waiver_counts_by_fy_psc.png",
    )


# ---------------------------------------------------------------------------
# 7. Foreign-manufacture composition (appendix Figure A0)
# ---------------------------------------------------------------------------
# Companion to appendix Table A0. Plots nonavailability (place-of-manufacture
# code "J") as a share of all foreign-manufacture transactions (everything not
# domestic) by fiscal year. This is the trend the Procurement-in-Context
# discussion points to: nonavailability is a small but, since FY2019, sharply
# larger slice of foreign sourcing. Computed from the same
# _compute_foreign_share() that builds Table A0, so the figure and table can
# never disagree. Unlike the Data-section count figures (F1-F3), FY2026 is
# kept here but drawn as a hollow marker on a dashed segment, since it is an
# in-progress year (data through January 2026).

def plot_foreign_manufacture_share() -> Path:
    import textwrap

    from matplotlib.ticker import PercentFormatter

    fs = _compute_foreign_share().sort_values("fy").reset_index(drop=True)
    foreign = fs["n_nonavailability"] + fs["n_other_foreign"]
    fs["share_pct"] = fs["n_nonavailability"] / foreign * 100.0

    # Split complete fiscal years from the in-progress final year.
    complete = fs[fs["fy"] < 2026]
    partial = fs[fs["fy"] >= 2026]
    color = sns.color_palette("deep")[0]

    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=150)

    # Solid line + filled markers for complete fiscal years.
    ax.plot(complete["fy"], complete["share_pct"],
            color=color, linewidth=2.0, marker="o", markersize=6,
            label="Complete fiscal year")

    # Dashed connector into the partial year, with a hollow marker, mirroring
    # the open-circle "reference" styling used in the event-study plots.
    if not partial.empty:
        bridge = fs[fs["fy"] >= 2025]   # last complete year -> partial year
        ax.plot(bridge["fy"], bridge["share_pct"],
                color=color, linewidth=1.6, linestyle="--")
        ax.scatter(partial["fy"], partial["share_pct"],
                   facecolors="white", edgecolors=color, s=70, zorder=3,
                   label="FY2026 (partial, through Jan)")

    # Value labels above each point, white bbox so gridlines pass behind them
    # (matching the n-count labels on the F1-F3 bar charts).
    for _, r in fs.iterrows():
        ax.annotate(f"{r['share_pct']:.1f}%",
                    xy=(r["fy"], r["share_pct"]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color="#333",
                    bbox=dict(facecolor="white", edgecolor="none",
                              boxstyle="round,pad=0.15"))

    ax.set_xlabel("Fiscal year")
    ax.set_ylabel("Nonavailability share of foreign-manufacture transactions")
    ax.set_title("Nonavailability Share of Foreign-Manufacture Transactions by Fiscal Year")
    ax.set_xticks(fs["fy"].tolist())
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_ylim(0, fs["share_pct"].max() * 1.20)
    ax.legend(loc="best", frameon=True, facecolor="white", framealpha=0.92,
              edgecolor="#cccccc", fontsize=9)

    note = (
        'Nonavailability (place-of-manufacture code "J") as a share of all '
        "foreign-manufacture transactions (every place-of-manufacture code "
        "except domestic). FY2026 is in progress (through January 2026) and is "
        "shown as a hollow marker on a dashed segment. Source: FPDS."
    )
    # Wrap the note so it never renders wider than the plot. fig.text does not
    # wrap on its own, and with bbox_inches="tight" a single long line would
    # stretch the saved figure well past the axes.
    note = textwrap.fill(note, width=115)
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=8, color="gray")
    fig.tight_layout()
    path = OUT / "A0_foreign_manufacture_share.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 8. Country of origin of treated-NSN transactions (Figures F4b / F4c)
# ---------------------------------------------------------------------------
# F4b: origin composition by fiscal year, two 100%-stacked-bar panels (by
#      transaction count and by amount paid).
# F4c: per-NSN transaction timeline for the non-container NSNs, one dot per
#      transaction colored by origin, with each NSN's waiver date marked. Item
#      labels from DLA ITEM_NAME (FSC class for the two NSNs that appear only in
#      the supplemental transaction source).
#
# Broadcast filters (no silent rules):
#   - Combined record = DLA deduped + supplemental rows NOT already in DLA
#     (anti-join on nsn/piid/mod/line/action_date), matching the panel build.
#   - Treated set = NSNs with at least one transaction in event years [-5,-1]
#     (pre-waiver presence filter): 27 NSNs (21 containers + 6 items).
#     6695-01-266-2248 drops; its observations sit at event years -8, -7, 0.
#   - F4b covers complete fiscal years FY2017-FY2025; FY2026 is partial and
#     dropped (the record ends March 2026), as in F1-F3. Shares are over
#     transactions with a recorded country of origin.
#   - Amount paid = unit price x quantity ordered, with the obligated amount as
#     a fallback only where price/qty are unavailable.

TREATED_CSV = P.TREATMENT_DATES
DLA_PARQ_ORIGIN = P.DLA_ENRICHED_LATEST
BL_PARQ_ORIGIN = P.COMBINED_BIDLINK_PANEL

ORIGIN_FY_MIN, ORIGIN_FY_MAX = 2017, 2025

# Stacked bottom -> top: U.S. on the floor, China on the ceiling so the
# displacement reads as a crossover. Title Case throughout.
ORIGIN_ORDER = ["United States", "South Korea", "Turkey", "Other", "Not Reported", "China"]
_deep = sns.color_palette("deep")
ORIGIN_COLORS = {
    "United States": _deep[0],        # blue
    "South Korea":   _deep[1],        # orange
    "Turkey":        _deep[2],        # green
    "Other":         (0.60, 0.60, 0.60),
    "Not Reported":  (0.82, 0.82, 0.82),
    "China":         _deep[3],        # red -- the salient band
}

# Non-container NSN labels (DLA ITEM_NAME, cleaned; FSC class for the two that
# appear only in the supplemental transaction source).
NONCONTAINER_LABELS = {
    "1660016004909": "Aircraft environmental part (1660)",
    "1680016229189": "Aircraft accessory (1680)",
    "2895145543146": "Pneumatic motor (2895)",
    "3990015742050": "Material-handling roller (3990)",
    "4240015387970": "Hearing protector (4240)",
    "4510015272274": "Shower/bath fixture (4510)",
    "6695012662248": "Motional transducer (6695)",
}


def _origin_bucket(country_col: str):
    return (pl.when(pl.col(country_col) == "UNITED STATES").then(pl.lit("United States"))
              .when(pl.col(country_col) == "CHINA").then(pl.lit("China"))
              .when(pl.col(country_col) == "KOREA, SOUTH").then(pl.lit("South Korea"))
              .when(pl.col(country_col) == "TURKEY").then(pl.lit("Turkey"))
              .when(pl.col(country_col).is_null()).then(pl.lit("Not Reported"))
              .otherwise(pl.lit("Other")))


def _build_treated_combined() -> pl.DataFrame:
    """Combined, deduped transaction record for the treated NSNs with origin,
    obligation, fiscal year, and origin bucket. Mirrors the panel-build merge."""
    def fed_fy(col: str):
        return (pl.when(pl.col(col).dt.month() >= 10)
                  .then(pl.col(col).dt.year() + 1)
                  .otherwise(pl.col(col).dt.year()))

    treated = pl.read_csv(TREATED_CSV,
                          schema_overrides={"nsn": pl.Utf8, "first_waiver_date": pl.Utf8})
    tnsns = [n.replace("-", "") for n in treated["nsn"].to_list()]

    dla = pl.scan_parquet(DLA_PARQ_ORIGIN).filter(
        pl.col("NSN").is_in(tnsns) & pl.col("AWARD_DATE").is_not_null()
    ).select([
        pl.col("NSN").alias("nsn"),
        pl.col("BASE_PIID").alias("piid"),
        pl.when((pl.col("MOD_NUMBER").is_null()) | (pl.col("MOD_NUMBER").str.strip_chars() == ""))
          .then(pl.lit("BASE")).otherwise(pl.col("MOD_NUMBER").str.strip_chars()).alias("mod"),
        pl.col("PO_ITMNO").str.strip_chars().str.strip_chars_start("0").alias("line"),
        pl.col("AWARD_DATE").alias("action_date"),
        pl.col("country_of_product_or_service_origin").alias("country"),
        pl.col("federal_action_obligation").cast(pl.Float64).alias("obligation"),
        pl.col("NETPRICE").cast(pl.Float64).alias("unit_price"),
        pl.col("ORDER_QTY").cast(pl.Float64).alias("qty"),
    ]).collect().unique(subset=["nsn", "piid", "mod", "line", "action_date"], keep="first")

    bl = pl.scan_parquet(BL_PARQ_ORIGIN).filter(
        pl.col("bl_nsn").is_in(tnsns) & pl.col("bl_date_parsed").is_not_null()
    ).select([
        pl.col("bl_nsn").alias("nsn"),
        pl.col("bl_contract").alias("piid"),
        pl.when((pl.col("bl_order").is_null()) | (pl.col("bl_order").str.strip_chars() == ""))
          .then(pl.lit("BASE")).otherwise(pl.col("bl_order").str.strip_chars()).alias("mod"),
        pl.col("bl_line").str.strip_chars().str.strip_chars_start("0").alias("line"),
        pl.col("bl_date_parsed").alias("action_date"),
        pl.col("proc_country_of_product_or_service_origin").alias("country"),
        pl.col("proc_federal_action_obligation").cast(pl.Float64).alias("obligation"),
        pl.col("bl_price").cast(pl.Float64).alias("unit_price"),
        pl.col("bl_qty").cast(pl.Float64).alias("qty"),
    ]).collect().unique(subset=["nsn", "piid", "mod", "line", "action_date"], keep="first")

    bl_only = bl.join(dla.select(["nsn", "piid", "mod", "line", "action_date"]),
                      on=["nsn", "piid", "mod", "line", "action_date"], how="anti")
    comb = pl.concat([dla, bl_only]).with_columns(
        fed_fy("action_date").alias("fy"),
        _origin_bucket("country").alias("origin"),
        pl.col("obligation").fill_null(0.0),
        pl.col("nsn").str.slice(0, 4).alias("fsc"),
        (pl.col("unit_price").fill_null(0.0) * pl.col("qty").fill_null(0.0)).alias("purchase_value"),
    ).with_columns(
        # Amount paid for the product = unit price x quantity ordered, which has
        # near-complete coverage. Fall back to the obligated amount only where
        # price/qty are unavailable: supplemental rows that never matched an FPDS
        # obligation carry a price (obligation is $0), and a few sole-source repair
        # lines (e.g. the heat exchanger) carry an obligation but a $0 scrape price.
        pl.when(pl.col("purchase_value") > 0).then(pl.col("purchase_value"))
          .otherwise(pl.col("obligation")).alias("amount_paid"),
    )

    # Pre-waiver presence filter: keep NSNs with >=1 transaction in event years
    # [-5,-1]. Day-precise event_year = floor((action - waiver)/365.25), matching
    # the panel build. Yields the 27-NSN treated set used in the data section.
    wdates = dict(zip(
        [n.replace("-", "") for n in treated["nsn"].to_list()],
        treated["first_waiver_date"].str.to_date().to_list(),
    ))
    comb = comb.with_columns(
        pl.col("nsn").replace_strict(wdates, default=None).alias("_wdate"))
    comb = comb.with_columns(
        (((pl.col("action_date") - pl.col("_wdate")).dt.total_days() / 365.25)
         .floor().cast(pl.Int32)).alias("_ey"))
    keep = (comb.filter(pl.col("_ey").is_between(-5, -1))
                .select("nsn").unique().to_series().to_list())
    comb = comb.filter(pl.col("nsn").is_in(keep)).drop(["_wdate", "_ey"])
    print(f"  origin figures: {len(keep)} treated NSNs after [-5,-1] presence filter")
    return comb


def _origin_matrix(tab: pl.DataFrame, share_col: str, fys: list[int]) -> dict:
    out = {}
    for origin in ORIGIN_ORDER:
        sub = tab.filter(pl.col("origin") == origin)
        m = {r["fy"]: r[share_col] for r in sub.iter_rows(named=True)}
        out[origin] = [m.get(fy, 0.0) for fy in fys]
    return out


def _money(v: float) -> str:
    if v >= 1e6:
        return f"${v / 1e6:.0f}M"
    if v >= 1e3:
        return f"${v / 1e3:.0f}k"
    return f"${v:.0f}"


def _present_origins(tab: pl.DataFrame, value_col: str = "n") -> list[str]:
    """ORIGIN_ORDER restricted to buckets that actually occur (nonzero total),
    so the legend never lists a country that has no bar."""
    totals = tab.group_by("origin").agg(pl.col(value_col).sum().alias("t"))
    nonzero = {r["origin"] for r in totals.iter_rows(named=True) if (r["t"] or 0) > 0}
    return [o for o in ORIGIN_ORDER if o in nonzero]


def plot_treatment_origin_by_fy() -> Path:
    from matplotlib.ticker import PercentFormatter

    comb = _build_treated_combined().filter(
        pl.col("fy").is_between(ORIGIN_FY_MIN, ORIGIN_FY_MAX)
        # Shares are taken over transactions with a recorded country of origin.
        & (pl.col("origin") != "Not Reported"))

    agg = comb.group_by(["fy", "origin"]).agg(
        pl.len().alias("n"), pl.col("amount_paid").sum().alias("paid"))
    tot = comb.group_by("fy").agg(
        pl.len().alias("fy_n"), pl.col("amount_paid").sum().alias("fy_paid"))
    tab = (agg.join(tot, on="fy")
              .with_columns((pl.col("n") / pl.col("fy_n")).alias("share_count"),
                            (pl.col("paid") / pl.col("fy_paid")).alias("share_value"))
              .sort(["fy", "origin"]))
    # Round before writing: the grouped dollar sums carry last-ulp wobble from
    # parallel summation order, which reshuffled this committed sidecar's bytes
    # run-to-run. Cents for dollar columns, 12 places for shares — both far
    # below any meaningful precision. Affects only the written CSV.
    tab.with_columns(
        pl.col("paid").round(2), pl.col("fy_paid").round(2),
        pl.col("share_count").round(12), pl.col("share_value").round(12),
    ).write_csv(OUT / "F4b_treatment_origin_by_fy.csv")

    fys = list(range(ORIGIN_FY_MIN, ORIGIN_FY_MAX + 1))
    present = _present_origins(tab, "n")
    cnt = _origin_matrix(tab, "share_count", fys)
    dol = _origin_matrix(tab, "share_value", fys)
    fy_n = {r["fy"]: r["fy_n"] for r in tot.iter_rows(named=True)}
    fy_obl = {r["fy"]: r["fy_paid"] for r in tot.iter_rows(named=True)}

    # 100%-stacked bars per fiscal year.
    x = np.arange(len(fys))
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.8), dpi=150, sharex=True, sharey=True)
    panels = [
        (axes[0], cnt, "A. Share of Procurement Actions",
         [f"{int(fy_n[fy]):,}" for fy in fys]),
        (axes[1], dol, "B. Share of Amount Paid",
         [_money(fy_obl[fy]) for fy in fys]),
    ]
    for ax, mat, title, toplab in panels:
        bottom = np.zeros(len(fys))
        for o in present:
            vals = np.array(mat[o])
            ax.bar(x, vals, 0.82, bottom=bottom, label=o, color=ORIGIN_COLORS[o],
                   edgecolor="white", linewidth=0.5)
            bottom = bottom + vals
        # Per-year magnitude above each bar (transactions in A, obligation in B).
        for i in range(len(fys)):
            ax.text(x[i], 1.015, toplab[i], ha="center", va="bottom", fontsize=7.5,
                    color="#333", bbox=dict(facecolor="white", edgecolor="none",
                                            boxstyle="round,pad=0.12"))
        ax.set_xlim(-0.6, len(fys) - 0.4)
        ax.set_ylim(0, 1.10)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_xticks(x)
        ax.set_xticklabels([f"FY{y}" for y in fys], rotation=45, ha="right")
        ax.set_xlabel("Fiscal year")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Share of transactions")

    fig.suptitle("Country of Origin by Fiscal Year, Treated National Stock Numbers",
                 fontsize=12.5)
    # Legend below the plot to keep the right margin clear.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], title="Country of Origin",
               loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=len(present),
               frameon=True, facecolor="white", framealpha=0.92,
               edgecolor="#cccccc", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    path = OUT / "F4b_treatment_origin_by_fy.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_treatment_origin_noncontainer() -> Path:
    import matplotlib.dates as mdates
    from datetime import date
    from matplotlib.lines import Line2D

    # Sort to a total order so the figure is deterministic: _build_treated_combined
    # returns rows in a non-deterministic (group_by/join) order, and the per-point
    # jitter below is drawn from a seeded RNG in row order, so an unsorted frame
    # would reshuffle the dots run-to-run.
    comb = (_build_treated_combined().filter(~pl.col("fsc").str.starts_with("8150"))
            .sort(["nsn", "action_date", "origin", "country"]))
    comb.select(["nsn", "action_date", "country", "origin"]).write_csv(
        OUT / "F4c_treatment_origin_noncontainer.csv")

    # Waiver date per NSN (drawn as a per-row marker).
    treated = pl.read_csv(TREATED_CSV,
                          schema_overrides={"nsn": pl.Utf8, "first_waiver_date": pl.Utf8})
    wdate = {r["nsn"].replace("-", ""): r["first_waiver_date"]
             for r in treated.iter_rows(named=True)}

    china_share, nsn_n = {}, {}
    for r in comb.group_by("nsn").agg(
            (pl.col("origin") == "China").mean().alias("cs"),
            pl.len().alias("n")).iter_rows(named=True):
        china_share[r["nsn"]] = r["cs"]
        nsn_n[r["nsn"]] = r["n"]
    # Sort rows by China share; break ties on the NSN id so the row order is
    # deterministic (polars .unique() does not guarantee a stable order, and
    # several NSNs share a China share of 0 or 1).
    nsns = sorted(comb["nsn"].unique().to_list(), key=lambda n: (china_share.get(n, 0.0), n))
    row = {n: i for i, n in enumerate(nsns)}
    present = [o for o in ORIGIN_ORDER if o in set(comb["origin"].to_list())]

    # One dot per transaction, placed at its date and colored by origin; a small
    # vertical jitter keeps same-date dots from fully overlapping.
    nrow = len(nsns)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=150)
    ax.set_axisbelow(True)
    # Alternating lane shading + a faint center guide so each NSN's dots read as
    # one horizontal track rather than a cloud of points.
    for i in range(nrow):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#f2f2f2", zorder=0)
        ax.axhline(i, color="#d8d8d8", lw=0.6, zorder=1)
    for o in present:
        sub = comb.filter(pl.col("origin") == o)
        xs = [mdates.date2num(d) for d in sub["action_date"].to_list()]
        ys = [row[n] + rng.uniform(-0.15, 0.15) for n in sub["nsn"].to_list()]
        ax.scatter(xs, ys, s=60, color=ORIGIN_COLORS[o], edgecolor="white",
                   linewidth=0.5, alpha=0.9, label=o, zorder=4)
    for n in nsns:
        wd = wdate.get(n)
        if wd:
            xn = mdates.date2num(date.fromisoformat(wd))
            ax.plot([xn, xn], [row[n] - 0.46, row[n] + 0.46], color="black",
                    lw=1.4, ls=(0, (2, 1.2)), zorder=5)

    ax.set_yticks(range(nrow))
    ax.set_yticklabels([f"{NONCONTAINER_LABELS.get(n, n)}  (n={nsn_n[n]})" for n in nsns])
    ax.set_ylim(-0.5, nrow - 0.5)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("Transaction date")
    ax.grid(axis="x", color="#e6e6e6", lw=0.6)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.02)
    ax.set_title("Transaction Timeline by Country of Origin, Non-Container Treated NSNs",
                 fontsize=12)

    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=7,
                      markerfacecolor=ORIGIN_COLORS[o], markeredgecolor="white", label=o)
               for o in present]
    handles.append(Line2D([0], [0], color="black", lw=1.3, ls=(0, (2, 1)),
                          label="Waiver issued"))
    fig.legend(handles=handles, title="Country of Origin", ncol=len(handles),
               loc="lower center", bbox_to_anchor=(0.5, -0.04), frameon=True,
               facecolor="white", framealpha=0.92, edgecolor="#cccccc", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    path = OUT / "F4c_treatment_origin_noncontainer.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def main() -> None:
    paths: list[Path] = []
    for outcome, spec in ES_COEFS.items():
        paths.append(plot_event_study(outcome, spec))
    for outcome in OUTCOME_LABEL:
        paths.append(plot_nn_did_dotplot(outcome, "headline"))
        paths.append(plot_nn_did_dotplot(outcome, "ey2plus"))
    for outcome in OUTCOME_LABEL:
        paths.append(plot_synth_forest(outcome))
    for outcome in OUTCOME_LABEL:
        paths.append(plot_synth_event_time(outcome))
    for outcome in OUTCOME_LABEL:
        paths.append(plot_event_study_ladder(outcome))

    # Motivation: where the waivered items are made.
    paths.append(plot_treatment_origin_by_fy())
    paths.append(plot_treatment_origin_noncontainer())

    # Data section: waiver universe descriptives.
    paths.append(plot_waiver_counts_by_agency())
    paths.append(plot_waiver_counts_by_duration())
    paths.append(plot_waiver_counts_by_psc())

    # Appendix Figure A0: companion to Table A0.
    paths.append(plot_foreign_manufacture_share())

    print(f"Wrote {len(paths)} figures to {OUT}")
    for p in paths:
        print("  ", p.relative_to(REPO))


if __name__ == "__main__":
    main()
