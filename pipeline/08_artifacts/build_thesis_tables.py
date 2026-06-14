"""Generate HTML result tables for the thesis Analysis section.

One .html file per table. The
intended workflow: open the HTML file in a browser, select the table,
copy, paste into Google Docs. Google Docs preserves the table structure
on paste and the cells become editable.

Re-run with:
    uv run python pipeline/08_artifacts/build_thesis_tables.py
"""
from __future__ import annotations

from pathlib import Path

import sys

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import json
import paths as P
import es_stats

REPO = P.REPO_ROOT
OUT = P.DESCRIPTIVES_TABLES
OUT.mkdir(parents=True, exist_ok=True)

# The headline event-study numbers (3-spec ladder, per-outcome summary) are read
# from the committed main event_study.json, written by
# pipeline/07_robustness/run_event_study_refit.py --variant main from the same
# fits the body regression tables use. _ES_ORDER fixes the row order (Data->Analysis).
_ES_ORDER = ["domestic_share", "max_log_unit_price", "mean_offers"]
_ES_JSON = json.loads(P.event_study_json("main").read_text(encoding="utf-8"))

# Data-section source files
WAIVERS_CSV       = P.WAIVERS_CLEANED
PROCUREMENT_PQ    = P.PROCUREMENT_PARQUET
WAIVER_NSN_REF    = P.NSN_REFERENCE / "waiver_nsn_reference.csv"
WAIVER_DOD_IDS    = P.NSN_REFERENCE / "waiver_dod_identifiers_manual.csv"
NSN_UNIVERSE_CSV  = P.panel_logs("main") / "nsn_universe.csv"
STAGE2_LOG        = P.matched_logs("main") / "stage2_per_treated_log.csv"
BIDLINK_NSN_DIR   = P.RAW_BIDLINK / "nsn" / "procurement_history_line_items"

# Treatment-summary (Table A1) source files
FIRST_WAIVER_CSV  = P.TREATMENT_DATES
PANEL_ENRICHED    = P.PANEL_ENRICHED
COMBINED_PANEL    = P.COMBINED_BIDLINK_PANEL


# ---------------------------------------------------------------------------
# Styling — kept inline so each HTML file is self-contained and renders
# the same when opened by itself.
# ---------------------------------------------------------------------------
CSS = """
<style>
  /* Cap body width so tables don't balloon on wide monitors. A long note
     in a tfoot colspan cell would otherwise pull the table out to fill the
     viewport's max-content. Centered for readability when rendered alone. */
  body { font-family: "Times New Roman", Times, serif; margin: 24px auto;
         max-width: 900px; color: #222; }
  table.thesis { border-collapse: collapse; margin: 16px 0; font-size: 14px;
                 font-family: "Times New Roman", Times, serif;
                 max-width: 100%; }
  table.thesis caption { caption-side: top; text-align: left; font-weight: 600; padding: 6px 0; }
  table.thesis th, table.thesis td { padding: 6px 12px; text-align: left; }
  /* Centered group headers (column-spanning cells). Without this, a colspan
     `class="num"` th aligns right and looks bizarre over its child columns. */
  table.thesis th[colspan] { text-align: center; }
  table.thesis thead th { border-bottom: 2px solid #444; }
  table.thesis tbody td { border-bottom: 1px solid #e0e0e0; }
  table.thesis tbody tr:last-child td { border-bottom: 2px solid #444; }
  table.thesis td.num, table.thesis th.num { text-align: right; font-variant-numeric: tabular-nums; }
  /* Note lives in <tfoot> so its width naturally matches the table. Suppress
     the default cell borders that would otherwise put a thin rule above it. */
  table.thesis tfoot td.note { font-size: 12px; color: #555; padding-top: 8px;
                               text-align: left; border-bottom: none !important;
                               font-family: "Times New Roman", Times, serif; }
  table.thesis tfoot tr td.note { border-top: none; }
</style>
""".strip()


def _note_tfoot(n_cols: int, note_html: str) -> str:
    """Wrap a note in a <tfoot> row spanning all columns. Keeps the note's
    width tied to the table's width regardless of viewport size."""
    return (f'  <tfoot>\n'
            f'    <tr><td colspan="{n_cols}" class="note">{note_html}</td></tr>\n'
            f'  </tfoot>\n')


def t_to_p(t: float, df: int) -> float:
    return 2.0 * (1.0 - stats.t.cdf(abs(t), df=df))


def wrap_html(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  {CSS}
</head>
<body>
{body}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Table 1 — Event-study headline summary (one row per outcome)
# ---------------------------------------------------------------------------
# Column (3) headline (NSN+FY FE, no extent controls), from the main
# event_study.json (see loader at the top of this module).
EVENT_STUDY_SUMMARY = [_ES_JSON[o]["summary_row"] for o in _ES_ORDER]


def build_event_study_table(*, summary: list | None = None,
                            out_dir: Path = OUT) -> Path:
    rows = "\n".join(
        f"""    <tr>
      <td>{r['outcome']}</td>
      <td class="num">{r['avg_post_beta']:+.3f}</td>
      <td class="num">{r['avg_post_se']:.3f}</td>
      <td class="num">{r['pretrend_p']}</td>
      <td class="num">{r['n_obs']:,}</td>
      <td class="num">{r['n_clusters']:,}</td>
    </tr>"""
        for r in (summary if summary is not None else EVENT_STUDY_SUMMARY)
    )
    table = f"""<table class="thesis">
  <caption>Table 2. Event Study Results by Outcome</caption>
  <thead>
    <tr>
      <th>Outcome</th>
      <th class="num">Avg post-period β</th>
      <th class="num">SE</th>
      <th class="num">Joint pre-trend p</th>
      <th class="num">N obs</th>
      <th class="num">N NSN clusters</th>
    </tr>
  </thead>
  <tbody>
{rows}
  </tbody>
{_note_tfoot(6,
    'Avg post-period β is the linear combination of event-year coefficients at '
    'event year ≥ 0. Joint pre-trend p is the F-test for the pre-period '
    'event-year dummies being jointly zero. Specification: NSN and FY fixed '
    'effects, CRV1 standard errors clustered on NSN. Full event-time coefficient '
    'paths are in Figures F6-F8.'
)}</table>"""
    html = wrap_html("Table 2 – Event Study Summary", table)
    path = out_dir / "T2_event_study_summary.html"
    path.write_text(html, encoding="utf-8", newline="\n")
    return path


# ---------------------------------------------------------------------------
# Table 2 — Matched Controls: ATT per outcome, both windows, both variants
# ---------------------------------------------------------------------------
# Two matching variants are reported:
#   * "Main" (FSG-off in the pipeline): Mahalanobis picks the 3 nearest donors
#     from the full eligible pool, regardless of product group.
#   * "Same-FSG" (FSG-on in the pipeline): adds an exact-match constraint
#     requiring each donor to share the treated NSN's two-digit Federal
#     Supply Group (the first two digits of the four-digit FSC code).
#     All 26 treated NSNs still match successfully under this constraint.
NN_DID_SUMMARY_OFF = P.matched_tables("main") / "nn_did_summary_fsg_off.csv"
NN_DID_SUMMARY_ON  = P.matched_tables("main") / "nn_did_summary_fsg_on.csv"

OUTCOME_LABEL = {
    "domestic_share": "Domestic sourcing share",
    "max_log_unit_price": "Maximum logged unit price",
    "mean_offers": "Mean offers",
}

OUTCOME_ORDER = ["domestic_share", "max_log_unit_price", "mean_offers"]


def _stars(p: float) -> str:
    """Significance stars: *** p < 0.01, ** p < 0.05, * p < 0.10."""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def _att_cell(att: float, se: float, stars: str) -> str:
    """Combined ATT (SE) cell with significance stars on the estimate."""
    return (f'{att:+.3f}{stars}<br>'
            f'<span style="color:#888;font-size:11px">({se:.3f})</span>')


def build_matched_table(*, summary_off: Path = NN_DID_SUMMARY_OFF,
                        summary_on: Path = NN_DID_SUMMARY_ON,
                        out_dir: Path = OUT) -> Path:
    def _load(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["pval"] = df.apply(
            lambda r: t_to_p(r["t_stat"], int(r["n_treated"]) - 1), axis=1)
        return df

    df_off = _load(summary_off)
    df_on  = _load(summary_on)
    window_label = {"headline": "event year ≥ 0", "ey2plus": "event year ≥ 2"}

    rows: list[str] = []
    for outcome in OUTCOME_ORDER:
        for post_label in ["headline", "ey2plus"]:
            r_off = df_off[(df_off["outcome"] == outcome)
                           & (df_off["post_label"] == post_label)].iloc[0]
            r_on = df_on[(df_on["outcome"] == outcome)
                         & (df_on["post_label"] == post_label)].iloc[0]
            label = OUTCOME_LABEL[outcome] if post_label == "headline" else ""
            sign_disagrees = (r_off["att"] * r_on["att"] < 0)
            row_style = (' style="background:#fff7e0"' if sign_disagrees else "")
            rows.append(f"""    <tr{row_style}>
      <td>{label}</td>
      <td>{window_label[post_label]}</td>
      <td class="num">{int(r_off['n_treated'])}</td>
      <td class="num">{_att_cell(r_off['att'], r_off['se'], _stars(r_off['pval']))}</td>
      <td class="num">{_att_cell(r_on['att'], r_on['se'], _stars(r_on['pval']))}</td>
    </tr>""")

    table = f"""<table class="thesis">
  <caption>Table 2. Matched Controls: ATT by Outcome and Post-Period Window, Main Matching vs. Same-FSG Sensitivity</caption>
  <thead>
    <tr>
      <th rowspan="2">Outcome</th>
      <th rowspan="2">Post window</th>
      <th rowspan="2" class="num">N</th>
      <th class="num">Main matching</th>
      <th class="num">Same-FSG sensitivity</th>
    </tr>
    <tr>
      <th class="num"><span style="font-weight:normal;color:#555">ATT (SE)</span></th>
      <th class="num"><span style="font-weight:normal;color:#555">ATT (SE)</span></th>
    </tr>
  </thead>
  <tbody>
{chr(10).join(rows)}
  </tbody>
{_note_tfoot(5,
    'ATT is the mean of the per-NSN τᵢ, with HC1 (heteroskedasticity-robust) '
    'standard errors. '
    '<i>Main matching</i> picks the three nearest donors by Mahalanobis distance '
    'from the full eligible pool. <i>Same-FSG sensitivity</i> adds an exact-match '
    "constraint requiring each donor to share the treated NSN's two-digit Federal "
    'Supply Group (the first two digits of the four-digit FSC code); all 26 '
    'treated NSNs still match under this rule. '
    'Significance: *** p &lt; 0.01, ** p &lt; 0.05, * p &lt; 0.10.'
)}</table>"""
    html = wrap_html("Table 2 – Matched Controls Summary", table)
    path = out_dir / "T2_matched_controls.html"
    path.write_text(html, encoding="utf-8", newline="\n")
    return path


# ---------------------------------------------------------------------------
# Table 3 — Synthetic Controls: distribution summary per outcome
# ---------------------------------------------------------------------------
SYNTH_SUMMARY = P.synth_tables("main") / "synth_summary.csv"


def build_synth_table(*, summary_path: Path = SYNTH_SUMMARY,
                      out_dir: Path = OUT) -> Path:
    df = pd.read_csv(summary_path).set_index("outcome")

    rows: list[str] = []
    for outcome in OUTCOME_ORDER:
        r = df.loc[outcome]
        # Active donor count vs pool size — how concentrated the weights are.
        active = int(r["median_n_active"])
        pool = int(r["median_balanced_pool"])
        rows.append(f"""    <tr>
      <td>{OUTCOME_LABEL[outcome]}</td>
      <td class="num">{int(r['n_fits'])}</td>
      <td class="num">{r['median_att']:+.3f}</td>
      <td class="num">{r['mean_att']:+.3f}</td>
      <td class="num">{r['sd_att']:.3f}</td>
      <td class="num">{int(r['n_empirical_p_lt_05'])} / {int(r['n_fits'])}</td>
      <td class="num">{r['median_pre_rmspe']:.4f}</td>
      <td class="num">{active:,} / {pool:,}</td>
    </tr>""")

    table = f"""<table class="thesis">
  <caption>Table 3. Synthetic Controls: Per-NSN ATT Distribution and Fit Diagnostics by Outcome</caption>
  <thead>
    <tr>
      <th>Outcome</th>
      <th class="num">N fits</th>
      <th class="num">Median ATT</th>
      <th class="num">Mean ATT</th>
      <th class="num">SD across NSNs</th>
      <th class="num">Significant (p &lt; 0.05)</th>
      <th class="num">Median pre-fit RMSPE</th>
      <th class="num">Median active donors / pool size</th>
    </tr>
  </thead>
  <tbody>
{chr(10).join(rows)}
  </tbody>
{_note_tfoot(8,
    'ATTs are estimated per treated NSN using <code>synthdid</code> '
    '(Arkhangelsky et al. 2021). Significance is assessed against the '
    'random-subsample placebo distribution (200 placebos per fit). Pre-fit '
    'RMSPE is the root mean squared prediction error in the pre-waiver '
    'period — smaller means the synthetic tracks its treated NSN more closely '
    'before the waiver. The "active donors / pool size" column reports the '
    'median count of non-treated NSNs that receive non-trivial weight '
    '(w &gt; 1 / (2·pool size)) against the total eligible pool — lower '
    'ratios indicate that the synthetic relies on a small subset of the pool.'
)}</table>"""
    html = wrap_html("Table 3 – Synthetic Controls Summary", table)
    path = out_dir / "T3_synth_controls.html"
    path.write_text(html, encoding="utf-8", newline="\n")
    return path


# ---------------------------------------------------------------------------
# Table 4 — Synthetic Controls: Aggregate event-time path
# ---------------------------------------------------------------------------
# Reports the mean effect across treated NSNs at each post-waiver event year,
# with the t-distribution 95% CI and p-value (df = n - 1 at each EY).
EFFECT_PATH_DIR = P.synth_tables("main") / "effect_path"


def build_synth_event_time_table(*, effect_path_dir: Path = EFFECT_PATH_DIR,
                                 out_dir: Path = OUT) -> Path:
    from scipy import stats as _stats
    rows: list[str] = []
    for outcome in OUTCOME_ORDER:
        # Pool per-NSN event-time effects across NSNs for this outcome.
        pieces = []
        for csv_path in sorted(effect_path_dir.glob(f"*__{outcome}.csv")):
            df = pd.read_csv(csv_path)
            df["nsn"] = csv_path.stem.split("__")[0]
            pieces.append(df[["event_year", "effect_demeaned", "nsn"]])
        long = pd.concat(pieces, ignore_index=True)
        post = long[long["event_year"] >= 0].sort_values("event_year")

        for ey, g in post.groupby("event_year"):
            x = g["effect_demeaned"].dropna().values
            n = len(x)
            if n < 2:
                continue
            m = float(x.mean())
            se = float(x.std(ddof=1)) / np.sqrt(n)
            p_val = 2 * (1 - _stats.t.cdf(abs(m / se), df=n - 1)) if se > 0 else float("nan")
            label = OUTCOME_LABEL[outcome] if ey == 0 else ""
            ey_int = int(ey)
            rows.append(f"""    <tr>
      <td>{label}</td>
      <td class="num">{ey_int:+d}</td>
      <td class="num">{n}</td>
      <td class="num">{m:+.3f}{_stars(p_val)}</td>
      <td class="num">{se:.3f}</td>
    </tr>""")

    table = f"""<table class="thesis">
  <caption>Table 4. Synthetic Controls: Aggregate Event-Time Path by Outcome</caption>
  <thead>
    <tr>
      <th>Outcome</th>
      <th class="num">Event year</th>
      <th class="num">N NSNs</th>
      <th class="num">Mean effect</th>
      <th class="num">SE</th>
    </tr>
  </thead>
  <tbody>
{chr(10).join(rows)}
  </tbody>
{_note_tfoot(5,
    'Aggregate effect at each post-waiver event year is the mean of per-NSN '
    'demeaned effects (treated minus synthetic, with the pre-waiver mean '
    'subtracted to remove any constant fit gap). SE and significance stars '
    'are from a one-sample t-test on the per-NSN effects at that event year '
    '(df = N − 1). Pre-waiver event years are flat by construction and not shown. '
    'Significance: *** p &lt; 0.01, ** p &lt; 0.05, * p &lt; 0.10.'
)}</table>"""
    html = wrap_html("Table 4 – Synth Event-Time Aggregates", table)
    path = out_dir / "T4_synth_event_time.html"
    path.write_text(html, encoding="utf-8", newline="\n")
    return path


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Appendix table — per-outcome 3-spec ladder for the event study.
# ---------------------------------------------------------------------------
# Specs 1-3 (OLS / +NSN FE / +NSN+FY FE), read from the main event_study.json
# (see loader at the top of this module). Spec 4 (extent controls) is not included.
LADDER_DATA = {o: es_stats.ladder_from_json(_ES_JSON[o]["ladder"]) for o in _ES_ORDER}

SPEC_LABELS = ["(1) OLS", "(2) +NSN FE", "(3) +NSN+FY FE"]
EVENT_YEARS_LADDER = [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3]


def _fmt_coef(coef: float, se: float, stars: str) -> str:
    return f'{coef:+.3f}{stars}<br><span style="color:#888;font-size:11px">({se:.3f})</span>'


def build_appendix_ladder(outcome: str, *,
                          ladder_data: dict | None = None,
                          event_years: list[int] | None = None,
                          out_dir: Path = OUT) -> Path:
    data = (ladder_data if ladder_data is not None else LADDER_DATA)[outcome]
    rows: list[str] = []
    for k in (event_years if event_years is not None else EVENT_YEARS_LADDER):
        if k == -1:
            rows.append('    <tr><td><i>k = -1</i></td>'
                        '<td class="num" colspan="3"><i>reference</i></td></tr>')
            continue
        cells = [_fmt_coef(*data["coefs"][k][i]) for i in range(3)]
        rows.append(
            f'    <tr><td><i>k = {k:+d}</i></td>'
            + "".join(f'<td class="num">{c}</td>' for c in cells)
            + "</tr>"
        )

    # Footer rows: NSN FE / FY FE indicators, pre-trend p, avg post, N, R²
    footer = (
        '    <tr style="border-top:1px solid #888"><td>NSN FE</td>'
        '<td class="num">—</td><td class="num">✓</td><td class="num">✓</td></tr>\n'
        '    <tr><td>FY FE</td>'
        '<td class="num">—</td><td class="num">—</td><td class="num">✓</td></tr>\n'
        '    <tr style="border-top:1px solid #888"><td>Joint pre-trend p</td>'
        + "".join(f'<td class="num">{p}{s}</td>' for p, s in data["pretrend_p"])
        + "</tr>\n"
        '    <tr><td>Avg post β (k ≥ 0)</td>'
        + "".join(f'<td class="num">{_fmt_coef(c, se, s)}</td>'
                  for c, se, s in data["avg_post"])
        + "</tr>\n"
        '    <tr style="border-top:1px solid #888"><td>N obs</td>'
        + "".join(f'<td class="num">{n:,}</td>' for n in data["n_obs"])
        + "</tr>\n"
        '    <tr><td>R²</td>'
        + "".join(f'<td class="num">{r:.3f}</td>' for r in data["r2"])
        + "</tr>"
    )

    table = f"""<table class="thesis">
  <caption>Appendix Table A. Event Study Specification Ladder: {data["title"]}</caption>
  <thead>
    <tr>
      <th>Event year</th>
      <th class="num">{SPEC_LABELS[0]}</th>
      <th class="num">{SPEC_LABELS[1]}</th>
      <th class="num">{SPEC_LABELS[2]}</th>
    </tr>
  </thead>
  <tbody>
{chr(10).join(rows)}
{footer}
  </tbody>
{_note_tfoot(4,
    'Coefficients on event-year dummies. Standard errors in parentheses, '
    'clustered on NSN. Reference event year is k = -1. '
    'Joint pre-trend p is the Wald F-test that all pre-treatment dummies are '
    'zero. Avg post β is the linear combination of post-period coefficients '
    '(k ≥ 0) with its CRV1 standard error. Specification (3), with NSN and FY '
    'fixed effects, is the one reported in the main Analysis section. '
    'Significance: *** p &lt; 0.01, ** p &lt; 0.05, * p &lt; 0.10.'
)}</table>"""
    html = wrap_html(f"Appendix Table A – {data['title']} (full ladder)", table)
    path = out_dir / f"appendix_event_study_ladder_{outcome}.html"
    path.write_text(html, encoding="utf-8", newline="\n")
    return path


# ---------------------------------------------------------------------------
# Table 6 — Foreign procurement share by fiscal year (Data section)
# ---------------------------------------------------------------------------
# Three categories per the thesis Data-section placeholder:
#   * Domestic           (place_of_manufacture_code == "D")
#   * Nonavailability    (place_of_manufacture_code == "J")
#   * Other foreign      (everything that is neither D nor J — i.e. L, G, E,
#                         K, H, F, A, B, I; excludes C which is dropped
#                         upstream as a non-manufactured-end-product action)
# Source: data/clean/procurement_data.parquet (full FPDS procurement extract).

def _compute_foreign_share() -> pd.DataFrame:
    df = (pl.scan_parquet(PROCUREMENT_PQ)
          .select(["action_date_fiscal_year",
                   "place_of_manufacture_code",
                   "federal_action_obligation"])
          .with_columns(
              pl.when(pl.col("place_of_manufacture_code") == "D")
              .then(pl.lit("domestic"))
              .when(pl.col("place_of_manufacture_code") == "J")
              .then(pl.lit("nonavailability"))
              .otherwise(pl.lit("other_foreign"))
              .alias("cat"))
          .filter(pl.col("action_date_fiscal_year").is_not_null())
          .group_by(["action_date_fiscal_year", "cat"])
          .agg([pl.len().alias("n"),
                pl.col("federal_action_obligation").cast(pl.Float64).sum().alias("dollars")])
          .collect(engine="streaming")
          .to_pandas())

    # Pivot to one row per FY, count + dollar columns side by side.
    counts = df.pivot(index="action_date_fiscal_year", columns="cat", values="n").fillna(0).astype(int)
    dollars = df.pivot(index="action_date_fiscal_year", columns="cat", values="dollars").fillna(0.0)
    for col in ["domestic", "nonavailability", "other_foreign"]:
        if col not in counts.columns:  counts[col]  = 0
        if col not in dollars.columns: dollars[col] = 0.0

    out = pd.DataFrame({"fy": counts.index.astype(int)})
    out["n_domestic"]        = counts["domestic"].values
    out["n_nonavailability"] = counts["nonavailability"].values
    out["n_other_foreign"]   = counts["other_foreign"].values
    out["n_total"]           = out[["n_domestic", "n_nonavailability", "n_other_foreign"]].sum(axis=1)
    out["d_domestic"]        = dollars["domestic"].values
    out["d_nonavailability"] = dollars["nonavailability"].values
    out["d_other_foreign"]   = dollars["other_foreign"].values
    out["d_total"]           = out[["d_domestic", "d_nonavailability", "d_other_foreign"]].sum(axis=1)
    return out.sort_values("fy").reset_index(drop=True)


def _fmt_dollars_b(x: float) -> str:
    """Render dollar amounts in billions to 2 decimals."""
    return f"${x / 1e9:,.2f}B"


def build_foreign_share_table() -> Path:
    df = _compute_foreign_share()

    rows: list[str] = []
    for _, r in df.iterrows():
        nt = r["n_total"]
        dt = r["d_total"]
        pct_n_dom    = r["n_domestic"]        / nt * 100 if nt else 0.0
        pct_n_nonav  = r["n_nonavailability"] / nt * 100 if nt else 0.0
        pct_n_other  = r["n_other_foreign"]   / nt * 100 if nt else 0.0
        pct_d_dom    = r["d_domestic"]        / dt * 100 if dt else 0.0
        pct_d_nonav  = r["d_nonavailability"] / dt * 100 if dt else 0.0
        pct_d_other  = r["d_other_foreign"]   / dt * 100 if dt else 0.0
        rows.append(f"""    <tr>
      <td class="num">{int(r['fy'])}</td>
      <td class="num">{int(r['n_domestic']):,}<br><span style="color:#888;font-size:11px">({pct_n_dom:.1f}%)</span></td>
      <td class="num">{int(r['n_nonavailability']):,}<br><span style="color:#888;font-size:11px">({pct_n_nonav:.2f}%)</span></td>
      <td class="num">{int(r['n_other_foreign']):,}<br><span style="color:#888;font-size:11px">({pct_n_other:.2f}%)</span></td>
      <td class="num">{_fmt_dollars_b(r['d_domestic'])}<br><span style="color:#888;font-size:11px">({pct_d_dom:.1f}%)</span></td>
      <td class="num">{_fmt_dollars_b(r['d_nonavailability'])}<br><span style="color:#888;font-size:11px">({pct_d_nonav:.2f}%)</span></td>
      <td class="num">{_fmt_dollars_b(r['d_other_foreign'])}<br><span style="color:#888;font-size:11px">({pct_d_other:.2f}%)</span></td>
    </tr>""")

    table = f"""<table class="thesis">
  <caption>Table A0. Foreign Manufacture Share by Fiscal Year</caption>
  <thead>
    <tr>
      <th rowspan="2" class="num">FY</th>
      <th colspan="3" class="num">Transactions (count, share of FY)</th>
      <th colspan="3" class="num">Obligations (dollars, share of FY)</th>
    </tr>
    <tr>
      <th class="num">Domestic</th>
      <th class="num">Nonavailability</th>
      <th class="num">Other foreign</th>
      <th class="num">Domestic</th>
      <th class="num">Nonavailability</th>
      <th class="num">Other foreign</th>
    </tr>
  </thead>
  <tbody>
{chr(10).join(rows)}
  </tbody>
{_note_tfoot(7,
    'Source: FPDS. Categories are place-of-manufacture codes: '
    '<i>Domestic</i> = D; <i>Nonavailability</i> = J; '
    '<i>Other foreign</i> = L, G, E, K, H, F, A, B, I combined '
    '(qualifying-country, trade-agreements, use-outside-US, unreasonable-cost, '
    'commercial IT, resale, foreign-content, foreign-services, and '
    'public-interest determinations). Code C (non-manufactured end-product actions) '
    'is excluded upstream. Percent shares are within fiscal year. '
    'Dollar amounts are in billions of USD.'
)}</table>"""
    html = wrap_html("Table A0 – Foreign Manufacture Share by FY", table)
    path = OUT / "TA0_foreign_share.html"
    path.write_text(html, encoding="utf-8", newline="\n")
    return path


# ---------------------------------------------------------------------------
# Table A1 — Treated NSN summary (Data section / appendix)
# ---------------------------------------------------------------------------
# One row per treated NSN: product, first waiver date, transaction count, and
# the average unit price and average bids observed over the study window. The
# 27 NSNs are the analysis set: the 28th treated NSN (6695-01-266-2248, waived
# 2025-07-10) has zero pre-waiver event years and is excluded from all three
# estimators. Counts come from panel_enriched.parquet (sums to 2,210 over the
# 27). The ~2,200 figure quoted in the Data section refers to this table.

# NSN excluded from all estimators (no pre-waiver event years).
TREATMENT_DROP_NSN = "6695012662248"

# Cleaned item names. The FLIS item-name field is empty for two NSNs (taken
# from the waiver title instead) and truncated for several container variants;
# standardized here.
TREATMENT_ITEM_NAMES = {
    "8150015741682": "Container, freight, utility",          # FLIS empty; waiver title "general freight containers"
    "2895145543146": "Motor, pneumatic",                     # FLIS empty (French NATO NSN); from waiver title
    "8150015338677": "Quadcon, freight, specific purpose",   # FLIS truncated "QUADCON,FREIGHT SPE"
    "8150015338675": "Quadcon, freight, specific purpose",
    "8150014813740": "Bicon, freight, general purpose",
    "8150014813741": "Bicon, freight, general purpose",
    "8150015338673": "Tricon, freight, specific purpose",
    "8150015338674": "Tricon, freight, specific purpose",
    "1680016229189": "Enhanced ground proximity warning system",  # FLIS truncated "ENHANCED GROUND"
    "1660016004909": "Heat exchanger, air-to-air, aircraft",
    "3990015742050": "Roller, material handling",
    "4510015272274": "Shower bath fixture",
    "4240015387970": "Protector, hearing",
}
# Excluded from the unit-price analysis (too few single-NSN contracts); its
# avg unit price is shown as a dash rather than a misleading figure.
TREATMENT_NO_PRICE_NSN = {"1680016229189"}


def build_treatment_summary_table() -> Path:
    import math

    fw = (pl.read_csv(FIRST_WAIVER_CSV, infer_schema_length=0)
          .with_columns(pl.col("nsn").str.replace_all("-", "").alias("k")))

    pe_all = pl.scan_parquet(PANEL_ENRICHED)
    pe = (pe_all.group_by("nsn")
          .agg(pl.col("n_transactions").sum().alias("tx"))
          .collect().with_columns(pl.col("nsn").cast(pl.Utf8)))

    cb = pl.scan_parquet(COMBINED_PANEL).filter(pl.col("role") == "treatment")
    price = (cb.filter(pl.col("bl_price") > 0)
             .group_by("bl_nsn").agg(pl.col("bl_price").mean().alias("avg_unit_price"))
             .collect().with_columns(pl.col("bl_nsn").cast(pl.Utf8)))
    bids = (cb.filter(pl.col("proc_number_of_offers_received").is_not_null())
            .group_by("bl_nsn")
            .agg(pl.col("proc_number_of_offers_received").cast(pl.Float64).mean().alias("avg_bids"))
            .collect().with_columns(pl.col("bl_nsn").cast(pl.Utf8)))

    # Proxy unit price for the heat exchanger (bl_price == 0 on its line items):
    # mean of the per-cell prices the panel recovered as max_log_unit_price,
    # itself obligation / quantity for single-NSN contracts.
    p1660 = (pe_all.filter(pl.col("nsn").cast(pl.Utf8) == "1660016004909")
             .select("max_log_unit_price").collect()["max_log_unit_price"])
    vals = [math.exp(v) for v in p1660 if v is not None]
    proxy_1660 = sum(vals) / len(vals) if vals else None

    def _name(k: str) -> str:
        return TREATMENT_ITEM_NAMES.get(k, "Container, freight, utility")

    t = (fw.join(pe, left_on="k", right_on="nsn", how="inner")
         .join(price, left_on="k", right_on="bl_nsn", how="left")
         .join(bids, left_on="k", right_on="bl_nsn", how="left")
         .filter(pl.col("k") != TREATMENT_DROP_NSN))
    t = t.with_columns(
        pl.col("k").map_elements(_name, return_dtype=pl.Utf8).alias("item_name"),
        pl.when(pl.col("k") == "1660016004909").then(pl.lit(proxy_1660))
          .otherwise(pl.col("avg_unit_price")).alias("avg_unit_price"),
    ).sort(["tx", "k"], descending=[True, False])  # k breaks tx ties deterministically

    n = t.height
    total_tx = int(t["tx"].sum())
    avg_tx = total_tx / n
    avg_price = t["avg_unit_price"].mean()   # ignores the one null (1680)
    avg_bids = t["avg_bids"].mean()

    def _price(v, k) -> str:
        if k in TREATMENT_NO_PRICE_NSN or v is None:
            return "—"
        return f"${v:,.0f}"

    rows: list[str] = []
    for r in t.iter_rows(named=True):
        rows.append(f"""    <tr>
      <td>{r['nsn']}</td>
      <td>{r['item_name']}</td>
      <td class="num">{r['first_waiver_date']}</td>
      <td class="num">{int(r['tx']):,}</td>
      <td class="num">{_price(r['avg_unit_price'], r['k'])}</td>
      <td class="num">{r['avg_bids']:.1f}</td>
    </tr>""")
    rows.append(f"""    <tr style="border-top:2px solid #444;font-weight:600">
      <td>Average across NSNs</td>
      <td>21 containers, 6 items</td>
      <td class="num">—</td>
      <td class="num">{avg_tx:.1f}</td>
      <td class="num">${avg_price:,.0f}</td>
      <td class="num">{avg_bids:.1f}</td>
    </tr>""")

    table = f"""<table class="thesis">
  <caption>Table A1. Treated NSNs: Product, Waiver Date, and Procurement Summary</caption>
  <thead>
    <tr>
      <th>NSN</th>
      <th>Item name</th>
      <th class="num">First waiver</th>
      <th class="num">Transactions</th>
      <th class="num">Avg unit price</th>
      <th class="num">Avg bids</th>
    </tr>
  </thead>
  <tbody>
{chr(10).join(rows)}
  </tbody>
{_note_tfoot(6,
    'The 27 treated NSNs used in the analysis (21 freight containers and six '
    'individual items), sorted by transaction count. Transactions are counted '
    'over fiscal years 2017 through January 2026 and total 2,210. Avg unit price '
    'is the mean unit price observed for the NSN; for 1660-01-600-4909 (heat '
    'exchanger) it is a proxy recovered as contract obligation ÷ quantity, and '
    'for 1680-01-622-9189 (excluded from the unit-price analysis) it is shown as '
    'a dash. Avg bids is the mean number of offers received per transaction. The '
    'final row reports the average across the 27 NSNs (unit price excludes the '
    'one NSN without a recorded price).'
)}</table>"""
    html = wrap_html("Table A1 – Treated NSN Summary", table)
    path = OUT / "TA1_treatment_summary.html"
    path.write_text(html, encoding="utf-8", newline="\n")
    return path


# ---------------------------------------------------------------------------
# Table 7 — Sample funnel: waiver universe to treatment panel (Data section)
# ---------------------------------------------------------------------------
# Counts are re-derived from canonical sources at build time. The final stage is the treated
# NSN universe (28 NSNs) from nsn_universe.csv; the estimators use subsets of it (27 enter
# the event study and matched panels, 26 match, 22-23 fit per outcome in the synthetic control).

# Contracting-office agency names treated as DoD for funnel purposes.
DOD_AGENCY_NAMES = {
    "DEFENSE LOGISTICS AGENCY (DLA)",
    "DEFENSE LOGISTICS AGENCY",
    "DEPT OF THE NAVY",
    "DEPT OF THE ARMY",
    "DEPT OF THE AIR FORCE",
    "DEFENSE HEALTH AGENCY (DHA)",
}


def _compute_sample_funnel() -> pd.DataFrame:
    waivers = pl.read_csv(WAIVERS_CSV, infer_schema_length=2000)
    n_all          = waivers.height
    n_nonavail     = waivers.filter(pl.col("Waiver_Type") == "Nonavailability").height
    n_dod          = waivers.filter(
        (pl.col("Waiver_Type") == "Nonavailability")
        & (pl.col("Contracting_Office_Agency_Name").is_in(list(DOD_AGENCY_NAMES)))
    ).height

    ids = pl.read_csv(WAIVER_DOD_IDS, infer_schema_length=1000)
    nsn_rows = ids.filter(pl.col("identifier_type").str.to_lowercase().str.contains("nsn"))
    n_waivers_with_nsn = nsn_rows.select("waiver_id").n_unique()

    nsn_ref = pl.read_csv(WAIVER_NSN_REF, infer_schema_length=0)
    n_unique_nsns = nsn_ref.select("nsn").n_unique()

    # BidLink coverage: count waiver NSNs that also appear as a file (one
    # CSV per NSN) in the BidLink NSN line-items folder.
    bidlink_nsns: set[str] = set()
    for csv_path in BIDLINK_NSN_DIR.glob("*.csv"):
        try:
            sample = pl.read_csv(csv_path, infer_schema_length=200, n_rows=1, infer_schema=False)
        except Exception:
            continue
        if "NSN" in sample.columns:
            full = pl.read_csv(csv_path, infer_schema_length=200, infer_schema=False)
            bidlink_nsns.update(full["NSN"].drop_nulls().unique().to_list())
    waiver_nsns_dashed = set(nsn_ref["nsn_formatted"].to_list())
    n_with_bidlink = len(waiver_nsns_dashed & bidlink_nsns)

    # Treated NSNs that made it into the analysis panel.
    nsn_universe = pl.read_csv(NSN_UNIVERSE_CSV, infer_schema_length=200)
    n_in_panel = nsn_universe.filter(pl.col("treated") == 1).height
    if n_in_panel != 28:
        raise RuntimeError(
            f"Canonical treated count is {n_in_panel}, not the expected 28. "
            f"Check {NSN_UNIVERSE_CSV.relative_to(REPO)}."
        )

    # Pre-waiver coverage filter: must have at least one observation in
    # event years -5 to -1 (the pre-waiver analysis window). Source is the
    # stage 2 log from the matched-controls pipeline, which records this
    # check explicitly per treated NSN.
    stage2 = pl.read_csv(STAGE2_LOG, infer_schema_length=200)
    n_pre_waiver = stage2.filter(pl.col("skipped") == False).height
    if n_pre_waiver != 27:
        raise RuntimeError(
            f"Pre-waiver-coverage count is {n_pre_waiver}, not the expected 27. "
            f"Check {STAGE2_LOG.relative_to(REPO)}."
        )

    return pd.DataFrame([
        {"stage": "All cleaned waivers",                          "unit": "waivers", "count": n_all},
        {"stage": "Nonavailability waivers only",                 "unit": "waivers", "count": n_nonavail},
        {"stage": "DoD / defense agency",                         "unit": "waivers", "count": n_dod},
        {"stage": "Waivers with explicit NSN codes",              "unit": "waivers", "count": n_waivers_with_nsn},
        {"stage": "Unique waiver NSNs",                           "unit": "NSNs",    "count": n_unique_nsns},
        {"stage": "NSNs with any transaction history",            "unit": "NSNs",    "count": n_with_bidlink},
        {"stage": "With transactions both pre- and post-waiver",  "unit": "NSNs",    "count": n_in_panel},
        {"stage": "With pre-waiver transactions inside the analysis window (event years -5 to -1)", "unit": "NSNs", "count": n_pre_waiver},
    ])


def build_sample_funnel_table() -> Path:
    df = _compute_sample_funnel()
    # Compute step-by-step attrition: percent retained from the previous row.
    prev = None
    rows: list[str] = []
    for _, r in df.iterrows():
        if prev is None or prev == 0:
            pct = "—"
        else:
            pct = f"{(r['count'] / prev * 100):.1f}%"
        prev = r["count"]
        rows.append(f"""    <tr>
      <td>{r['stage']}</td>
      <td class="num">{int(r['count']):,}</td>
      <td>{r['unit']}</td>
      <td class="num">{pct}</td>
    </tr>""")

    table = f"""<table class="thesis">
  <caption>Table 1. Sample Funnel: From Waiver Universe to Treatment Panel</caption>
  <thead>
    <tr>
      <th>Stage</th>
      <th class="num">Count</th>
      <th>Unit</th>
      <th class="num">% retained</th>
    </tr>
  </thead>
  <tbody>
{chr(10).join(rows)}
  </tbody>
{_note_tfoot(4,
    'Stages 1-4 count waivers; stages 5-8 count NSNs. Stage 7 requires '
    'transactions on both sides of the waiver date so a pre/post comparison is '
    'possible. Stage 8 requires that pre-waiver transactions must fall within '
    'the 5-year analysis window.'
)}</table>"""
    html = wrap_html("Table 1 – Sample Funnel", table)
    path = OUT / "T1_sample_funnel.html"
    path.write_text(html, encoding="utf-8", newline="\n")
    return path




# ---------------------------------------------------------------------------
# Combined index — all four tables in one page for convenient browsing.
# ---------------------------------------------------------------------------
TABLE_BUILDERS = (
    build_sample_funnel_table,     # Table 1 (Data)
    # The event study is presented inline plus the appendix ladders A2-A4 (no body summary table).
    build_matched_table,           # Table 2 (Analysis)
    build_synth_table,             # Table 3 (Analysis)
    build_synth_event_time_table,  # Table 4 (Analysis)
    build_foreign_share_table,        # Table A0 (Appendix)
    build_treatment_summary_table,    # Table A1 (Appendix)
)


def build_tables() -> None:
    for fn in TABLE_BUILDERS:
        fn()


def main() -> None:
    build_tables()
    print(f"Wrote {len(TABLE_BUILDERS)} tables to {OUT}")
    for outcome in OUTCOME_ORDER:
        ap = build_appendix_ladder(outcome)
        print(f"  appendix ladder: {ap.relative_to(REPO)}")


if __name__ == "__main__":
    main()
