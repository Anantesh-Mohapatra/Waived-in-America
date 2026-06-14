"""
export_tables.py - Export paper-style event-study regression tables.

Reads cached fitted models from data/cache/main/ and writes:
  results/main/event_study/tables/{outcome}_{panel}.html   6 primary stargazer tables (4 cols)

Each table:
  - One row per event-year coefficient (k=-8...+3), with k=-1 inserted as the
    omitted reference (-- (ref) --).
  - Footer panel:
      Pre-trend p (joint)   joint Wald F-test that beta_{-K}...beta_{-2} = 0
      Avg post beta (k>=0)  linear combination + CRV1 SE
      NSN FE / FY FE        x / -
      N obs, N clusters, R^2

HTML is the repo-wide interchange format for generated tables (the thesis
build parses generator HTML; see thesis/build.py).

Cache-only: never refits. Run-time ~60 seconds (regressions cached).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyfixest.report import etable
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import twfe_helpers as h
from paths import event_study_tables

TABLES_DIR = event_study_tables("main")
TABLES_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_OUTCOMES = ["max_log_unit_price", "mean_offers", "domestic_share"]
PANELS = ["dla_only", "enriched"]
SPEC_NAMES = ["ols", "nsn_fe", "nsn_fy_fe", "with_ctrl"]
SPEC_HEADERS = ["(1) OLS", "(2) +NSN FE", "(3) +FY FE", "(4) +Controls"]
FE_FLAGS = {  # (NSN FE present, FY FE present)
    "ols":       ("-", "-"),
    "nsn_fe":    ("x", "-"),
    "nsn_fy_fe": ("x", "x"),
    "with_ctrl": ("x", "x"),
}

PANEL_LABELS = {"dla_only": "DLA-only", "enriched": "Enriched"}
OUTCOME_LABELS = {
    "max_log_unit_price": "max log unit price",
    "mean_offers": "mean offers",
    "domestic_share": "domestic share",
}


# ---------- Cache loader + statistics ----------
def load_model(cache_key: str):
    """Load a cached fit via the shared helper, which honours `_CACHE_VERSION`
    and rejects stale dumps. Raises if the cache file is missing or stale —
    `estimate_twfe.py` should be run first to populate.
    """
    p = h.CACHE_DIR / f"{cache_key}.joblib"
    m = h._load_cached_model(p)
    if m is None:
        raise FileNotFoundError(
            f"Cache miss or version-stale: {p}. Run estimate_twfe.py first."
        )
    return m


# Model-statistics helpers shared with the robustness refits live in lib.
from es_stats import (  # noqa: E402
    avg_post_beta,
    joint_pretrend_pvalue,
    n_clusters,
)


def fmt_p(p: float | None) -> str:
    if p is None:
        return "-"
    star = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    return f"{p:.3f}{star}"


def fmt_avg(est: float | None, se: float | None) -> str:
    if est is None:
        return "-"
    z = est / se if se > 0 else float("nan")
    p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else float("nan")
    star = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    return f"{est:.3f}{star}\n({se:.3f})"


# ---------- Table assembly ----------
EVENT_YEAR_LABEL_RE = re.compile(r"^event_year[^\-\d]*(-?\d+)\s*$")


def relabel_event_year(label: str) -> str | None:
    """pyfixest renders categorical interactions as 'event_year x x -8'.
    Convert to 'k=-8', 'k=0', 'k=+1' etc. Returns None if not an event_year row."""
    m = EVENT_YEAR_LABEL_RE.match(str(label).strip())
    if not m:
        return None
    ey = int(m.group(1))
    if ey == 0:
        return "k=0"
    return f"k={ey:+d}"


EXPECTED_CONTROLS_BY_SPEC = {
    "ols":       [],
    "nsn_fe":    [],
    "nsn_fy_fe": [],
    "with_ctrl": ["competitive_share", "sole_source_share"],
}


def control_cell(model, spec_name: str, var: str,
                 raw_df: pd.DataFrame, col_name: str) -> str:
    """Return the rendered cell for a control variable in this model.
    - If the variable is in ``_collin_vars``, the model dropped it for collinearity.
    - If it is in the model's coef table, use the etable-rendered cell.
    - Otherwise, the variable wasn't in this spec at all.
    """
    if var not in EXPECTED_CONTROLS_BY_SPEC.get(spec_name, []):
        return ""
    collin = list(getattr(model, "_collin_vars", []) or [])
    if var in collin:
        return "(collinear, dropped)"
    # Pull from raw etable output; rows are MultiIndex ('coef', name).
    key = ("coef", var)
    if key in raw_df.index:
        return str(raw_df.loc[key, col_name])
    return "(not estimated)"


def build_table_df(models: list, spec_names: list[str],
                   headers: list[str], outcome: str) -> pd.DataFrame:
    """Build the post-processed DataFrame for one outcome x panel: per-event-year
    rows + ref row + footer (Pre-trend p, Avg post beta, FE markers, N, clusters, R^2).
    """
    pretrend_ps = [joint_pretrend_pvalue(m) for m in models]
    post_avgs = [avg_post_beta(m) for m in models]
    nsn_fe = [FE_FLAGS[s][0] for s in spec_names]
    fy_fe = [FE_FLAGS[s][1] for s in spec_names]
    n_obs = [f"{int(getattr(m, '_N', 0)):,}" for m in models]
    n_clu = [f"{n_clusters(m):,}" for m in models]
    r2 = [f"{float(getattr(m, '_r2', float('nan'))):.3f}" for m in models]

    raw = etable(
        models, type="df",
        model_heads=headers,
        signif_code=[0.01, 0.05, 0.1],
        coef_fmt="b:.3f* \n (se:.3f)",
        drop=[r"1000", r"^Intercept$"],
        show_fe=False,
    )

    # Flatten 3-level column MultiIndex to single header strings.
    new_cols = [c[1] if isinstance(c, tuple) else c for c in raw.columns]
    raw.columns = new_cols

    # Walk the rows: keep event_year coefs (relabeled), drop FE/stats blocks
    # (we add our own footer); covariate rows are rebuilt below from the expected
    # control list so collinearity-dropped vars surface explicitly.
    coef_rows: dict[str, list[str]] = {}  # ordered
    for tup in raw.index:
        kind, name = tup if isinstance(tup, tuple) else ("coef", tup)
        if kind in ("fe", "stats"):
            continue
        relabeled = relabel_event_year(name)
        if relabeled is not None:
            coef_rows[relabeled] = [str(raw.loc[tup, c]) for c in new_cols]

    # Rebuild covariate rows: for each expected control across all specs in the
    # set, render a row showing coef/SE, "(collinear, dropped)", or "" per spec.
    expected_ctrls = []
    seen = set()
    for sn in spec_names:
        for v in EXPECTED_CONTROLS_BY_SPEC.get(sn, []):
            if v not in seen:
                expected_ctrls.append(v)
                seen.add(v)
    cov_rows: dict[str, list[str]] = {}
    for v in expected_ctrls:
        cov_rows[v] = [
            control_cell(m, sn, v, raw, col)
            for m, sn, col in zip(models, spec_names, new_cols)
        ]

    # Sort coef rows by event-year order, insert k=-1 reference row.
    def ey_key(label: str) -> int:
        return int(label.replace("k=", ""))
    sorted_keys = sorted(coef_rows.keys(), key=ey_key)
    insert_at = next((i for i, k in enumerate(sorted_keys) if ey_key(k) >= 0),
                     len(sorted_keys))
    ordered_coef_rows = []
    for k in sorted_keys[:insert_at]:
        ordered_coef_rows.append((k, coef_rows[k]))
    ordered_coef_rows.append(("k=-1", ["- (ref) -"] * len(headers)))
    for k in sorted_keys[insert_at:]:
        ordered_coef_rows.append((k, coef_rows[k]))

    # Build final DataFrame.
    rows: list[tuple[str, str, list[str]]] = []  # (section, label, values)
    for label, vals in ordered_coef_rows:
        rows.append(("Event-year", label, vals))
    for label, vals in cov_rows.items():
        rows.append(("Controls", label, vals))
    rows.append(("Fixed effects", "NSN FE", nsn_fe))
    rows.append(("Fixed effects", "FY FE", fy_fe))
    rows.append(("Aggregates",
                 "Pre-trend p (joint, k<-1 = 0)",
                 [fmt_p(p) for p in pretrend_ps]))
    rows.append(("Aggregates",
                 "Avg post beta (k>=0)",
                 [fmt_avg(est, se) for est, se in post_avgs]))
    rows.append(("Fit", "N obs", n_obs))
    rows.append(("Fit", "N clusters (NSN)", n_clu))
    rows.append(("Fit", "R-squared", r2))

    # Build a flat DataFrame with a section-tagged index for HTML rendering.
    idx = pd.MultiIndex.from_tuples(
        [(s, l) for s, l, _ in rows], names=["section", ""]
    )
    data = [v for _, _, v in rows]
    df = pd.DataFrame(data, index=idx, columns=headers)
    return df


# ---------- Output writers ----------
# Stargazer-style: thick top + bottom rules, thin mid-rule, two-line header,
# tabular numerals, italic notes panel. Serif body.
HTML_CSS = r"""
<style>
:root {
  --rule-thick: #000;
  --rule-thin: #666;
  --rule-section: #999;
  --text: #111;
  --muted: #555;
  --se: #555;
}
html, body { background: #fff; color: var(--text); margin: 0; }
body {
  font-family: 'Charter', 'Iowan Old Style', 'Cambria', 'Georgia',
               'Times New Roman', serif;
  padding: 36px 40px;
  font-feature-settings: 'tnum' 1, 'lnum' 1;  /* tabular, lining numerals */
}
.title { font-size: 15pt; font-weight: 600; margin: 0 0 4px 0; letter-spacing: 0.01em; }
.subtitle {
  font-size: 10pt; color: var(--muted); font-style: italic;
  margin: 0 0 18px 0; max-width: 760px;
}
table.es {
  border-collapse: collapse;
  border-spacing: 0;
  margin: 0;
  font-size: 11pt;
  border-top: 2px solid var(--rule-thick);
  border-bottom: 2px solid var(--rule-thick);
}
table.es thead tr.h-num th { padding: 6px 22px 1px; font-weight: 400; }
table.es thead tr.h-desc th {
  padding: 1px 22px 6px;
  font-weight: 600;
  border-bottom: 1px solid var(--rule-thick);
}
table.es thead tr.h-only th {
  padding: 6px 22px;
  font-weight: 600;
  border-bottom: 1px solid var(--rule-thick);
}
table.es th.row { text-align: left; padding-left: 0; padding-right: 26px; }
table.es th { text-align: center; }
table.es td {
  padding: 1px 22px;
  text-align: center;
  vertical-align: top;
  border: none;
  white-space: nowrap;
  font-variant-numeric: tabular-nums lining-nums;
}
table.es td.row {
  text-align: left; padding-left: 0; padding-right: 26px;
  white-space: nowrap;
}
table.es tr.coefrow td.b, table.es tr.coefrow td.row {
  padding-top: 5px;       /* breathing room above each coef pair */
}
table.es tr.coefrow td .b  { display: block; line-height: 1.18; }
table.es tr.coefrow td .se { display: block; color: var(--se);
                              font-size: 0.88em; line-height: 1.18;
                              padding-bottom: 4px; }
/* Section separator: a thin rule above the first row of each new section. */
table.es tr.sep td, table.es tr.sep th {
  border-top: 1px solid var(--rule-section);
  padding-top: 6px;
}
table.es td.ref { color: var(--muted); font-style: italic; }
table.es td.dropped { color: var(--muted); font-style: italic; font-size: 0.95em; }
.notes {
  font-size: 9.5pt; color: var(--muted); margin: 14px 0 0 0;
  max-width: 760px; line-height: 1.5; font-style: italic;
}
.notes b { font-weight: 600; font-style: normal; color: var(--text); }
.notes code { font-style: normal; font-family: 'SF Mono', 'Consolas',
                                                'Liberation Mono', monospace;
              font-size: 0.92em; }
</style>
"""

# Section IDs that should render coef/SE as a stacked pair (cells contain '\n').
_COEF_SECTIONS = {"Event-year", "Controls", "Aggregates"}


def _split_header(h: str) -> tuple[str, str]:
    """For 'primary' headers like '(1) OLS', return ('(1)', 'OLS').
    For headers without a leading paren, return ('', h)."""
    s = h.strip()
    if s.startswith("(") and ")" in s:
        end = s.index(")") + 1
        num = s[:end]
        rest = s[end:].strip()
        return num, rest
    return "", s


def _render_cell(value: str, section: str) -> str:
    """Convert pyfixest-style 'b* \\n (se)' cells into stacked spans.
    Other rows (FE markers, N, R^2) render as plain text."""
    s = str(value).strip()
    if section in _COEF_SECTIONS and "\n" in s:
        top, bot = s.split("\n", 1)
        return (f"<span class='b'>{top.strip()}</span>"
                f"<span class='se'>{bot.strip()}</span>")
    if s == "- (ref) -":
        return "<span class='b'>—</span><span class='se'>(ref.)</span>"
    if "(collinear, dropped)" in s:
        return "<span class='dropped'>(collinear, dropped)</span>"
    return s


def df_to_styled_html(df: pd.DataFrame, title: str, subtitle: str,
                      notes: str) -> str:
    """Render the post-processed DataFrame as a stargazer-style HTML table.

    Two-row header (column numbers + descriptors) when headers contain '(N)'
    prefixes; single-row header otherwise. Section separators rendered as
    thin top-borders on the first row of each new section. Coef/SE cells
    rendered as stacked spans for clean numeric alignment.
    """
    headers = list(df.columns)
    split = [_split_header(h) for h in headers]
    has_two_row_header = any(num for num, _ in split)

    parts = ["<!DOCTYPE html><html><head><meta charset='utf-8'>"]
    parts.append(HTML_CSS)
    parts.append(f"<title>{title}</title></head><body>")
    parts.append(f"<div class='title'>{title}</div>")
    parts.append(f"<div class='subtitle'>{subtitle}</div>")
    parts.append("<table class='es'>")

    # ---- Header rows ----
    parts.append("<thead>")
    if has_two_row_header:
        parts.append("<tr class='h-num'><th class='row'></th>")
        for num, _ in split:
            parts.append(f"<th>{num}</th>")
        parts.append("</tr>")
        parts.append("<tr class='h-desc'><th class='row'></th>")
        for _, desc in split:
            parts.append(f"<th>{desc}</th>")
        parts.append("</tr>")
    else:
        parts.append("<tr class='h-only'><th class='row'></th>")
        for h_ in headers:
            parts.append(f"<th>{h_}</th>")
        parts.append("</tr>")
    parts.append("</thead><tbody>")

    # ---- Body rows ----
    prev_section = None
    for (section, label) in df.index:
        is_section_start = (prev_section is not None and section != prev_section)
        classes = []
        if section in _COEF_SECTIONS:
            classes.append("coefrow")
        if is_section_start:
            classes.append("sep")
        cls_attr = f" class='{' '.join(classes)}'" if classes else ""
        parts.append(f"<tr{cls_attr}>")
        parts.append(f"<td class='row'>{label}</td>")
        for h_ in headers:
            v = df.loc[(section, label), h_]
            parts.append(f"<td>{_render_cell(v, section)}</td>")
        parts.append("</tr>")
        prev_section = section
    parts.append("</tbody></table>")
    if notes:
        parts.append(f"<div class='notes'>{notes}</div>")
    parts.append("</body></html>")
    return "".join(parts)


# ---------- Driver ----------
def export_one(outcome: str, panel: str,
               spec_names: list[str], headers: list[str]):
    models = [
        load_model(f"{outcome}__{panel}__{sn}")
        for sn in spec_names
    ]
    df = build_table_df(models, spec_names, headers, outcome)

    title = f"{OUTCOME_LABELS[outcome]} - {PANEL_LABELS[panel]}"
    subtitle = (
        f"TWFE event study, ref k=-1. NSN-clustered SEs (CRV1). "
        f"Stars: *** p&lt;0.01, ** p&lt;0.05, * p&lt;0.1."
    )
    # Detect collinearity drops in the with_ctrl spec — but only for vars we
    # care about (the controls we passed in). The sentinel ``event_year::1000``
    # is in ``_collin_vars`` by design (it absorbs into NSN FE) and shouldn't
    # be flagged.
    dropped_in_with_ctrl: list[str] = []
    for sn, mdl in zip(spec_names, models):
        if sn == "with_ctrl":
            all_dropped = list(getattr(mdl, "_collin_vars", []) or [])
            dropped_in_with_ctrl = [
                v for v in all_dropped
                if v in EXPECTED_CONTROLS_BY_SPEC.get("with_ctrl", [])
            ]
    collin_note_html = ""
    if dropped_in_with_ctrl:
        joined = ", ".join(f"<code>{v}</code>" for v in dropped_in_with_ctrl)
        collin_note_html = (
            f" Spec (4) was specified with both <code>competitive_share</code> "
            f"and <code>sole_source_share</code>; pyfixest dropped {joined} "
            f"for perfect collinearity (within-NSN-and-FY the cell shares of "
            f"competition codes effectively sum to a constant)."
        )

    notes = (
        "<b>Notes.</b> Coefficients are event-time dummies on the "
        "(NSN, FY, event_year) cell panel. The reference period is k=-1 "
        "(year before the first waiver). Endpoint coefficients (k=-8..-5, k=+3) "
        "are identified off &le;9 NSNs each. <b>Pre-trend p (joint)</b> is the "
        "Wald F-test that &beta;<sub>k</sub>=0 jointly for k&lt;-1. "
        "<b>Avg post &beta; (k&ge;0)</b> is the linear combination of "
        "post-period coefficients with CRV1 standard error."
        f"{collin_note_html}"
    )

    slug = f"{outcome}_{panel}"
    (TABLES_DIR / f"{slug}.html").write_text(
        df_to_styled_html(df, title, subtitle, notes), encoding="utf-8", newline="\n")
    print(f"  wrote {slug}.html")


def main():
    print("Exporting primary tables (3 outcomes x 2 panels = 6 tables)...")
    for outcome in PRIMARY_OUTCOMES:
        for panel in PANELS:
            export_one(outcome, panel, SPEC_NAMES, SPEC_HEADERS)

    print(f"\nDONE. Tables written to {TABLES_DIR}")


if __name__ == "__main__":
    main()
