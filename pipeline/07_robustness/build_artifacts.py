"""Build thesis-style figures + tables for each robustness variant.

Reuses the mainline generators (pipeline/08_artifacts/build_thesis_figures.py
and build_thesis_tables.py) by passing the variant inputs/outputs as explicit
arguments (the generator functions default to the main-analysis values), so
the robustness artifacts are byte-for-byte the thesis convention (styling,
naming, significance legend, notes).

Per variant it emits, into results/<variant>/figures and results/<variant>/tables:
  figures: event-study coefplots + 3-spec ladders, matched per-NSN dotplots
           (both windows), synth forests, synth event-time paths.
  tables:  event-study summary, matched-controls summary, synth summary,
           synth event-time, appendix event-study ladders.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import variants as C

# Import the mainline generators (btf imports btt's LADDER_DATA at load).
THESIS_SCRIPTS = C.REPO / "pipeline" / "08_artifacts"
sys.path.insert(0, str(THESIS_SCRIPTS))
import build_thesis_tables as btt  # noqa: E402
import build_thesis_figures as btf  # noqa: E402

OUTCOME_ORDER = btt.OUTCOME_ORDER  # ["domestic_share", "max_log_unit_price", "mean_offers"]


# ----- reconstruct the python structures the generators expect from JSON ------
# JSON -> generator-shape transforms now live in es_stats (single owner, shared
# with the main thesis generators). Local aliases keep the call sites below short.
sys.path.insert(0, str(C.REPO / "pipeline" / "lib"))
from es_stats import es_spec_from_json as _es_spec, ladder_from_json as _ladder  # noqa: E402


# Reader-facing item names for the non-container per-item table.
NONCONTAINER_ITEM_NAMES = {
    "1660016004909": "Heat exchanger",
    "1680016229189": "Ground-proximity warning system",
    "2895145543146": "Pneumatic motor",
    "3990015742050": "Materials-handling roller",
    "4240015387970": "Hearing protector",
    "4510015272274": "Shower bath fixture",
}

VARIANT_LABEL = C.VARIANT_LABEL  # reader-facing labels, single source in variants.py
MAIN_MATCHED_SUMMARY = C.MATCHED_TABLES / "nn_did_summary_fsg_off.csv"


def _fmt_cmp(v) -> str:
    return "—" if v is None else f"{v:+.3f}"


def build_comparison_table(variant: str, es_json: dict, paths: dict,
                           have_matched: bool, have_synth: bool, out_dir: Path) -> Path:
    """Thesis-style table putting each estimator's robustness estimate beside its
    main-analysis (full-sample) value, so a reader sees the comparison directly
    rather than flipping between the appendix and the main Analysis tables.
    Within-estimator comparison only: the three estimators use different metrics
    and are not comparable across rows."""
    rows: list[tuple[str, str, float | None, float | None]] = []
    for o in OUTCOME_ORDER:
        rows.append(("Event study (avg. post-period coefficient)", btt.OUTCOME_LABEL[o],
                     btf.ES_COEFS[o]["avg_post"][0],
                     es_json[o]["summary_row"]["avg_post_beta"]))
    if have_matched:
        mb = pl.read_csv(MAIN_MATCHED_SUMMARY)
        mc = pl.read_csv(paths["matched_tables"] / "nn_did_summary_fsg_off.csv")
        for win, lab in [("headline", "Matched controls, ATT (event year ≥ 0)"),
                         ("ey2plus", "Matched controls, ATT (event year ≥ 2)")]:
            for o in OUTCOME_ORDER:
                m = mb.filter((pl.col("outcome") == o) & (pl.col("post_label") == win))
                c = mc.filter((pl.col("outcome") == o) & (pl.col("post_label") == win))
                rows.append((lab, btt.OUTCOME_LABEL[o],
                             m["att"][0] if m.height else None,
                             c["att"][0] if c.height else None))
    if have_synth:
        sb = pl.read_csv(C.SYNTH_SUMMARY)
        sc = pl.read_csv(paths["synth_tables"] / "synth_summary.csv")
        for o in OUTCOME_ORDER:
            m = sb.filter(pl.col("outcome") == o)
            c = sc.filter(pl.col("outcome") == o)
            rows.append(("Synthetic controls (mean per-NSN ATT)", btt.OUTCOME_LABEL[o],
                         m["mean_att"][0] if m.height else None,
                         c["mean_att"][0] if c.height else None))

    label = VARIANT_LABEL.get(variant, variant)
    body, prev = [], None
    for estimator, outcome, main, chk in rows:
        sep = ' style="border-top:1px solid #999"' if (prev is not None and estimator != prev) else ""
        e_cell = estimator if estimator != prev else ""
        body.append(f'    <tr{sep}><td>{e_cell}</td><td>{outcome}</td>'
                    f'<td class="num">{_fmt_cmp(main)}</td>'
                    f'<td class="num">{_fmt_cmp(chk)}</td></tr>')
        prev = estimator
    note = (
        "Each robustness estimate beside its main-analysis (full-sample) value for the "
        "same estimator. The event study reports the average post-period event-year "
        "coefficient; matched controls the collapsed pre/post ATT in two post-period "
        "windows (all post-waiver years, and event year ≥ 2 onward); synthetic controls "
        "the mean per-NSN ATT."
    )
    table = (f'<table class="thesis">\n'
             f'  <caption>Table R. {label}: Robustness Estimates vs the Main Analysis</caption>\n'
             f'  <thead>\n    <tr><th>Estimator</th><th>Outcome</th>'
             f'<th class="num">Main analysis</th><th class="num">{label}</th></tr>\n'
             f'  </thead>\n  <tbody>\n' + "\n".join(body) + "\n  </tbody>\n"
             + btt._note_tfoot(4, note) + "</table>")
    p = out_dir / "T0_comparison_vs_main.html"
    p.write_text(btt.wrap_html(f"{label} vs Main Analysis", table), encoding="utf-8", newline="\n")
    return p


def build_per_item_domestic_table(paths: dict, out_dir: Path) -> Path | None:
    """Per-item domestic-sourcing effect for the (tiny) non-container sample, so a
    reader can see that the long-term aggregate is driven by one item. Shows the
    synth demeaned effect at each post-waiver event year and the matched per-item
    τ in both post windows. Built only for the non-container check."""
    eff_dir = paths["effect_path"]
    tau_path = paths["matched_tables"] / "nn_did_tau_per_nsn_fsg_off.csv"
    if not eff_dir.exists() or not tau_path.exists():
        return None

    # synth per-item demeaned effect by post event year (domestic_share)
    synth = {}
    for f in eff_dir.glob("*__domestic_share.csv"):
        nsn = f.stem.split("__")[0]
        d = pl.read_csv(f).filter(pl.col("is_pre") == False)
        synth[nsn] = {int(r["event_year"]): r["effect_demeaned"] for r in d.iter_rows(named=True)}
    # matched per-item tau by window (domestic_share)
    tau = pl.read_csv(tau_path, schema_overrides={"treated_nsn": pl.Utf8}).filter(
        pl.col("outcome") == "domestic_share")
    mt = {}
    for r in tau.iter_rows(named=True):
        mt.setdefault(r["treated_nsn"], {})[r["post_label"]] = r["tau"]

    items = sorted(set(synth) | set(mt))
    if not items:
        return None

    def cell(v):
        return "—" if v is None else f"{v:+.3f}"

    body = []
    for nsn in items:
        s = synth.get(nsn, {})
        m = mt.get(nsn, {})
        name = NONCONTAINER_ITEM_NAMES.get(nsn, nsn)
        body.append(
            f'    <tr><td>{name}</td>'
            f'<td class="num">{cell(s.get(0))}</td>'
            f'<td class="num">{cell(s.get(1))}</td>'
            f'<td class="num">{cell(s.get(2))}</td>'
            f'<td class="num">{cell(m.get("headline"))}</td>'
            f'<td class="num">{cell(m.get("ey2plus"))}</td></tr>')
    note = (
        "Domestic-sourcing effect per non-container item. Synthetic-control columns are "
        "the demeaned effect (treated minus synthetic) at each post-waiver event year; "
        "matched-controls columns are the per-item τ over all post-waiver years (event "
        "year ≥ 0) and from event year ≥ 2 onward. A dash means the "
        "estimator produced no estimate for that item or window (item dropped, or no "
        "transactions in that window)."
    )
    table = (
        '<table class="thesis">\n'
        '  <caption>Table R2. Non-Container Check: Domestic-Sourcing Effect by Item</caption>\n'
        '  <thead>\n'
        '    <tr><th rowspan="2">Item</th>'
        '<th colspan="3" class="num">Synthetic controls (effect by event year)</th>'
        '<th colspan="2" class="num">Matched controls (τ by event-year window)</th></tr>\n'
        '    <tr><th class="num">0</th><th class="num">1</th><th class="num">2</th>'
        '<th class="num">≥ 0</th><th class="num">≥ 2</th></tr>\n'
        '  </thead>\n  <tbody>\n' + "\n".join(body) + "\n  </tbody>\n"
        + btt._note_tfoot(6, note) + "</table>")
    p = out_dir / "T5_per_item_domestic.html"
    p.write_text(btt.wrap_html("Non-Container: Per-Item Domestic Sourcing", table), encoding="utf-8", newline="\n")
    return p


def _sanitize_undefined_se(path: Path) -> None:
    """Render undefined statistics (n=1 cells have no SE/p) as an em-dash
    instead of the literal 'nan' that float formatting produces. Honest and
    clean; the N column still shows the single observation."""
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    text = text.replace("(nan) &middot; p nan", "(&mdash;) &middot; p &mdash;")
    text = text.replace("(nan) · p nan", "(—) · p —")
    text = text.replace("(nan)", "(—)").replace("p nan", "p —")
    # The reused generator's note interprets a result ("all N treated NSNs still
    # match under this rule"); appendix table notes stay descriptive, so drop the
    # clause entirely (the per-row N column already shows the match counts).
    text = text.replace("; all 26 treated NSNs still match under this rule", "")
    path.write_text(text, encoding="utf-8", newline="\n")


# ----- per-variant build -----------------------------------------------------
def build(variant: str) -> None:
    """Build whatever designs have inputs present in this variant's results tree.

    Graceful per-design existence checks make the build decoupled: after the fast
    designs run, the event-study and matched artifacts appear; once synth
    finishes, re-running build adds the synth artifacts. (The re-aggregation
    variants always have all inputs, so they get the full set.)
    """
    paths = C.variant_paths(variant)
    figs, tabs = paths["figures"], paths["tables"]
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    es_json = json.loads(paths["es_json"].read_text(encoding="utf-8"))

    matched_summary_off = paths["matched_tables"] / "nn_did_summary_fsg_off.csv"
    have_matched = matched_summary_off.exists()
    have_synth = (paths["synth_tables"] / "synth_att.csv").exists()

    ladder = {o: _ladder(es_json[o]["ladder"]) for o in OUTCOME_ORDER}

    # ---- figures ----
    mn: dict[tuple[str, str], int] = {}
    if have_matched:
        _msum = pl.read_csv(matched_summary_off)
        mn = {(r["outcome"], r["post_label"]): int(r["n_treated"])
              for r in _msum.iter_rows(named=True)}

    made = []
    for o in OUTCOME_ORDER:
        made.append(btf.plot_event_study(o, _es_spec(es_json[o]["es_coefs"]),
                                         out_dir=figs))
        made.append(btf.plot_event_study_ladder(o, ladder_data=ladder, out_dir=figs))
        if have_matched:
            # A per-item dotplot needs at least two items to show a distribution;
            # a one-item window also has no computable SE for the note.
            for win in ("headline", "ey2plus"):
                if mn.get((o, win), 0) >= 2:
                    made.append(btf.plot_nn_did_dotplot(
                        o, win,
                        tau_path=paths["matched_tables"] / "nn_did_tau_per_nsn_fsg_off.csv",
                        summary_path=matched_summary_off,
                        out_dir=figs))
        if have_synth:
            made.append(btf.plot_synth_forest(
                o,
                att_path=paths["synth_tables"] / "synth_att.csv",
                summary_path=paths["synth_tables"] / "synth_summary.csv",
                out_dir=figs))
            made.append(btf.plot_synth_event_time(
                o, effect_path_dir=paths["effect_path"], out_dir=figs))

    # ---- tables ----
    btt.build_event_study_table(
        summary=[es_json[o]["summary_row"] for o in OUTCOME_ORDER], out_dir=tabs)
    for o in OUTCOME_ORDER:
        # The small samples can drop endpoint event years; restrict the ladder
        # table's row list to the EYs actually estimated for this outcome.
        btt.build_appendix_ladder(o, ladder_data=ladder,
                                  event_years=es_json[o]["present_eys"],
                                  out_dir=tabs)

    if have_matched:
        btt.build_matched_table(
            summary_off=matched_summary_off,
            summary_on=paths["matched_tables"] / "nn_did_summary_fsg_on.csv",
            out_dir=tabs)
        # n=1 cells (tiny windows) have no SE/p; render as em-dash.
        _sanitize_undefined_se(tabs / "T2_matched_controls.html")
    if have_synth:
        btt.build_synth_table(
            summary_path=paths["synth_tables"] / "synth_summary.csv", out_dir=tabs)
        btt.build_synth_event_time_table(
            effect_path_dir=paths["effect_path"], out_dir=tabs)

    # Comparison table: each robustness estimate beside its main-analysis value.
    build_comparison_table(variant, es_json, paths, have_matched, have_synth, tabs)
    # Non-container check: per-item domestic-sourcing table (shows the long-term
    # aggregate is driven by a single item).
    if variant == "non_container" and have_matched and have_synth:
        build_per_item_domestic_table(paths, tabs)

    designs = "event study" + (" + matched" if have_matched else "") + (" + synth" if have_synth else "")
    print(f"  {variant}: built [{designs}] - {len(made)} figures -> {figs.relative_to(C.REPO)}")
    print(f"  {variant}: tables -> {tabs.relative_to(C.REPO)}")


def run() -> None:
    for variant in C.REAGG_VARIANTS:
        build(variant)
    # dla_only is a full re-fit with its own schedule; build it here too if it
    # has at least the event-study output.
    if C.variant_paths("dla_only")["es_json"].exists():
        build("dla_only")


if __name__ == "__main__":
    run()
