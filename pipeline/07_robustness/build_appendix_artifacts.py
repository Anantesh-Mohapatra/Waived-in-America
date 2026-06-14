"""Generate the robustness-appendix artifacts (figures + tables), no preview/draft coupling.

Reads already-computed results only (no re-estimation) and emits, for the trimmed
appendix design:
All deliverables are written to results/appendix/:
  - Figures A13-A14: the two non-container synthetic-control per-NSN forests
    (domestic sourcing, maximum logged unit price).
  - Tables A7-A25: a single standalone HTML document
    (robustness_appendix_tables.html) with every table, A-numbered,
    styled in the thesis convention (Times New Roman, en dashes) — open in a browser
    and copy each table into Google Docs.

Per check the tables are: robustness estimates vs main; event-study coefficient by
event year vs main; matched-controls ATT by window; matched-controls fit diagnostics
vs main; synthetic-control event-time path by year vs main; synthetic-control per-NSN
ATT distribution; and (non-container only) per-item domestic-sourcing effect.
"""
from __future__ import annotations
import json, re, pathlib, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats as _stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import variants as V
import paths as P  # importable because variants put pipeline/lib on sys.path

ROOT = V.REPO
ART_DIR = P.APPENDIX                                    # all deliverables land here
TABLES_OUT = ART_DIR / "robustness_appendix_tables.html"
MAIN_SYNTH = V.SYNTH_ATT.parent
MAIN_MATCHED = V.MATCHED_TABLES
sys.path.insert(0, str(ROOT / "pipeline" / "08_artifacts"))
import build_thesis_figures as btf   # for the canonical main ES_COEFS

OUTCOMES = ["domestic_share", "max_log_unit_price", "mean_offers"]
OLAB = {"domestic_share": "domestic sourcing share",
        "max_log_unit_price": "maximum logged unit price",
        "mean_offers": "mean offers"}
CHECK = V.VARIANT_LABEL  # reader-facing labels, single source in variants.py
VARIANTS = [*V.REAGG_VARIANTS, "dla_only"]
_vp = V.variant_paths

def _samp(add):
    """Sample-name phrase for prose notes; avoids 'Common Sample sample' doubling."""
    lab = CHECK[add]
    return lab if lab.lower().endswith("sample") else f"{lab} sample"

# Match the mainline thesis figure style (Times New Roman serif, STIX math).
sns.set_theme(context="paper", style="whitegrid", palette="deep")
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
matplotlib.rcParams["mathtext.fontset"] = "stix"
PAL = sns.color_palette("deep")
C_CHECK, C_SIG, C_MAIN = PAL[0], PAL[3], "#7f7f7f"


def _es_json(add):
    return json.loads(_vp(add)["es_json"].read_text(encoding="utf-8"))

def _event_time_agg(eff_dir, outcome):
    rows = [pd.read_csv(f)[["event_year", "effect_demeaned"]]
            for f in sorted(pathlib.Path(eff_dir).glob(f"*__{outcome}.csv"))]
    if not rows:
        return None
    long = pd.concat(rows, ignore_index=True).rename(columns={"effect_demeaned": "e"})
    def agg(g):
        n = len(g); m = g.mean()
        if n > 1:
            se = g.std(ddof=1) / np.sqrt(n)
            p = 2 * _stats.t.sf(abs(m / se), n - 1) if se > 0 else np.nan
        else:
            se = p = np.nan
        return pd.Series({"mean": m, "se": se, "n": n, "p": p})
    return long.groupby("event_year")["e"].apply(agg).unstack().reset_index()

def _pstars(p):
    if p != p:
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# ----- the one figure: non-container synth per-NSN forest ---------------------
def fig_synth_forest(add, outcome, out_path):
    att = pd.read_csv(_vp(add)["synth_tables"] / "synth_att.csv")
    att = att[att.outcome == outcome].dropna(subset=["att"]).sort_values("att").reset_index(drop=True)
    if att.empty:
        return None
    att["lab"] = att["treated_nsn"].astype(str).str[-4:]
    att["sig"] = att["empirical_p"] < 0.05
    main_mean = pd.read_csv(MAIN_SYNTH / "synth_summary.csv").set_index("outcome").loc[outcome, "mean_att"]

    fig, ax = plt.subplots(figsize=(6.5, max(3.0, .22 * len(att) + 1.0)))
    ax.axvline(0, color="black", lw=0.8)
    mline = ax.axvline(main_mean, color=C_MAIN, ls="--", lw=1.2, label=f"Main mean ATT = {main_mean:+.3f}")
    cols = [C_SIG if s else C_CHECK for s in att["sig"]]
    for i, (_, r) in enumerate(att.iterrows()):
        ax.plot([r["ci_lo"], r["ci_hi"]], [i, i], color=cols[i], lw=1.0, alpha=.7)
    ax.scatter(att["att"], range(len(att)), c=cols, s=35, zorder=3)
    ax.set_yticks(range(len(att))); ax.set_yticklabels(att["lab"], fontsize=8)
    ax.set_ylabel("Treated NSN (last 4 digits)"); ax.set_xlabel(f"Synth ATT ({OLAB[outcome]})")
    ax.set_title(f"{CHECK[add]}: Synthetic-control per-NSN ATT – {OLAB[outcome]}")
    h = [plt.Line2D([0], [0], marker="o", color="w", mfc=C_SIG, ms=7, label="Empirical p < 0.05"),
         plt.Line2D([0], [0], marker="o", color="w", mfc=C_CHECK, ms=7, label="Not significant"),
         mline]
    ax.legend(handles=h, loc="best", frameon=True, facecolor="white", framealpha=.92, edgecolor="#ccc", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


# ============================ TABLES (HTML) ===================================
def _wrap(caption, head_html, body_rows, note, ncols):
    return (f'<table class="thesis">\n  <caption>{caption}</caption>\n'
            f'  <thead>\n{head_html}\n  </thead>\n  <tbody>\n'
            + "\n".join(body_rows) + "\n  </tbody>\n"
            f'  <tfoot><tr><td colspan="{ncols}" class="note">{note}</td></tr></tfoot>\n</table>')

SIG_LEGEND = "Significance: *** p &lt; 0.01, ** p &lt; 0.05, * p &lt; 0.10."

def _znorm_stars(b, se):
    if not se or se != se:
        return ""
    p = 2 * _stats.norm.sf(abs(b / se))
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

def tbl_event_study_yearly(add, num):
    es = _es_json(add)
    per, mper, eyset = {}, {}, set()
    for o in OUTCOMES:
        per[o] = {int(ey): specs[2] for ey, specs in es[o]["ladder"]["coefs"]}   # (beta, se, stars)
        mper[o] = {int(k): (b, se) for k, b, se in btf.ES_COEFS[o]["coefs"]}      # main (beta, se)
        eyset |= {k for k in per[o] if -4 <= k <= 3}
    eys = sorted(eyset | {-1})
    h1 = ('    <tr><th rowspan="2" class="num">Event year</th>'
          + "".join(f'<th colspan="2" class="num">{OLAB[o].capitalize()}</th>' for o in OUTCOMES) + "</tr>")
    h2 = ('    <tr>' + "".join(f'<th class="num">{CHECK[add]}</th><th class="num">Main</th>'
                               for _ in OUTCOMES) + "</tr>")
    body = []
    for ey in eys:
        if ey == -1:
            body.append('    <tr><td class="num">-1</td>'
                        + "".join('<td class="num">0 (ref.)</td><td class="num">0 (ref.)</td>'
                                  for _ in OUTCOMES) + "</tr>")
            continue
        ylab = "0" if ey == 0 else f"{ey:+d}"
        cells = []
        for o in OUTCOMES:
            c = per[o].get(ey)
            chk = "–" if c is None else f"{c[0]:+.3f}{c[2]} ({c[1]:.3f})"
            m = mper[o].get(ey)
            mn = "–" if m is None else f"{m[0]:+.3f}{_znorm_stars(m[0], m[1])}"
            cells.append(f'<td class="num">{chk}</td><td class="num">{mn}</td>')
        body.append(f'    <tr><td class="num">{ylab}</td>' + "".join(cells) + "</tr>")
    note = (f"Event-year coefficient for the {_samp(add)} beside the main-analysis value, by outcome. "
            f"{CHECK[add]} column: coefficient with CRV1 SE clustered on NSN in parentheses; main column: "
            "coefficient only. NSN + FY fixed effects; the reference period is event year -1. " + SIG_LEGEND)
    cap = f"Table A{num}. {CHECK[add]}: Event-Study Coefficient by Event Year and Outcome (vs Main Analysis)"
    return _wrap(cap, h1 + "\n" + h2, body, note, 1 + 2 * len(OUTCOMES))

def tbl_synth_event_time(add, num):
    head = ('    <tr><th>Outcome</th><th class="num">Event year</th>'
            f'<th class="num">{CHECK[add]} N</th><th class="num">{CHECK[add]} mean</th>'
            '<th class="num">SE</th>'
            '<th class="num">Main N</th><th class="num">Main mean</th></tr>')
    body = []
    for o in OUTCOMES:
        chk = _event_time_agg(_vp(add)["effect_path"], o)
        main = _event_time_agg(MAIN_SYNTH / "effect_path", o)
        if chk is None:
            continue
        mmap = ({} if main is None else
                {int(r["event_year"]): (r["mean"], r["p"], int(r["n"])) for _, r in main.iterrows()})
        post = chk[chk.event_year >= 0].sort_values("event_year")
        first = True
        for _, r in post.iterrows():
            ey = int(r["event_year"])
            olab = OLAB[o].capitalize() if first else ""
            first = False
            se = f'{r["se"]:.3f}' if r["n"] > 1 else "–"
            chk_cell = f'{r["mean"]:+.3f}{_pstars(r["p"])}'
            mm = mmap.get(ey)
            main_mean = "–" if mm is None else f'{mm[0]:+.3f}{_pstars(mm[1])}'
            main_n = "–" if mm is None else str(mm[2])
            body.append(f'    <tr><td>{olab}</td><td class="num">+{ey}</td>'
                        f'<td class="num">{int(r["n"])}</td><td class="num">{chk_cell}</td>'
                        f'<td class="num">{se}</td>'
                        f'<td class="num">{main_n}</td><td class="num">{main_mean}</td></tr>')
    note = (f"Aggregate synthetic-control effect (mean of per-NSN demeaned effects) at each post-waiver event "
            f"year, for the {_samp(add)} beside the main-analysis aggregate, each with its own fit count N. "
            f"SE and significance stars from a one-sample t-test on the per-NSN effects (df = N - 1); both "
            f"mean columns carry stars. {SIG_LEGEND}")
    cap = f"Table A{num}. {CHECK[add]}: Synthetic Controls – Event-Time Path by Year (vs Main Analysis)"
    return _wrap(cap, head, body, note, 7)

def _read_existing_table(add, fname):
    """Pull a <table> already generated by build_artifacts.py and convert em->en dashes."""
    txt = (_vp(add)["tables"] / fname).read_text(encoding="utf-8")
    return re.search(r"<table.*?</table>", txt, re.S).group(0).replace("—", "–")

def _recaption(html, new_caption):
    return re.sub(r"<caption>.*?</caption>", f"<caption>{new_caption}</caption>", html, count=1, flags=re.S)

def tbl_matched_fit(add, num):
    bal_check = _vp(add)["matched_tables"] / "balance_nn_fsg_off.csv"
    main_bal = pd.read_csv(MAIN_MATCHED / "balance_nn_fsg_off.csv")
    if bal_check.exists():                       # DLA-only re-runs matching
        chk = pd.read_csv(bal_check)
    else:                                        # re-aggregated checks: subset main to this check's NSNs
        nsns = pd.read_csv(_vp(add)["matched_tables"] / "nn_did_tau_per_nsn_fsg_off.csv")["treated_nsn"].astype(str).unique()
        chk = main_bal[main_bal["treated_nsn"].astype(str).isin(nsns)]
    def summ(df, stage):
        d = df[(df.stage == stage) & (df.covariate != "distance")]
        smd = d["Std. Mean Diff."].abs()
        return smd.mean(), smd.max(), (smd < 0.1).mean()
    rows_spec = [("Mean |SMD|, before matching", "before", 0),
                 ("Mean |SMD|, after matching", "after", 0),
                 ("Max |SMD|, after matching", "after", 1),
                 ("Share balanced (|SMD| &lt; 0.1), after matching", "after", 2)]
    head = ('    <tr><th>Match-quality statistic</th><th class="num">Main analysis</th>'
            f'<th class="num">{CHECK[add]}</th></tr>')
    ms = {st: summ(main_bal, st) for st in ("before", "after")}
    cs = {st: summ(chk, st) for st in ("before", "after")}
    body = []
    for label, st, idx in rows_spec:
        mv, cv = ms[st][idx], cs[st][idx]
        fmt = (lambda v: f"{v:.0%}") if idx == 2 else (lambda v: f"{v:.3f}")
        body.append(f'    <tr><td>{label}</td><td class="num">{fmt(mv)}</td><td class="num">{fmt(cv)}</td></tr>')
    note = ("Standardized mean differences between each treated NSN and its matched control pool, averaged "
            "over treated NSNs and the four matching covariates (the propensity distance is excluded). Lower "
            "is better balance; |SMD| &lt; 0.1 is the conventional threshold. "
            + ("Matching is re-run on DLA-only data here." if bal_check.exists()
               else "Matching is invariant to the treated set, so these are the main-analysis matches restricted to this check's items."))
    cap = f"Table A{num}. {CHECK[add]}: Matched-Controls Fit Diagnostics (Covariate Balance)"
    return _wrap(cap, head, body, note, 3)


def _tables_for(add, tbl_n):
    out = []
    def take():
        n = tbl_n[0]; tbl_n[0] += 1; return n
    out.append(_recaption(_read_existing_table(add, "T0_comparison_vs_main.html"),
                          f"Table A{take()}. {CHECK[add]}: Robustness Estimates vs the Main Analysis"))
    out.append(tbl_event_study_yearly(add, take()))
    out.append(_recaption(_read_existing_table(add, "T2_matched_controls.html"),
                          f"Table A{take()}. {CHECK[add]}: Matched-Controls ATT by Outcome and Window"))
    out.append(tbl_matched_fit(add, take()))
    out.append(tbl_synth_event_time(add, take()))
    out.append(_recaption(_read_existing_table(add, "T3_synth_controls.html"),
                          f"Table A{take()}. {CHECK[add]}: Synthetic Controls – Per-NSN ATT Distribution"))
    if (_vp(add)["tables"] / "T5_per_item_domestic.html").exists():
        out.append(_recaption(_read_existing_table(add, "T5_per_item_domestic.html"),
                              f"Table A{take()}. {CHECK[add]}: Per-Item Domestic-Sourcing Effect"))
    return out


STYLE = """<style>
  body { font-family: "Times New Roman", Times, serif; color:#222; max-width: 1000px; margin: 24px auto; }
  h2 { font-size: 18px; border-bottom: 1px solid #999; padding-bottom: 4px; margin-top: 34px; }
  table.thesis { border-collapse: collapse; margin: 16px 0; font-size: 14px; font-family: "Times New Roman", Times, serif; color:#222; }
  table.thesis caption { caption-side: top; text-align: left; font-weight: 600; padding: 6px 0; }
  table.thesis th, table.thesis td { padding: 6px 12px; text-align: left; background:#fff; }
  table.thesis th[colspan] { text-align: center; }
  table.thesis thead th { border-bottom: 2px solid #444; }
  table.thesis tbody td { border-bottom: 1px solid #e0e0e0; }
  table.thesis tbody tr:last-child td { border-bottom: 2px solid #444; }
  table.thesis td.num, table.thesis th.num { text-align: right; font-variant-numeric: tabular-nums; }
  table.thesis tfoot td.note { font-size: 12px; color: #555; padding-top: 8px; border:none; }
  hr.tbl { border:none; border-top:1px solid #eee; margin:18px 0; }
</style>"""


def build():
    ART_DIR.mkdir(parents=True, exist_ok=True)
    # ---- Figures A13-A14: the two non-container synth forests ----
    figs = []
    for n, o in [(13, "domestic_share"), (14, "max_log_unit_price")]:
        p = fig_synth_forest("non_container", o, ART_DIR / f"A{n}_synth_forest_{o}.png")
        figs.append((n, p))
    fig_n = 15

    # ---- Tables A7..: one standalone HTML doc ----
    tbl_n = [7]
    sections = []
    for add in VARIANTS:
        sections.append(f"<h2>{CHECK[add]}</h2>\n" + "\n<hr class='tbl'>\n".join(_tables_for(add, tbl_n)))
    intro = (f"<p>Robustness appendix tables A7-A{tbl_n[0]-1}, styled in the thesis convention. Open in a "
             f"browser and copy each table into Google Docs. Figures A13-A14 are the two PNGs alongside "
             f"this file in results/appendix/.</p>")
    doc = (f"<!DOCTYPE html>\n<html lang='en'><head><meta charset='utf-8'>"
           f"<title>Robustness appendix tables (A7-A{tbl_n[0]-1})</title>\n{STYLE}\n</head>\n<body>\n"
           f"{intro}\n" + "\n".join(sections) + "\n</body></html>")
    TABLES_OUT.write_text(doc, encoding="utf-8", newline="\n")

    print(f"figures A13-A{fig_n-1}: " + ", ".join(p.name for _, p in figs))
    print(f"tables  A7-A{tbl_n[0]-1}: {TABLES_OUT}")


if __name__ == "__main__":
    build()
