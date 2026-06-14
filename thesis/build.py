"""Build a LaTeX/PDF thesis from the Google Docs Markdown export.

The Google Doc is the canonical source. Export it as Markdown (.md) into
thesis/, then run this script. It does NOT edit the prose; it only:

  * drops the front-matter (title/subtitle/author) and the inline TOC, which
    are rebuilt natively (cover page + \\tableofcontents);
  * applies ONE author-approved wording change: in the Acknowledgments,
    "Professor Christopher Conlon" -> "Professor Conlon" (his full name now
    appears on the cover as Thesis Advisor);
  * injects the three Research-Design equations as native LaTeX (their images
    do not survive the Markdown export reliably);
  * replaces each embedded figure image with the high-resolution source figure
    from the analysis pipeline;
  * rebuilds every table as a native booktabs table with its title and note
    bundled in (non-breaking): the real Markdown tables in the export, and the
    19 robustness-appendix tables (A7-A25), which are sourced from
    results/appendix/robustness_appendix_tables.html
    (replacing the screenshots that were in the doc, in document order).

Re-run after re-exporting the .md. Everything is keyed off stable text anchors,
caption labels, and table order -- not export-specific image numbers -- so it
survives re-exports where numbering shifts.

    uv run python thesis/build.py

Output: thesis/output/"Waived in America - Honors Thesis.pdf" (working copy),
also copied to the repo root as the committed deliverable.
(+ main.tex, clean.md for inspection).
"""
from __future__ import annotations

import base64
import glob
import re
import shutil
import subprocess
import sys
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString

REPO = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO / "pipeline" / "lib"))
from paths import APPENDIX, DESCRIPTIVES_FIGURES, DESCRIPTIVES_TABLES  # noqa: E402

BUILD = REPO / "thesis"
WORK = BUILD / "build"
IMG_OUT = WORK / "md_images"
OUT = BUILD / "output"

MD_SRC = REPO / "thesis" / "Waived in America - Honors Thesis.md"

FIG_MAIN = DESCRIPTIVES_FIGURES
THESIS_TABLES = DESCRIPTIVES_TABLES
FIG_ROBUST_APPENDIX = APPENDIX
ROBUSTNESS_TABLES_HTML = FIG_ROBUST_APPENDIX / "robustness_appendix_tables.html"

# Body tables that were screenshots in the doc and dropped on Markdown export
# leaving NO caption behind (so they can't be found by heading like the orphan
# tables). Re-inserted from their generator HTML at a stable text anchor.
# (anchor substring, "before"|"after" that line, generator HTML)
ANCHORED_TABLES = [
    ("waiver records yield 37 unique", "before", THESIS_TABLES / "T1_sample_funnel.html"),
    ("Table 2 reports both variants side by side", "after", THESIS_TABLES / "T2_matched_controls.html"),
]

# Six tables that ship as Markdown in the export but are produced by the
# pipeline generators. We re-source them so they can never silently desync from
# the analysis (e.g. the A3 unit-price ladder's standard errors were frozen at a
# pre-CRV1 vintage in the doc). The manuscript caption is kept (correct table
# numbering); the grid + note come from the generator HTML. `columns` selects a
# subset of generator columns by header text -- the fit-diagnostics (Table 3)
# and per-NSN-ATT (A5) tables are two views carved out of the combined 8-column
# synth table, which has no standalone generator.
# (manuscript caption substring, generator HTML, columns | None for all)
SYNTH_COMBINED = THESIS_TABLES / "T3_synth_controls.html"
GENERATOR_TABLES = [
    ("Synthetic-controls fit diagnostics", SYNTH_COMBINED,
     ["Outcome", "N fits", "Median pre-fit RMSPE", "Median active donors / pool size"]),
    ("Per-NSN ATT Distribution by Outcome", SYNTH_COMBINED,
     ["Outcome", "N fits", "Median ATT", "Mean ATT", "SD across NSNs", "Significant (p < 0.05)"]),
    ("Aggregate Event-Time Path by Outcome", THESIS_TABLES / "T4_synth_event_time.html", None),
    ("ladder, domestic sourcing share", THESIS_TABLES / "appendix_event_study_ladder_domestic_share.html", None),
    ("ladder, maximum logged unit price", THESIS_TABLES / "appendix_event_study_ladder_max_log_unit_price.html", None),
    ("ladder, mean offers", THESIS_TABLES / "appendix_event_study_ladder_mean_offers.html", None),
]

# The robustness-appendix TABLE screenshots are the *caption-less* images that
# live in these three appendix subsections (the A13/A14 figures in the same
# region DO have captions, so they're excluded). Identifying them by section/caption
# makes the mapping immune to renumbering when
# figures are added/removed elsewhere. In document order they correspond 1:1 to
# the 19 tables (A7-A25) in the HTML.
ROBUSTNESS_SECTIONS = {
    "Common Treated Sample",
    "Excluding Freight Containers",
    "Using Defense Logistics Agency Data Only",
}
# Figure-caption label: a letter + digits + optional lowercase suffix (F4a, A12).
FIG_LABEL_RE = r"([A-Z]?\d+[a-z]?)"

EQ1 = r"y_{it} = \alpha_i + \gamma_t + \sum_{k \neq -1} \beta_k \mathbb{1}\{e_{it} = k\} + \varepsilon_{it}"
EQ2 = r"\hat{\tau}_i = (\bar{y}^{\,\text{post}}_i - \bar{y}^{\,\text{pre}}_i) - \sum_j w_{ij}\,(\bar{y}^{\,\text{post}}_{ij} - \bar{y}^{\,\text{pre}}_{ij})"
EQ3 = (r"(\hat{\tau}, \hat{\mu}, \hat{\alpha}, \hat{\beta}) = \arg\min_{\tau, \mu, \alpha, \beta} "
       r"\sum_{i=1}^{N} \sum_{t=1}^{T} \left( Y_{it} - \mu - \alpha_i - \beta_t - W_{it}\,\tau \right)^2 "
       r"\hat{\omega}_i\, \hat{\lambda}_t")
EQ2_ANCHOR = "pre-to-post changes:"
EQ3_ANCHOR = "weighted least-squares problem:"

# Widow fixes: paragraphs whose final line spills onto a page of its own.
# \enlargethispage{\baselineskip} is injected before each anchored paragraph,
# giving the page it starts on one extra line so the widow pulls back
# (the same device the Acknowledgments page uses). Page breaks move whenever
# the manuscript is re-exported — re-check for widows after every export and
# update this list; a stale anchor emits a warning.
WIDOW_ANCHORS = [
    "The Biden administration also oversaw the rollout",   # New Waiver System, last line
    "Ohashi (2009), for example",                          # Information Transparency, last line
    "In summary, the robustness checks generally agree",   # body Robustness, last line
]

_DROP = "@@__DROP_LINE__@@"
_STAR = "@@STAR@@"
_USC = "@@USC@@"
CHECK = "✓"
warnings: list[str] = []
_robust_tables: list[str] = []   # native LaTeX for A7-A25, popped in order

# Unicode characters STIX Two Text lacks a glyph for. newunicodechar maps each
# (globally, in body text and tables) to a LaTeX rendering, so none render as a
# missing-glyph box. Keyed by codepoint so the source stays plain ASCII.
GLYPH_MAP = {
    0x2265: r"\ensuremath{\geq}",     # >=
    0x2248: r"\ensuremath{\approx}",  # ~~
    0x2713: r"\ensuremath{\checkmark}",
    0x1D62: r"\textsubscript{i}",     # subscript i (alpha_i etc.)
    0x209C: r"\textsubscript{t}",     # subscript t
    0x2C7C: r"\textsubscript{j}",     # subscript j
    0x2096: r"\textsubscript{k}",     # subscript k
    0x0233: r"\={y}",                 # y with macron
    # Greek letters used as math-variable references in the prose (Latin Modern
    # Roman has no Greek text glyphs; render them as math italic).
    0x03B1: r"\ensuremath{\alpha}",
    0x03B2: r"\ensuremath{\beta}",
    0x03B3: r"\ensuremath{\gamma}",
    0x03B5: r"\ensuremath{\varepsilon}",
    0x03BB: r"\ensuremath{\lambda}",
    0x03BC: r"\ensuremath{\mu}",
    0x03C4: r"\ensuremath{\tau}",
    0x03C9: r"\ensuremath{\omega}",
}


def glyph_defs() -> str:
    return "\n".join(r"\newunicodechar{%s}{%s}" % (chr(cp), tex)
                     for cp, tex in GLYPH_MAP.items())


def check_glyph_coverage(clean: str) -> None:
    """Warn if clean.md contains a non-ASCII char that STIX Two Text lacks and
    GLYPH_MAP does not cover -- i.e. a glyph that would render as a box. Needs
    fonttools; skips quietly if unavailable."""
    try:
        from fontTools.ttLib import TTFont
    except Exception:
        return
    import unicodedata
    found = subprocess.run(["kpsewhich", "lmroman10-regular.otf"],
                           capture_output=True, text=True).stdout.strip()
    if not found or not Path(found).exists():
        return
    cmap = TTFont(found).getBestCmap()
    bad = {ch for ch in set(clean) if ord(ch) > 127
           and ord(ch) not in cmap and ord(ch) not in GLYPH_MAP}
    for ch in sorted(bad):
        warnings.append(f"Unmapped missing glyph U+{ord(ch):04X} "
                        f"({unicodedata.name(ch, '?')}) -- add to GLYPH_MAP.")


# ---------------------------------------------------------------------------
# Low-level text helpers
# ---------------------------------------------------------------------------
def sanitize(text: str) -> str:
    """Normalize line endings; turn the vertical-tab in-cell line breaks Google
    Docs emits into spaces; drop other C0 control characters."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x0b", " ").replace("\x0c", " ")
    return "".join(ch for ch in text if ch >= " " or ch in "\n\t")


def tex_escape_plain(s: str) -> str:
    """Escape LaTeX specials in plain text (from HTML get_text). Unicode passes
    through to xelatex (STIX covers it); check mark -> math \\checkmark."""
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"),
                 ("$", r"\$"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    s = re.sub(r'"([^"]*)"', r"``\1''", s)   # straight double quotes -> LaTeX curly
    return s.replace(CHECK, r"\ensuremath{\checkmark}")


def clean_md_cell(s: str) -> str:
    """Turn a Markdown table cell / title / note into LaTeX. Handles pandoc's
    backslash escapes and * emphasis; cells here carry no % & # $ specials."""
    s = s.strip().rstrip(" ").rstrip("\\")               # drop hard-break trailing slash
    s = s.replace(r"\*", _STAR).replace(r"\_", _USC)     # protect escaped specials
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)     # emphasis on unescaped *
    s = re.sub(r"\*(.+?)\*", r"\\emph{\1}", s)
    s = s.replace(_STAR, "*").replace(_USC, r"\_")       # restore LaTeX-safe
    s = re.sub(r"\\([-+=.()\[\]<>!/|])", r"\1", s)       # unescape safe punctuation
    for ch in "%&#$":                                    # escape any bare LaTeX specials
        s = re.sub(r"(?<!\\)" + re.escape(ch), lambda m, c=ch: "\\" + c, s)
    return s.replace(CHECK, r"\ensuremath{\checkmark}")


# ---------------------------------------------------------------------------
# Native table emission (shared by HTML and Markdown converters)
# ---------------------------------------------------------------------------
def _table_font(ncols: int) -> str:
    """Largest font that keeps a table inside the text width, by column count."""
    return r"\small" if ncols <= 6 else r"\footnotesize"


def emit_table(title: str, colspec: str, header_rows: list[str],
               body_rows: list[str], note: str, ncols: int) -> str:
    """Assemble a non-breaking booktabs table, title + note bundled.

    Fit logic: render at a readable font (by column count); if the tabular is
    still wider than the text block, shrink it to fit (only then). The note is
    set in a minipage tied to the table's *displayed* width, so it never spills
    past the margin and is never wider than the table. "Significance:" legends
    break onto their own line."""
    size = _table_font(ncols)
    H = "\n".join(r + r" \\" for r in header_rows)
    B = "\n".join(r + r" \\" for r in body_rows)
    title_block = f"\\caption*{{\\textbf{{{title}}}}}\n" if title.strip() else ""
    note = re.sub(r"(?<=\S)\s+(Significance:)", r"\\newline \1", note.strip())
    note_block = ""
    if note:
        note_block = (f"\\par\\vspace{{3pt}}\n{{{size}\\begin{{minipage}}{{\\dimen0}}"
                      f"\\raggedright {note}\\end{{minipage}}}}\n")
    return (
        "```{=latex}\n"
        f"\\begin{{table}}[H]\n\\centering\\singlespacing{size}\n\\setlength{{\\tabcolsep}}{{4pt}}\n"
        f"{title_block}"
        "\\sbox0{%\n"
        f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{H}\n\\midrule\n{B}\n"
        "\\bottomrule\n\\end{tabular}}%\n"
        "\\ifdim\\wd0>\\linewidth \\dimen0=\\linewidth \\else \\dimen0=\\wd0 \\fi\n"
        "\\ifdim\\wd0>\\linewidth \\resizebox{\\linewidth}{!}{\\usebox0}\\else \\usebox0\\fi\n"
        f"{note_block}"
        "\\end{table}\n"
        "```"
    )


# ---------------------------------------------------------------------------
# HTML table -> LaTeX (robustness appendix tables)
# ---------------------------------------------------------------------------
def _html_cell(node) -> str:
    """Render an HTML <td>/<th>'s contents to LaTeX (<br>->space, <i>/<code>/<b>
    preserved, <span> styling dropped)."""
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            parts.append(tex_escape_plain(str(child)))
        elif child.name == "br":
            parts.append(" ")
        elif child.name in ("i", "em"):
            parts.append(r"\emph{" + _html_cell(child) + "}")
        elif child.name in ("b", "strong"):
            parts.append(r"\textbf{" + _html_cell(child) + "}")
        elif child.name == "code":
            parts.append(r"\texttt{" + _html_cell(child) + "}")
        else:  # span and anything else: keep text, drop styling
            parts.append(_html_cell(child))
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def html_table_to_latex(t) -> str:
    caption_el = t.find("caption")
    title = tex_escape_plain(caption_el.get_text(" ", strip=True)) if caption_el else ""
    tfoot = t.find("tfoot")
    note = _html_cell(tfoot.find("td")) if (tfoot and tfoot.find("td")) else ""

    placements = []          # (r, c, rowspan, colspan, node, is_header)
    occupied: set = set()
    r = header_rows = 0
    for tag, is_h in (("thead", True), ("tbody", False)):
        sec = t.find(tag)
        if not sec:
            continue
        for tr in sec.find_all("tr", recursive=False):
            c = 0
            for cell in tr.find_all(["td", "th"], recursive=False):
                while (r, c) in occupied:
                    c += 1
                rs, cs = int(cell.get("rowspan", 1)), int(cell.get("colspan", 1))
                placements.append((r, c, rs, cs, cell, is_h))
                for dr in range(rs):
                    for dc in range(cs):
                        occupied.add((r + dr, c + dc))
                c += cs
            r += 1
            if is_h:
                header_rows += 1
    nrows = r
    ncols = max((c + cs for (_, c, _, cs, _, _) in placements), default=1)

    origin = {(pr, pc): (rs, cs, cell) for (pr, pc, rs, cs, cell, _) in placements}
    cover = {}
    for (pr, pc, rs, cs, cell, _) in placements:
        for dr in range(rs):
            for dc in range(cs):
                cover[(pr + dr, pc + dc)] = (pr, pc)

    rowtex: list[str] = []
    for ri in range(nrows):
        parts, c = [], 0
        while c < ncols:
            if cover.get((ri, c)) == (ri, c):           # origin cell
                rs, cs, cell = origin[(ri, c)]
                inner = _html_cell(cell)
                if rs > 1:
                    inner = r"\multirow{%d}{*}{%s}" % (rs, inner)
                parts.append(r"\multicolumn{%d}{c}{%s}" % (cs, inner) if cs > 1 else inner)
                c += cs
            else:                                        # continuation of a rowspan above
                o = cover.get((ri, c))
                cs = origin[o][1] if o else 1
                parts.append(r"\multicolumn{%d}{c}{}" % cs if cs > 1 else "")
                c += cs
        rowtex.append(" & ".join(parts))

    colspec = "l" + "r" * (ncols - 1)
    return emit_table(title, colspec, rowtex[:header_rows], rowtex[header_rows:], note, ncols)


def load_robustness_tables() -> list[str]:
    if not ROBUSTNESS_TABLES_HTML.exists():
        warnings.append(f"Robustness tables HTML not found: {ROBUSTNESS_TABLES_HTML}")
        return []
    soup = BeautifulSoup(ROBUSTNESS_TABLES_HTML.read_text(encoding="utf-8"), "html.parser")
    return [html_table_to_latex(t) for t in soup.find_all("table")]


# ---------------------------------------------------------------------------
# Markdown pipe tables -> LaTeX (title above + notes below, in the export)
# ---------------------------------------------------------------------------
def _md_table_to_latex(title_line: str | None, tbl_lines: list[str],
                       note_lines: list[str]) -> str:
    rows = []
    for ln in tbl_lines:
        cells = re.split(r"(?<!\\)\|", ln.strip())
        if cells and cells[0].strip() == "":
            cells = cells[1:]
        if cells and cells[-1].strip() == "":
            cells = cells[:-1]
        rows.append(cells)
    header = rows[0]
    ncols = len(header)
    body = [(r + [""] * ncols)[:ncols] for r in rows[2:]]  # row 1 is the :--- separator
    colspec = "l" + "r" * (ncols - 1)
    H = " & ".join(clean_md_cell(c) for c in header)
    B = [" & ".join(clean_md_cell(c) for c in r) for r in body]
    title = clean_md_cell(re.sub(r"^\*\*|\*\*$", "", title_line.strip())) if title_line else ""
    note = " ".join(clean_md_cell(re.sub(r"^\*|\*$", "", n.strip())) for n in note_lines)
    return emit_table(title, colspec, [H], B, note, ncols)


def convert_markdown_tables(text: str) -> str:
    lines = text.split("\n")
    is_pipe = lambda s: s.lstrip().startswith("|")
    is_title = lambda s: bool(re.match(r"^\*\*.+\*\*\s*$", s.strip()))
    is_note = lambda s: bool(re.match(r"^\*[^*].*\*\s*$", s.strip()))

    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        if not is_pipe(lines[i]):
            out.append(lines[i])
            i += 1
            continue
        ts = i
        while i < n and is_pipe(lines[i]):
            i += 1
        tbl = lines[ts:i]

        # Title: nearest preceding non-blank line, if it is a bold-only line.
        title = None
        j = len(out) - 1
        while j >= 0 and out[j].strip() == "":
            j -= 1
        if j >= 0 and is_title(out[j]):
            title = out[j]
            out = out[:j] + out[j + 1:]

        # Notes: consecutive italic lines following the table (after blanks).
        k = i
        while k < n and lines[k].strip() == "":
            k += 1
        notes = []
        while k < n and is_note(lines[k]):
            notes.append(lines[k])
            k += 1

        out += ["", _md_table_to_latex(title, tbl, notes), ""]
        i = k
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def resolve_figure(label: str, image_name: str, embedded: dict[str, Path]) -> Path:
    def one(root: Path):
        hits = sorted(glob.glob(str(root / f"{label}_*.png")))
        return Path(hits[0]) if len(hits) == 1 else None

    p = one(FIG_ROBUST_APPENDIX) if label in ("A13", "A14") else one(FIG_MAIN)
    if p:
        return p
    if image_name in embedded:
        warnings.append(f"Figure {label}: no pipeline PNG; using exported image.")
        return embedded[image_name]
    raise FileNotFoundError(f"Cannot resolve figure {label} ({image_name}).")


def tex_path(p: Path) -> str:
    return str(p.resolve()).replace("\\", "/")


def fig_block(src: Path, caption: str | None) -> str:
    inc = (f"\\includegraphics[width=\\linewidth,height=0.42\\textheight,"
           f"keepaspectratio]{{{tex_path(src)}}}")
    cap = f"\n\\caption*{{{tex_escape_plain(caption)}}}" if caption else ""
    return ("```{=latex}\n"
            f"\\begin{{figure}}[htbp]\\centering\\singlespacing\n{inc}{cap}\n"
            "\\end{figure}\n```")


def eq_block(latex: str) -> str:
    return "```{=latex}\n\\[\n" + latex + "\n\\]\n```"


def process_orphan_figures(text: str, embedded: dict[str, Path]) -> str:
    """Some figures (F1-F4) lose their image on Markdown export but keep their
    caption as a standalone *Figure N. ...* line. Re-insert each from its
    high-resolution source PNG. Runs after the image walk, so any caption still
    standing here had no image."""
    out = []
    for ln in text.split("\n"):
        m = re.match(r"^\*?\s*(Figure\s+" + FIG_LABEL_RE + r"\..*)$", ln.strip())
        if m and "![" not in ln and "{=latex}" not in ln:
            cap = m.group(1).strip().rstrip("*").strip()
            try:
                out.append(fig_block(resolve_figure(m.group(2), "", embedded), cap))
                continue
            except Exception:
                warnings.append(f"Orphan figure {m.group(2)}: could not resolve a source PNG.")
        out.append(ln)
    return "\n".join(out)


def strip_zotero_links(text: str) -> str:
    """Drop Google-Docs/Zotero citation hyperlinks, keeping the visible text;
    these point back to the author's Zotero library and must not ship."""
    return re.sub(r"\[([^\]]*)\]\(https?://[^)]*zotero[^)]*\)", r"\1", text)


def split_paragraphs(text: str) -> str:
    """Google Docs exports each paragraph on its own line separated by a hard
    line break (not a blank line), so pandoc fuses consecutive paragraphs into
    one block joined by \\\\ -- and \\parskip never applies. Insert a blank line
    between consecutive prose lines so each becomes a real paragraph. Skips
    headings, lists, tables, blockquotes, footnote defs, and raw-LaTeX blocks
    (figures/tables/equations), and strips the trailing hard-break spaces."""
    def is_prose(s: str) -> bool:
        st = s.strip()
        if not st:
            return False
        if st.startswith(("#", "|", ">", "```", "[^")):
            return False
        if re.match(r"^[-*+]\s", st) or re.match(r"^\d+\.\s", st):
            return False
        return True

    lines = text.split("\n")
    out: list[str] = []
    in_raw = False
    for i, ln in enumerate(lines):
        if ln.strip().startswith("```"):
            in_raw = not in_raw
            out.append(ln)
            continue
        if in_raw:
            out.append(ln)
            continue
        prose = is_prose(ln)
        # Strip leading whitespace too: a prose line indented with a tab / 4+
        # spaces would otherwise become a Markdown code block (verbatim, no
        # wrapping -> overflows the page and truncates).
        out.append(ln.strip() if prose else ln)
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if prose and is_prose(nxt):
            out.append("")
    return "\n".join(out)


# Appendix subsections whose table dropped on Markdown export (heading survives,
# content does not), re-inserted from the generator HTML if the section is empty.
ORPHAN_TABLE_SECTIONS = {
    "Treated NSN Summary": DESCRIPTIVES_TABLES / "TA1_treatment_summary.html",
}


def insert_widow_fixes(text: str) -> str:
    """Insert \\enlargethispage before each WIDOW_ANCHORS paragraph (see the
    constant's comment). The block must precede the paragraph so it is typeset
    on the page where the extra line of room is needed."""
    lines = text.split("\n")
    for anchor in WIDOW_ANCHORS:
        idx = next((i for i, l in enumerate(lines) if anchor in l), None)
        if idx is None:
            warnings.append(f"Widow fix: anchor {anchor!r} not found; not applied.")
            continue
        lines[idx:idx] = ["", "```{=latex}", r"\enlargethispage{\baselineskip}", "```", ""]
    return "\n".join(lines)


def insert_anchored_tables(text: str) -> str:
    """Re-insert body tables whose screenshots dropped on export (no caption
    survived), placing each at its text anchor."""
    lines = text.split("\n")
    for anchor, where, html in ANCHORED_TABLES:
        idx = next((i for i, l in enumerate(lines) if anchor in l), None)
        if idx is None:
            warnings.append(f"Anchored table {html.name}: anchor {anchor!r} not found.")
            continue
        if not html.exists():
            warnings.append(f"Anchored table {html.name}: HTML not found.")
            continue
        t = BeautifulSoup(html.read_text(encoding="utf-8"), "html.parser").find("table")
        if t is None:
            continue
        at = idx if where == "before" else idx + 1
        lines[at:at] = ["", html_table_to_latex(t), ""]
        warnings.append(f"Re-inserted dropped table {html.name} ({where} its anchor).")
    return "\n".join(lines)


def _norm_hdr(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip().lower()


def _generator_grid(html_path: Path, columns: list[str] | None):
    """Parse a generator HTML table -> (header_latex, body_latex_rows, note_latex,
    ncols). With `columns`, keep only those generator columns (by header text)."""
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    t = soup.find("table")
    head_cells = t.find("thead").find_all(["th", "td"])
    raw = [c.get_text(" ", strip=True) for c in head_cells]
    if columns:
        keep = []
        for col in columns:
            idx = next((k for k, h in enumerate(raw) if _norm_hdr(h) == _norm_hdr(col)), None)
            if idx is None:
                raise ValueError(f"{html_path.name}: column {col!r} not in {raw}")
            keep.append(idx)
    else:
        keep = list(range(len(head_cells)))
    header_latex = " & ".join(_html_cell(head_cells[k]) for k in keep)
    body_latex = []
    for tr in t.find("tbody").find_all("tr"):
        cells = tr.find_all(["td", "th"])
        body_latex.append(" & ".join(_html_cell(cells[k]) for k in keep if k < len(cells)))
    tfoot = t.find("tfoot")
    note = _html_cell(tfoot.find("td")) if (tfoot and tfoot.find("td")) else ""
    return header_latex, body_latex, note, len(keep)


def replace_tables_from_generators(text: str) -> str:
    """Swap each Markdown table in GENERATOR_TABLES for its generator HTML: keep
    the manuscript's caption (table numbering), take the grid + note from the
    generator. Runs before convert_markdown_tables so no Markdown table is left
    to double-render. Discards the doc's stale note (the generator's is current,
    and carries the significance legend)."""
    def is_plain_note(s: str) -> bool:
        st = s.strip()
        if not re.match(r"^\*[^*].*\*\s*$", st):
            return False
        return not re.match(r"^(Figure|Table)\b", st.strip("*").strip())

    for cap_sub, html_path, columns in GENERATOR_TABLES:
        lines = text.split("\n")
        cidx = next((i for i, l in enumerate(lines)
                     if l.strip().startswith("**") and cap_sub in l), None)
        if cidx is None:
            warnings.append(f"Generator table {html_path.name}: caption {cap_sub!r} not found.")
            continue
        j = cidx + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines) or not lines[j].lstrip().startswith("|"):
            warnings.append(f"Generator table {html_path.name}: no Markdown table at {cap_sub!r}.")
            continue
        while j < len(lines) and lines[j].lstrip().startswith("|"):
            j += 1
        end = j
        while end < len(lines) and not lines[end].strip():
            end += 1
        while end < len(lines) and is_plain_note(lines[end]):
            end += 1
        if not html_path.exists():
            warnings.append(f"Generator table {html_path.name}: HTML not found.")
            continue
        title = clean_md_cell(re.sub(r"^\*\*|\*\*$", "", lines[cidx].strip()))
        header_latex, body_latex, note, ncols = _generator_grid(html_path, columns)
        colspec = "l" + "r" * (ncols - 1)
        latex = emit_table(title, colspec, [header_latex], body_latex, note, ncols)
        text = "\n".join(lines[:cidx] + ["", latex, ""] + lines[end:])
        warnings.append(f"Sourced table {cap_sub!r} from generator {html_path.name} ({ncols} cols).")
    return text


def process_orphan_tables(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    for i, ln in enumerate(lines):
        out.append(ln)
        m = re.match(r"^##\s+(.*?)(\s*\{#.*\})?\s*$", ln)
        if not m or m.group(1).strip() not in ORPHAN_TABLE_SECTIONS:
            continue
        # Only fill if the subsection is empty (next real content is a heading).
        j = i + 1
        while j < len(lines) and (not lines[j].strip() or re.match(r"^#{1,6}\s*$", lines[j])):
            j += 1
        if j < len(lines) and not lines[j].lstrip().startswith("#"):
            continue  # has content already; leave it
        html = ORPHAN_TABLE_SECTIONS[m.group(1).strip()]
        if not html.exists():
            warnings.append(f"Orphan table for {m.group(1).strip()!r}: HTML not found.")
            continue
        t = BeautifulSoup(html.read_text(encoding="utf-8"), "html.parser").find("table")
        if t is not None:
            out += ["", html_table_to_latex(t)]
            warnings.append(f"Re-inserted dropped table under {m.group(1).strip()!r} "
                            f"from {html.name}.")
    return "\n".join(out)


def render_image(img: str, caption: str | None, embedded: dict[str, Path],
                 section: str = "") -> str:
    if img == "image1":                       # EQ1 (only equation that exports as an image)
        return eq_block(EQ1)
    captioned = bool(caption and caption.startswith(("Figure", "Table")))
    if section in ROBUSTNESS_SECTIONS and not captioned:  # robustness table screenshot
        if _robust_tables:
            return _robust_tables.pop(0)
        warnings.append(f"{img}: ran out of robustness tables to assign.")
        return ""
    if caption and caption.startswith("Figure"):
        m = re.match(r"Figure\s+" + FIG_LABEL_RE + r"\.", caption)
        if not m:
            warnings.append(f"{img}: figure caption without parseable label: {caption[:40]!r}")
            return fig_block(embedded.get(img, IMG_OUT / f"{img}.png"), caption)
        return fig_block(resolve_figure(m.group(1), img, embedded), caption)
    warnings.append(f"{img}: unclassified image (caption={caption!r}); embedding exported copy.")
    return fig_block(embedded[img], caption) if img in embedded else ""


# ---------------------------------------------------------------------------
# Markdown preprocessing
# ---------------------------------------------------------------------------
def extract_embedded_images(md_text: str) -> dict[str, Path]:
    IMG_OUT.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for name, ext, b64 in re.findall(
        r"^\[(image\d+)\]:\s*<data:image/(png|jpeg|jpg|gif);base64,([^>]+)>",
        md_text, re.M,
    ):
        ext = "jpg" if ext == "jpeg" else ext
        p = IMG_OUT / f"{name}.{ext}"
        p.write_bytes(base64.b64decode(b64))
        out[name] = p
    return out


def preprocess(md_text: str, embedded: dict[str, Path]) -> str:
    lines = sanitize(md_text).split("\n")

    start = next((i for i, ln in enumerate(lines)
                  if ln.strip().startswith("# Acknowledgments")), None)
    if start is None:
        sys.exit("Could not find '# Acknowledgments' heading in the export.")
    lines = lines[start:]
    lines = [ln for ln in lines if not re.match(r"^\[image\d+\]:", ln)]
    # Acknowledgments runs ~1 line over a single page; give it imperceptible
    # extra room so the orphan line fits without changing visible spacing.
    lines = [lines[0], "", "```{=latex}",
             r"\enlargethispage{\baselineskip}", "```", ""] + lines[1:]
    text = "\n".join(lines)

    if "Professor Christopher Conlon" not in text:
        warnings.append("Acknowledgments: 'Professor Christopher Conlon' not found to shorten.")
    text = text.replace("Professor Christopher Conlon", "Professor Conlon")

    text = strip_zotero_links(text)

    for anchor, eq, tag in ((EQ2_ANCHOR, EQ2, "EQ2"), (EQ3_ANCHOR, EQ3, "EQ3")):
        out, hit = [], False
        for ln in text.split("\n"):
            out.append(ln)
            if not hit and ln.rstrip().endswith(anchor):
                out += ["", eq_block(eq)]
                hit = True
        if not hit:
            warnings.append(f"{tag}: anchor '{anchor}' not found; equation not inserted.")
        text = "\n".join(out)

    # Images (figures, EQ1, robustness tables). Captions trail on the same line
    # or sit on the next non-empty line.
    lines = text.split("\n")
    out, i, section = [], 0, ""
    fig_re = re.compile(r"!\[\]\[(image\d+)\]")
    head_re = re.compile(r"^#{1,6}\s+(.*?)(\s*\{#.*\})?\s*$")
    while i < len(lines):
        ln = lines[i]
        h = head_re.match(ln)
        if h and h.group(1).strip():
            section = h.group(1).strip()          # track section for robustness-table detection
        refs = fig_re.findall(ln)
        if not refs:
            out.append(ln)
            i += 1
            continue
        residual = fig_re.sub("", ln).replace("*", "").strip()
        skip = False
        if len(refs) == 1:
            caption = residual or None
            if caption is None:
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines):
                    cand = lines[j].strip().strip("*").strip()
                    if re.match(r"(Figure|Table)\b", cand):
                        caption, skip = cand, j
            out.append(render_image(refs[0], caption, embedded, section))
        else:
            for img in refs:
                out.append(render_image(img, None, embedded, section))
        if skip is not False:
            lines[skip] = _DROP
        i += 1
    lines = [ln for ln in out if ln != _DROP]
    text = "\n".join(lines)

    # Figures whose image dropped on export but whose caption survived.
    text = process_orphan_figures(text, embedded)

    # Appendix tables that dropped on export (empty heading left behind).
    text = process_orphan_tables(text)

    # Body tables (T1, T2) whose screenshots dropped on export (no caption left).
    text = insert_anchored_tables(text)

    # Pipeline-generated tables (6): swap the Markdown for generator HTML so they
    # stay in lock-step with the analysis. Must run before convert_markdown_tables.
    text = replace_tables_from_generators(text)

    # Real Markdown tables -> native booktabs tables (must run after images).
    text = convert_markdown_tables(text)

    # One-line widows pulled back onto their paragraph's page.
    text = insert_widow_fixes(text)

    # From the Appendix onward, start each subsection on its own page -- EXCEPT
    # the first one, which sits under the "Appendix" heading (no near-empty
    # divider page). Injected after the Appendix heading so main-body
    # subsections are unaffected.
    inject = ("```{=latex}\n\\let\\appsub\\subsection\n"
              "\\newif\\iffirstappsub \\firstappsubtrue\n"
              "\\renewcommand{\\subsection}{\\iffirstappsub\\firstappsubfalse\\else"
              "\\clearpage\\fi\\appsub}\n```")
    out_lines = []
    for ln in text.split("\n"):
        out_lines.append(ln)
        if re.match(r"^# Appendix\b", ln):
            out_lines += ["", inject]
    text = "\n".join(out_lines)

    # Turn hard-break-separated prose lines into real paragraphs (so \parskip
    # actually creates inter-paragraph gaps).
    text = split_paragraphs(text)

    # Drop empty headings that would otherwise become empty sections.
    lines = [ln for ln in text.split("\n") if not re.match(r"^#{1,6}\s*$", ln)]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Preamble + title page (generated here so all config lives in one file)
# ---------------------------------------------------------------------------
PREAMBLE = r"""
% --- thesis preamble (generated by thesis/build.py) ---
\usepackage{fancyhdr}
\usepackage{float}
\usepackage{caption}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{multirow}
\usepackage{threeparttable}
\usepackage{enumitem}
\usepackage{setspace}
\usepackage{newunicodechar}  % map glyphs the text font lacks (see GLYPH_MAP)
\setstretch{1.5}             % 1.5 line spacing (Word-style)
% Block paragraphs: no indent, with a clear gap between paragraphs (~one line).
% (Re-asserted after the front matter in TITLEPAGE, since parskip.sty and the
% title page's local settings otherwise clobber these.)
\setlength{\parindent}{0pt}
\setlength{\parskip}{10pt plus 2pt minus 1pt}
% Tight, single-spaced bullet lists (override the document's double spacing).
\setlist{topsep=4pt, itemsep=3pt, parsep=0pt, before=\singlespacing}
% Major (level-1) sections start on a new page.
\let\oldsection\section
\renewcommand{\section}{\clearpage\oldsection}
% Page number only, centered at the bottom; no header.
\pagestyle{fancy}
\fancyhf{}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}
\captionsetup{font=small,justification=raggedright,singlelinecheck=false}
\captionsetup[figure]{justification=centering,singlelinecheck=false}
% Let lines stretch a little to avoid long URLs/strings spilling past the margin.
\setlength{\emergencystretch}{3em}
"""

TITLEPAGE = r"""
\begin{titlepage}
\centering
\thispagestyle{empty}
\singlespacing
\setlength{\parskip}{0pt}
\vspace*{1.1in}
{\LARGE\bfseries Waived in America\par}
\vspace{0.3in}
{\large Procurement Transparency As Industrial Policy\par}
\vspace{0.8in}
{Anantesh Mohapatra\par}
\vspace{0.5in}
{May 2026\par}
\vspace{0.8in}
{An honors thesis submitted in partial fulfillment of the requirements of the degree:\par}
\vspace{0.2in}
{Bachelor of Science\par}
\vspace{2pt}{Undergraduate College\par}
\vspace{2pt}{Leonard N. Stern School of Business\par}
\vspace{2pt}{New York University\par}
\vfill
\begin{flushleft}
\textbf{Thesis Advisor:}\quad Professor Christopher Conlon\\[6pt]
\textbf{Faculty Advisor:}\quad Professor Lawrence White
\end{flushleft}
\end{titlepage}
\clearpage
\pagenumbering{roman}
\tableofcontents
\clearpage
\pagenumbering{arabic}
% Lock in body spacing here, after all front matter (nothing downstream resets it).
\setstretch{1.5}
\setlength{\parindent}{0pt}
\setlength{\parskip}{10pt plus 2pt minus 1pt}
"""


def main() -> None:
    if not MD_SRC.exists():
        sys.exit(f"Markdown export not found: {MD_SRC}")
    for d in (WORK, OUT):
        d.mkdir(parents=True, exist_ok=True)

    md_text = MD_SRC.read_text(encoding="utf-8")
    embedded = extract_embedded_images(md_text)
    print(f"Extracted {len(embedded)} embedded images.")

    global _robust_tables
    _robust_tables = load_robustness_tables()
    print(f"Loaded {len(_robust_tables)} robustness appendix tables.")

    clean = preprocess(md_text, embedded)
    clean_md = OUT / "clean.md"
    clean_md.write_text(clean, encoding="utf-8")
    check_glyph_coverage(clean)

    # Tripwire: count figures/tables placed and warn loudly if it deviates from
    # the known-good set, so a silently dropped figure/table can't slip past.
    n_fig = clean.count("\\includegraphics")
    n_tab = clean.count("\\begin{table}")
    EXPECTED_FIG, EXPECTED_TAB = 28, 28
    print(f"Placed: {n_fig} figures, {n_tab} tables (expected {EXPECTED_FIG}, {EXPECTED_TAB}).")
    if (n_fig, n_tab) != (EXPECTED_FIG, EXPECTED_TAB):
        warnings.append(f"FIGURE/TABLE COUNT CHANGED: {n_fig} figures, {n_tab} tables "
                        f"(expected {EXPECTED_FIG}, {EXPECTED_TAB}). Verify nothing was "
                        f"dropped; if intentional, update EXPECTED_FIG/EXPECTED_TAB.")

    if _robust_tables:
        warnings.append(f"{len(_robust_tables)} robustness tables were not placed.")
    remaining = len(re.findall(r"!\[\]\[image\d+\]", clean))
    if remaining:
        warnings.append(f"{remaining} image refs left unreplaced (will break pandoc).")

    (BUILD / "preamble.tex").write_text(PREAMBLE + "\n" + glyph_defs() + "\n", encoding="utf-8")
    (BUILD / "titlepage.tex").write_text(TITLEPAGE, encoding="utf-8")

    pandoc = [
        "pandoc", str(clean_md), "-f", "markdown", "-s",
        "--pdf-engine=xelatex",
        "-V", "documentclass=article", "-V", "fontsize=12pt",
        "-V", "geometry:margin=1in",
        "-V", "mainfont=lmroman10",
        "-V", ("mainfontoptions=Extension=.otf,UprightFont=*-regular,"
               "BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic"),
        "-V", "mathfont=latinmodern-math.otf",
        "-V", "colorlinks=true", "-V", "linkcolor=black", "-V", "urlcolor=blue",
        f"--include-in-header={BUILD / 'preamble.tex'}",
        f"--include-before-body={BUILD / 'titlepage.tex'}",
    ]

    print("Running pandoc -> main.tex ...")
    subprocess.run(pandoc + ["-t", "latex", "-o", str(OUT / "main.tex")], check=True)
    # Compile directly (self-contained .tex): three xelatex passes so the table
    # of contents and any references resolve. pandoc only runs one pass, which
    # leaves the TOC empty on a clean build.
    print("Compiling with xelatex (3 passes) ...")
    r = None
    for _ in range(3):
        r = subprocess.run(["xelatex", "-interaction=nonstopmode", "main.tex"],
                           cwd=OUT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
    if r is not None and r.returncode != 0:
        print((r.stdout or "")[-4000:])

    if warnings:
        print("\n=== WARNINGS ===")
        for w in warnings:
            print(" -", w)

    # Deliverable PDF gets the thesis title; main.tex/clean.md stay for inspection.
    built = OUT / "main.pdf"
    pdf = OUT / "Waived in America - Honors Thesis.pdf"
    if built.exists():
        shutil.move(str(built), str(pdf))
    ok = pdf.exists()
    if ok:
        # Also place the deliverable at the repo root so a visitor to the
        # repository finds the thesis immediately (thesis/output/ is a
        # gitignored working dir; the root copy is the committed deliverable).
        root_pdf = REPO / "Waived in America - Honors Thesis.pdf"
        shutil.copy2(str(pdf), str(root_pdf))
        print(f"  copied to repo root: {root_pdf.name}")
    print(f"\n{'OK' if ok else 'FAILED'}: {pdf}"
          + (f" ({pdf.stat().st_size // 1024} KB)" if pdf.exists() else ""))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
