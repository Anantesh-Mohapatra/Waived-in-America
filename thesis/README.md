# LaTeX build

Turns the Google Docs Markdown export of the thesis into a typeset PDF, with no
edits to the prose. The Google Doc stays canonical; this is a one-way render.

## How to run

1. In Google Docs: **File -> Download -> Markdown (.md)**.
2. Save it over `thesis/Waived in America - Honors Thesis.md` (same name).
3. From the repo root:

   ```
   uv run python thesis/build.py
   ```

Output lands in `thesis/output/`:
- `Waived in America - Honors Thesis.pdf` - the typeset thesis (deliverable)
- `main.tex` - the generated LaTeX (for inspection)
- `clean.md` - the preprocessed Markdown pandoc actually consumed

Takes ~25s. Requires `pandoc` and `xelatex` (both already installed here:
pandoc 3.8.3, TeX Live via TinyTeX). Font: **Latin Modern** (Computer Modern),
text and math, to match the standard LaTeX thesis look.

## What it does (and does not touch)

Everything is keyed off **stable text anchors, caption labels, and table order**,
not export-specific image numbers, so it survives re-exports where numbering
shifts.

- **Prose, footnotes, bibliography, lists** - pandoc converts verbatim. Zotero
  citation hyperlinks (from the Google Docs plugin) are stripped; the visible
  citation text stays.
- **Cover page + Table of Contents** - rebuilt natively (the doc's inline TOC
  and title block are dropped). Cover matches the NYU Stern honors format
  (degree block + advisors at the bottom). Page number only, centered at the
  bottom.
- **Layout** - 1.5 line spacing, a clear one-line gap between paragraphs (no
  first-line indent), tight single-spaced bullet lists, each major (level-1)
  section starts on a new page, and within the appendix each subsection starts
  on a new page. Compiled with three xelatex passes so the TOC populates.
  - Note: Google Docs separates paragraphs with *hard line breaks*, not blank
    lines, so pandoc would otherwise fuse them into one block (no paragraph
    gaps). `split_paragraphs()` inserts a real paragraph break between
    consecutive prose lines (skipping lists/tables/headings/raw-LaTeX).
- **One approved wording change**: in the Acknowledgments,
  "Professor Christopher Conlon" -> "Professor Conlon" (full name is on the
  cover as Thesis Advisor).
- **Equations** - the three Research-Design equations are injected as native
  LaTeX (their images do not survive the export). Source: `equations.md`.
- **Figures** - centered, with centered captions, using the **high-resolution
  originals** from `results/descriptives/figures/` (or `results/appendix/`
  for A13-A14), including Figure A0 (`A0_foreign_manufacture_share.png`).
  Four figures (F1-F4) lose their image on Markdown export but keep their
  caption; those are re-inserted from the source PNGs.
- **Dropped tables, re-inserted.** Several tables were *screenshots* in the doc
  and the Markdown export drops them. They are re-inserted from their generator
  HTML: the "Treated NSN Summary" table (empty heading left behind, see
  `ORPHAN_TABLE_SECTIONS`), and **Table 1 (Sample Funnel)** and **Table 2
  (Matched Controls)**, which leave no caption at all and are placed at stable
  text anchors (see `ANCHORED_TABLES`). NOTE: the `.md` export does not reveal
  these drops — always cross-check figures/tables against the doc's own PDF.
- **Tables** - all rebuilt as native `booktabs` tables, each non-breaking with
  its **title and note bundled in**. The note is set in a box tied to the
  table's displayed width, so it never spills past the margin or runs wider
  than the table; "Significance:" legends break onto their own line. Tables use
  a readable font by column count and are shrunk to fit only if still too wide.
  Two sources:
  - the real Markdown tables in the export (parsed directly), and
  - the 19 robustness-appendix tables (A7-A25), generated from
    `results/appendix/robustness_appendix_tables.html`
    and slotted in document order (replacing the screenshots in the doc).

## Fonts / missing glyphs

Latin Modern Roman lacks several characters the thesis uses (Greek letters in
`alpha_i` etc., subscript letters, `>=`, `~~`, the check mark, `y`-macron).
`GLYPH_MAP` in `build.py` maps each to a LaTeX rendering via `newunicodechar`,
applied globally (Greek -> math italic, subscripts -> `\textsubscript`). The
build runs a coverage check against the font and **warns** if a re-export
introduces a new unmapped glyph (so it never silently prints a box) - when that
happens, add the codepoint to `GLYPH_MAP`. (The check needs `fonttools`; it
skips quietly if that isn't installed.)

## Known limitation (next pass)

The robustness tables' HTML marks a few rows with background shading to flag
sign disagreements; the native conversion drops that shading (the data and
notes are intact). Add row coloring later if it's wanted.

## Editing the layout

Cover page, fonts, margins, running header, spacing, and the glyph map are all
generated from constants near the top and bottom of `build.py` (`PREAMBLE`,
`TITLEPAGE`, `GLYPH_MAP`). Edit there; `thesis/preamble.tex` and
`titlepage.tex` are regenerated each run.
