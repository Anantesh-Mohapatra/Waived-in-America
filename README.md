# Waived in America

Replication package for **"Waived in America"**, an honors thesis on the effects of public Buy American Act (BAA) waivers on US federal procurement.

When no domestic supplier can fill a federal order, the buying agency files a nonavailability waiver and buys the foreign-made product. Since 2021 these waivers have been posted on a public portal, madeinamerica.gov, on the premise that procurement transparency can act as a tool of industrial policy: a posted waiver advertises the government's unmet demand, a signal meant to draw domestic manufacturers in to fill the gap.

Three estimators measure what happened to the waived items: a TWFE event study, nearest-neighbor matched controls with a collapsed DiD, and per-item synthetic difference-in-differences. The outcomes are price, competition, and domestic sourcing, over a window running from FY2017 through 2026-01-31.

**The thesis PDF is [`Waived in America - Honors Thesis.pdf`](Waived%20in%20America%20-%20Honors%20Thesis.pdf) at the repository root, and every number in it is reproducible from this repository, from raw data to the final PDF, with one command.**

## Quickstart

```bash
uv sync                              # Python env (uv.lock-pinned)
Rscript install_r_packages.R         # R packages (see r_requirements.md)
uv run python preflight.py           # environment + data checks
uv run python reproduce.py --list    # the full stage list
uv run python reproduce.py           # run everything (two ~13-18h synthdid stages)
```

Large raw data (the archived USAspending snapshot and the DLA FOIA extract) lives outside the repository; the Data section below documents where each comes from.

**Why both Python and R?**

Python (Polars) does the data engineering and fits the TWFE event study (`pyfixest`); the matching and synthetic-DiD estimators run in R, each via its canonical package (`MatchIt` for matching, `synthdid` for synthetic DiD). `reproduce.py` runs both.

## Repository map

| Path | Contents |
|---|---|
| `reproduce.py` / `preflight.py` | one-command pipeline + environment checks |
| `pipeline/01_clean` … `08_artifacts` | the pipeline, in execution order |
| `pipeline/lib/` | shared code; `paths.py` is the single path registry |
| `inputs/` | inputs, committed to the repo |
| `raw_data/` | committed small sources; large snapshots arrive from the data deposit |
| `results/<variant>/` | committed results: estimates, tables, figures, logs |
| `results/descriptives/`, `results/appendix/` | thesis tables/figures and appendix tables A7-A25 |
| `output/` | the cleaned waiver dataset |
| `thesis/` | manuscript source (.md), LaTeX builder (inserts pipeline-generated tables/figures), final PDF |
| `data/` | gitignored regenerables (clean parquets, panels, caches) |
| `data_defs/` | column codebooks for the FPDS and waiver data |

## Requirements

| Requirement | Version verified | Install |
|---|---|---|
| Python | 3.13 | [`uv`](https://docs.astral.sh/uv/) – `uv sync` creates the env from `uv.lock` |
| R | 4.5.1 | [r-project.org](https://www.r-project.org/), then `Rscript install_r_packages.R` (see `r_requirements.md`; `synthdid` is installed from GitHub) |
| TeX | pandoc 3.x + xelatex | e.g. TinyTeX + pandoc; needed for the thesis PDF |
| Disk / CPU | ≥ 80 GB free; 16 cores recommended | regression caches + panels are large |

`preflight.py` checks all of this and the data inventory below.

## Data

Committed in this repository (small):

- `raw_data/procurement-waivers.csv` – waiver exports from the Made in America Office portal (madeinamerica.gov).
- `raw_data/bidlink/` – per-NSN transaction-record exports (BidLink) for the treated items and the 63 control items.
- `inputs/` – inputs: `unit_cost_clin_tagging.csv` (manually validated CLIN tags) and `nsn_reference/` (waiver-NSN reference extracts).

Obtained separately (large; place under `raw_data/`):

| Data | Size | Where |
|---|---|---|
| `raw_data/procurement/FY*_All_Contracts_Full_*.zip` (10 zips) + `api_backfill/*.zip` (1 zip) | ~17 GB | **Archived data deposit** (DOI: *to be added at publication*). A fresh USAspending bulk download may not match exactly: agencies revise FPDS records over time, so the archived snapshot is the reference for an exact reproduction. |
| `raw_data/DLA_FOIA/` (13 zips) | ~0.7 GB | DLA contract-history line items, published in the Defense Logistics Agency's FOIA reading room; also archived in the deposit. |

## The pipeline

`reproduce.py --list` shows all stages. In order:

| Stage | What |
|---|---|
| `parquement` | FPDS bulk zips → `data/clean/procurement_data.parquet` |
| `backfill_replay` | merge the archived API rows into the parquet – **offline replay**; never re-run live (drift) |
| `waivering` | clean the waiver CSV; PIID join → `output/procurement-waivers-cleaned.csv` |
| `bidlink` / `dla_foia` | stack transaction exports; build enriched DLA parquets |
| `treatment_id` | identify the treated NSNs + first-waiver dates |
| `combined_panel` | stack treatment+control BidLink line items + FPDS enrichment |
| `event_panel` / `anchored_panel` | the panel products |
| `event_study` | TWFE fits (cached) + table export |
| `matched` | Mahalanobis 3-NN matching + collapsed DiD, FSG off & on |
| **`synth`** | **HEAVY: ~13-18 h** – per-NSN synthdid fits, 200 placebos each |
| `dla_only_panels` / `dla_only_fast` / **`dla_only_synth`** | the DLA-only variant; the synth leg is **HEAVY: ~13-18 h** |
| `reagg` | uniform_sample + non_container re-aggregations + event-study refits |
| `variant_artifacts` / `appendix` / `descriptives` / `thesis_artifacts` | every figure and table |
| `latex` | the thesis PDF (`thesis/output/`, copied to the repo root) |

**The two HEAVY stages** are resumable (checkpoints under `data/_partial/`) and independent; on a 16-core machine they can run concurrently:

```
terminal 1:  uv run python reproduce.py --to matched
             uv run python reproduce.py --only synth --workers 5
terminal 2:  uv run python reproduce.py --only dla_only_panels,dla_only_fast,dla_only_synth --workers 5
after both:  uv run python reproduce.py --from reagg
```

## Variants

One codebase, four configurations (`pipeline/07_robustness/variants.py`):

| Variant | What it is | How it's computed | Thesis location |
|---|---|---|---|
| `main` | the headline analysis (27 treated NSNs, DLA + transaction-record overlay) | full pipeline | body: Tables 1-4, Figures 1-28 |
| `uniform_sample` | one 22-NSN treated set across all estimators and outcomes | re-aggregation of main per-NSN outputs + event-study refit | appendix A7-A12 ("Common Sample") |
| `non_container` | freight containers (NSN prefix 8150) dropped | re-aggregation + event-study refit | appendix A13-A19 ("Non-Container"), figures A13-A14 |
| `dla_only` | overlay removed; all three estimators re-fit on DLA-only data | full re-fit (`--variant dla_only` / `WIA_VARIANT`) | appendix A20-A25 ("DLA-Only") |

Every variant writes the same tree shape: `results/<variant>/{event_study,matched,synth}/…`.

## Verifying a reproduction

After a full run, stage the tree and check it against the committed results:

```
git add -A && git status --porcelain
```

`git add` of an identical file stages nothing, so anything still listed is a real change.

An exact reproduction leaves the tree clean **except** a known whitelist of embedded non-determinism:

- Wall-clock timestamp and fit-timing columns in the run, filter, and synthdid logs, plus the "generated …" headers in the descriptives stats files
- PDF metadata in the root thesis PDF (the TeX engine stamps `/CreationDate`, `/ModDate`, and `/ID` on every build)
- PNG byte-equality holds only on the same OS + matplotlib version; compare flagged PNGs visually across machines

Everything else should byte-match: every CSV of estimates, every HTML table. All estimation randomness is seeded (`set.seed(20260430)` for matching; per-fit `digest2int` seeds for synthdid placebos; `default_rng(0)` for figures), and committed text outputs use LF on every platform (enforced by `.gitattributes`), so given the archived data snapshot and the verified package versions, byte-exactness is the expectation.
