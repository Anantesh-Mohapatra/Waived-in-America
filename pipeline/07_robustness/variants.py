"""Variant configuration for the robustness analyses — the single source of
truth for what each variant means and where its artifacts live.

Variants:
  main           The headline analysis (DLA + transaction overlay, 28 treated).
  dla_only       Full RE-FIT with the transaction overlay removed: the panel
                 stages and all three estimators run again on DLA-only data
                 (WIA_VARIANT=dla_only / --variant dla_only on the main
                 pipeline scripts). Nothing in this module recomputes it.
  uniform_sample RE-AGGREGATION of main per-NSN outputs on synth's most-
                 restrictive treated set. Because matched controls cannot
                 match NSN 1680016229189 (no computable matching covariates),
                 a genuinely uniform sample must exclude it. Synth's
                 domestic_share set minus 1680 is identical to its price and
                 offers sets, giving a single 22-NSN set applied to all three
                 estimators and all three outcomes. Event study is re-fit
                 (pooled regression; cannot be re-aggregated).
  non_container  RE-AGGREGATION dropping freight-container NSNs (13-digit form
                 starting "8150"). Each estimator keeps whatever of the
                 remaining NSNs it already fits (1680 stays where an estimator
                 naturally has it). Event study re-fit, as above.

Key fact exploited by the re-aggregation variants: each treated NSN's effect
is estimated independently of which other treated NSNs are in the sample
(synth fits per-NSN; matched controls matches each treated NSN against a
donor pool that excludes all waived NSNs, so it is invariant to treated-set
composition). Restricting the treated set is therefore a pure re-aggregation
of existing per-NSN outputs — proven equal to a re-run by the self-checks in
reaggregate_matched.py / reaggregate_synth.py.

Every variant's results tree mirrors main's estimator layout
(results/<variant>/{event_study,matched,synth}/...); the robustness variants
additionally get figures/ + tables/ built by build_artifacts.py (main's
thesis-facing figures and tables live in results/descriptives/).
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import paths as P

REPO = P.REPO_ROOT

# The two re-aggregation variants this package computes.
REAGG_VARIANTS = ("uniform_sample", "non_container")

# Reader-facing labels (used in emitted table captions — keep stable).
VARIANT_LABEL = {
    "uniform_sample": "Common Sample",
    "non_container": "Non-Container",
    "dla_only": "DLA-Only",
}

# --- mainline inputs (read-only) ---------------------------------------------
SYNTH_ATT = P.synth_tables("main") / "synth_att.csv"
SYNTH_SUMMARY = P.synth_tables("main") / "synth_summary.csv"
SYNTH_EFFECT_PATH = P.synth_tables("main") / "effect_path"
MATCHED_TABLES = P.matched_tables("main")
TREATED_DATES = P.TREATMENT_DATES

OUTCOMES = ("domestic_share", "max_log_unit_price", "mean_offers")
CONTAINER_PREFIX = "8150"


def variant_paths(variant: str) -> dict[str, Path]:
    """Artifact locations for a variant (layout owned by paths.py)."""
    base = P.results_root(variant)
    return {
        "matched_tables": P.matched_tables(variant),
        "synth_tables": P.synth_tables(variant),
        "effect_path": P.synth_tables(variant) / "effect_path",
        "es_json": P.event_study_json(variant),
        "figures": base / "figures",
        "tables": base / "tables",
        "cache": P.cache_dir(variant),
    }


def ensure_dirs(variant: str) -> dict[str, Path]:
    p = variant_paths(variant)
    for key in ("matched_tables", "synth_tables", "effect_path", "figures", "tables", "cache"):
        p[key].mkdir(parents=True, exist_ok=True)
    p["es_json"].parent.mkdir(parents=True, exist_ok=True)
    return p


# --- treated-set definitions --------------------------------------------------
def _nodash(nsn: str) -> str:
    return nsn.replace("-", "")


def treated_nsns() -> set[str]:
    """All 28 treated NSNs in 13-digit no-dash form."""
    fw = pl.read_csv(TREATED_DATES, infer_schema_length=0)
    return {_nodash(n) for n in fw["nsn"].to_list()}


def is_container(nsn: str) -> bool:
    return _nodash(str(nsn)).startswith(CONTAINER_PREFIX)


def synth_fitted_sets() -> dict[str, set[str]]:
    """{outcome: set of treated NSNs synth produced a successful fit for}."""
    att = pl.read_csv(SYNTH_ATT, schema_overrides={"treated_nsn": pl.Utf8})
    att = att.filter(pl.col("error").is_null() | (pl.col("error") == "NA"))
    return {
        o: set(att.filter(pl.col("outcome") == o)["treated_nsn"].to_list())
        for o in OUTCOMES
    }


def uniform22_set() -> set[str]:
    """uniform_sample's single treated set: synth's price set, which equals
    synth's domestic_share set minus 1680 and its offers set. Asserted here so
    the build fails loudly if that equality ever stops holding."""
    s = synth_fitted_sets()
    price = s["max_log_unit_price"]
    offers = s["mean_offers"]
    dom_minus_1680 = {n for n in s["domestic_share"] if n != "1680016229189"}
    assert price == offers == dom_minus_1680, (
        "uniform_sample assumption broken: synth price/offers/(domestic-1680) "
        f"sets differ. price={sorted(price)} offers={sorted(offers)} "
        f"dom-1680={sorted(dom_minus_1680)}"
    )
    return set(price)


def treated_keep_set(variant: str) -> set[str]:
    """Treated NSNs to KEEP for a re-aggregation variant (outcome-independent)."""
    if variant == "uniform_sample":
        return uniform22_set()
    if variant == "non_container":
        return {n for n in treated_nsns() if not is_container(n)}
    raise ValueError(f"not a re-aggregation variant: {variant}")


# --- significance stars (match the mainline generators exactly) ---------------
def stars(p: float | None) -> str:
    """*** p<0.01, ** p<0.05, * p<0.10 -- identical thresholds to
    export_tables.py / build_thesis_tables.py."""
    if p is None:
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""
