"""Re-fit the pooled TWFE event study on the robustness variant samples.

The event study is the only estimator that must be re-fit (dropping treated
units changes the joint estimation of the event-time dummies and absorbed FEs).
We reuse the mainline machinery verbatim -- the same panel loaders/filters
(twfe_helpers), the same feols formula + CRV1 vcov (run_spec), and the same
pre-trend / avg-post statistics (export_tables) -- so there is no drift in
specification.

Two modes (--variant):
  reagg (default)  uniform_sample + non_container: enriched panel, one extra
                   filter (keep controls plus the variant's treated NSNs).
  dla_only         the event-study leg of the dla_only variant: DLA-only panel,
                   NO treated-set restriction (overlay-dependent treated units
                   fall out naturally), resumable per-fit cache.

Specs: ols, nsn_fe, nsn_fy_fe (the 3-spec ladder; nsn_fy_fe is the headline
Spec 3). Writes results/<variant>/event_study/event_study.json with, per
outcome, the coefficient structures the thesis figure/table generators expect.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
import variants as C

# Reuse mainline event-study code (no reimplementation).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import twfe_helpers as h  # noqa: E402
import es_stats as ee  # noqa: E402  joint_pretrend_pvalue, avg_post_beta, n_clusters, event_year_of

PANEL = "enriched"
SPECS = [("ols", None), ("nsn_fe", "nsn"), ("nsn_fy_fe", "nsn + fy")]
PLOT_RANGE = range(-4, 4)
# The mainline fits ALL specs on the common sample where both extent covariates
# are recorded (estimate_twfe.py), so the thesis-body Spec-3 numbers are on
# that sample. We hold the same sample construction here so the only difference
# from the body is the restricted treated set.
COMMON_NONNULL_COLS = ["competitive_share", "sole_source_share"]

ES_TITLE = {
    "domestic_share": "Event Study: Domestic Share",
    "max_log_unit_price": "Event Study: Max Log Unit Price",
    "mean_offers": "Event Study: Mean Offers",
}
LADDER_TITLE = {
    "domestic_share": "Domestic Sourcing Share",
    "max_log_unit_price": "Maximum Logged Unit Price",
    "mean_offers": "Mean Offers Received",
}
SUMMARY_OUTCOME = {
    "domestic_share": "Domestic sourcing share",
    "max_log_unit_price": "Maximum logged unit price",
    "mean_offers": "Mean offers",
}


def _avg_post_stars(est: float | None, se: float | None) -> str:
    if est is None or not se or not np.isfinite(se):
        return ""
    p = 2 * (1 - norm.cdf(abs(est / se)))
    return C.stars(p)


def _pretrend_str(p: float | None) -> str:
    if p is None:
        return "-"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def per_ey(model) -> dict[int, tuple[float, float, str]]:
    """{event_year: (coef, se, stars)} for non-sentinel event_year terms."""
    coefs, ses, ps = model.coef(), model.se(), model.pvalue()
    out: dict[int, tuple[float, float, str]] = {}
    for name in coefs.index:
        ey = ee.event_year_of(name)
        if ey is None or ey == h.EVENT_YEAR_SENTINEL:
            continue
        out[ey] = (float(coefs[name]), float(ses[name]), C.stars(float(ps[name])))
    return out


def fit_outcome(panel_sub: pl.DataFrame, outcome: str, cache_dir: Path,
                panel_label: str = PANEL, force_refresh: bool = True) -> dict:
    data = h.prep(panel_sub, outcome, common_sample_cols=COMMON_NONNULL_COLS)
    models = []
    for spec_name, fe in SPECS:
        # force_refresh=False (dla_only) makes the fit resumable: a re-run after
        # an interruption reuses the per-fit cache.
        m = h.run_spec(data, outcome, None,
                       cache_key=f"{outcome}__{panel_label}__{spec_name}",
                       panel_label=panel_label, fe=fe, force_refresh=force_refresh,
                       cache_dir=cache_dir)
        models.append(m)

    per = [per_ey(m) for m in models]            # one dict per spec
    pretrend = [ee.joint_pretrend_pvalue(m) for m in models]
    avgpost = [ee.avg_post_beta(m) for m in models]
    n_obs = [int(getattr(m, "_N", 0)) for m in models]
    n_clu = [ee.n_clusters(m) for m in models]
    r2 = [float(getattr(m, "_r2", float("nan"))) for m in models]

    # n treated NSNs per event year (restricted sample), plot range.
    n_treated = h.compute_n_treated_per_ey(
        panel_sub.filter(pl.col("treated") == 1), list(PLOT_RANGE))

    coef_eys = sorted({ey for d in per for ey in d if ey != -1})
    s3 = per[2]  # nsn_fy_fe == Spec 3

    es_coefs = {
        "title": ES_TITLE[outcome],
        "ylabel": "Coefficient on event-year dummy",
        "n_obs": n_obs[2],
        "n_clusters": n_clu[2],
        "pretrend_p": pretrend[2] if pretrend[2] is not None else 1.0,
        "avg_post": list(avgpost[2]),
        # Include the (-1, 0, 0) reference row so the coefplot can mark it
        # (matches the hardcoded thesis ES_COEFS shape).
        "coefs": ([[ey, s3[ey][0], s3[ey][1]] for ey in sorted(s3) if ey != -1]
                  + [[-1, 0.0, 0.0]]),
        "n_treated_per_ey": [[ey, n] for ey, n in n_treated.items()],
    }

    # 3-spec ladder. coefs only for present non-(-1) EYs; pad a spec with a
    # neutral (0,0,"") entry only if it lacks an EY the others have, so the
    # row still renders.
    def cell(spec_i: int, ey: int):
        return list(per[spec_i].get(ey, (0.0, 0.0, "")))

    ladder = {
        "title": LADDER_TITLE[outcome],
        "coefs": [[ey, [cell(i, ey) for i in range(3)]] for ey in coef_eys],
        "pretrend_p": [[_pretrend_str(p).replace("< 0.001", "0.000"), C.stars(p)]
                       for p in pretrend],
        "avg_post": [[a[0], a[1], _avg_post_stars(*a)] for a in avgpost],
        "n_obs": n_obs,
        "r2": r2,
    }

    summary_row = {
        "outcome": SUMMARY_OUTCOME[outcome],
        "avg_post_beta": avgpost[2][0],
        "avg_post_se": avgpost[2][1],
        "pretrend_p": _pretrend_str(pretrend[2]),
        "n_obs": n_obs[2],
        "n_clusters": n_clu[2],
    }

    present_eys = sorted(set(coef_eys) | {-1})
    return {"es_coefs": es_coefs, "ladder": ladder,
            "summary_row": summary_row, "present_eys": present_eys}


def run_reagg() -> None:
    print("run_event_study_refit: loading enriched panel")
    panel = pl.read_parquet(h.PANEL_PARQUETS[PANEL])
    panel = h.apply_control_filter(panel, PANEL)
    panel, _ = h.apply_pre_window_filter(panel, PANEL)
    panel = panel.with_columns(pl.col("nsn").cast(pl.Utf8))

    for variant in C.REAGG_VARIANTS:
        keep = C.treated_keep_set(variant)
        paths = C.ensure_dirs(variant)
        sub = panel.filter((pl.col("treated") == 0) | pl.col("nsn").is_in(list(keep)))
        n_treated = sub.filter(pl.col("treated") == 1)["nsn"].n_unique()
        print(f"\n{variant}: enriched panel restricted to {n_treated} treated NSNs "
              f"({len(sub):,} cells)")
        result = {}
        for outcome in C.OUTCOMES:
            result[outcome] = fit_outcome(sub, outcome, paths["cache"])
            es = result[outcome]["es_coefs"]
            print(f"  {outcome}: Spec3 avg-post beta="
                  f"{es['avg_post'][0]:+.3f} (se {es['avg_post'][1]:.3f}), "
                  f"pre-trend p={es['pretrend_p']:.3g}, N={es['n_obs']:,}")
        paths["es_json"].write_text(json.dumps(result, indent=2), encoding="utf-8", newline="\n")
        print(f"  wrote {paths['es_json'].relative_to(C.REPO)}")


def run_dla_only() -> None:
    """Event-study leg of the dla_only variant.

    Treated units carry DLA + overlay transaction histories, while the ~947k
    controls are almost entirely DLA-only. This re-fits the event study on the
    DLA-only panel, where the overlay is removed from every unit. Dropping the
    overlay also drops the overlay-dependent treated units (27 -> 23 after the
    standard pre-window filter)."""
    dla_panel = "dla_only"
    print(f"run_event_study_refit: loading {dla_panel} panel")
    panel = pl.read_parquet(h.PANEL_PARQUETS[dla_panel])
    panel = h.apply_control_filter(panel, dla_panel)
    panel, _ = h.apply_pre_window_filter(panel, dla_panel)
    panel = panel.with_columns(pl.col("nsn").cast(pl.Utf8))

    paths = C.ensure_dirs("dla_only")
    n_treated = panel.filter(pl.col("treated") == 1)["nsn"].n_unique()
    print(f"dla_only: DLA-only panel, {n_treated} treated NSNs "
          f"(overlay removed; overlay-dependent units fall out)")

    result = {}
    for outcome in C.OUTCOMES:
        # No treated-set restriction: use every treated unit the DLA-only panel
        # supports. panel_label tags the cache as dla_only. force_refresh=False
        # so an interrupted run resumes from the per-fit cache.
        result[outcome] = fit_outcome(panel, outcome, paths["cache"],
                                      panel_label=dla_panel, force_refresh=False)
        es = result[outcome]["es_coefs"]
        print(f"  {outcome}: Spec3 avg-post beta="
              f"{es['avg_post'][0]:+.3f} (se {es['avg_post'][1]:.3f}), "
              f"pre-trend p={es['pretrend_p']:.3g}, N={es['n_obs']:,}")
    paths["es_json"].write_text(json.dumps(result, indent=2), encoding="utf-8", newline="\n")
    print(f"  wrote {paths['es_json'].relative_to(C.REPO)}")


def run_main() -> None:
    """Emit results/main/event_study/event_study.json for the headline analysis.

    Same enriched panel + filters + 3-spec ladder as the body event study, with
    NO treated-set restriction (the full main sample). Reuses the main fit cache
    written by estimate_twfe (force_refresh=False), so the JSON carries exactly
    the body regression's numbers. This JSON is the single source for the thesis
    figure/ladder/summary generators (build_thesis_{figures,tables}), replacing
    the former hardcoded transcriptions."""
    import paths as P  # lib already on sys.path via the twfe_helpers import
    print("run_event_study_refit: loading enriched panel (main)")
    panel = pl.read_parquet(h.PANEL_PARQUETS[PANEL])
    panel = h.apply_control_filter(panel, PANEL)
    panel, _ = h.apply_pre_window_filter(panel, PANEL)
    panel = panel.with_columns(pl.col("nsn").cast(pl.Utf8))

    out = P.event_study_json("main")
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {}
    for outcome in C.OUTCOMES:
        result[outcome] = fit_outcome(panel, outcome, P.cache_dir("main"),
                                      panel_label=PANEL, force_refresh=False)
        es = result[outcome]["es_coefs"]
        print(f"  {outcome}: Spec3 avg-post beta="
              f"{es['avg_post'][0]:+.3f} (se {es['avg_post'][1]:.3f}), "
              f"pre-trend p={es['pretrend_p']:.3g}, N={es['n_obs']:,}, "
              f"clusters={es['n_clusters']:,}")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8", newline="\n")
    print(f"  wrote {out.relative_to(C.REPO)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=["main", "reagg", "dla_only"], default="reagg",
                    help="main = headline event_study.json; reagg = uniform_sample + "
                         "non_container; dla_only = DLA-only ES leg")
    args = ap.parse_args()
    if args.variant == "main":
        run_main()
    elif args.variant == "dla_only":
        run_dla_only()
    else:
        run_reagg()
