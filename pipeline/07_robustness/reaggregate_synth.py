"""Re-aggregate synthdid onto the robustness samples (no re-fitting).

Each synthdid fit is per treated NSN, independent of the rest of the treated
set, so restricting the sample is a pure re-aggregation of the existing per-fit
synth_att.csv plus the per-fit effect_path CSVs. A self-check reproduces the
published synth_summary.csv from the full synth_att.csv before any filtering,
proving re-aggregation equals a (13-hour) re-run.

For BOTH re-aggregation variants we filter synth_att to the variant's treated
keep-set and recompute the summary:
  * uniform_sample (22 NSNs): drops 1680 from the domestic_share fits too, so
    all three estimators sit on the same 22-NSN set.
  * non_container: keeps 1680 where synth has it (domestic_share only);
    price/offers are the 3 non-container fits automatically.

Writes, per variant, into results/<variant>/synth/tables/:
  synth_att.csv            filtered per-fit rows
  synth_summary.csv        recomputed (mainline aggregate.R schema)
  effect_path/*.csv        copies of the kept NSNs' effect paths
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import variants as C

SUMMARY_COLS = ["outcome", "n_fits", "mean_att", "median_att", "sd_att",
                "n_asymp_p_lt_05", "n_empirical_p_lt_05", "median_pre_rmspe",
                "q25_pre_rmspe", "q75_pre_rmspe", "median_n_active",
                "median_weight_hhi", "median_balanced_pool"]


def summarize(att_ok: pl.DataFrame) -> pl.DataFrame:
    """Reproduce aggregate.R's per-outcome summary."""
    rows = []
    for (outcome,), g in att_ok.group_by(["outcome"]):
        a = g["att"].to_numpy()
        rows.append({
            "outcome": outcome,
            "n_fits": len(a),
            "mean_att": float(a.mean()),
            "median_att": float(np.median(a)),
            "sd_att": float(a.std(ddof=1)) if len(a) > 1 else float("nan"),
            "n_asymp_p_lt_05": int((g["asymp_p"] < 0.05).sum()),
            "n_empirical_p_lt_05": int((g["empirical_p"] < 0.05).sum()),
            "median_pre_rmspe": float(np.median(g["pre_rmspe"].to_numpy())),
            "q25_pre_rmspe": float(np.quantile(g["pre_rmspe"].to_numpy(), 0.25)),
            "q75_pre_rmspe": float(np.quantile(g["pre_rmspe"].to_numpy(), 0.75)),
            "median_n_active": float(np.median(g["n_active"].to_numpy())),
            "median_weight_hhi": float(np.median(g["weight_hhi"].to_numpy())),
            "median_balanced_pool": float(np.median(g["n_balanced_pool"].to_numpy())),
        })
    return pl.DataFrame(rows).select(SUMMARY_COLS).sort("outcome")


def _ok(att: pl.DataFrame) -> pl.DataFrame:
    return att.filter(pl.col("error").is_null() | (pl.col("error") == "NA"))


def self_check(att: pl.DataFrame, tol: float = 1e-9) -> None:
    got = summarize(_ok(att))
    pub = pl.read_csv(C.SYNTH_SUMMARY).sort("outcome")
    j = got.join(pub, on="outcome", suffix="_pub")
    worst = 0.0
    for col in SUMMARY_COLS[1:]:
        d = (j[col].cast(pl.Float64) - j[f"{col}_pub"].cast(pl.Float64)).abs().max()
        worst = max(worst, float(d))
    if worst > tol:
        raise SystemExit(
            f"[FAIL] synth re-aggregation does not reproduce synth_summary.csv "
            f"(max abs diff {worst:.3e} > {tol:.0e}); stop."
        )
    print(f"  [self-check] synth: reproduces synth_summary.csv "
          f"(max abs diff {worst:.2e})")


def run() -> None:
    print("reaggregate_synth: loading synth_att.csv")
    att = pl.read_csv(C.SYNTH_ATT, schema_overrides={"treated_nsn": pl.Utf8})
    self_check(att)

    att_ok = _ok(att)
    for variant in C.REAGG_VARIANTS:
        keep = C.treated_keep_set(variant)
        paths = C.ensure_dirs(variant)
        sub = att_ok.filter(pl.col("treated_nsn").is_in(list(keep)))
        sub.write_csv(paths["synth_tables"] / "synth_att.csv")
        summarize(sub).write_csv(paths["synth_tables"] / "synth_summary.csv")

        # Copy effect paths for the kept NSNs (used by event-time table/figure).
        for f in paths["effect_path"].glob("*.csv"):
            f.unlink()
        copied = 0
        for src in sorted(C.SYNTH_EFFECT_PATH.glob("*.csv")):
            nsn = src.stem.split("__")[0]
            if nsn in keep:
                shutil.copy(src, paths["effect_path"] / src.name)
                copied += 1
        counts = {o: sub.filter(pl.col("outcome") == o).height for o in C.OUTCOMES}
        print(f"  {variant}: synth re-aggregated, fits per outcome={counts}, "
              f"{copied} effect-path files copied")


if __name__ == "__main__":
    run()
