"""Re-aggregate matched-controls (NN-DiD Path 1) onto the robustness samples.

The pooled ATT is ATT = mean(tau_i), SE = sd(tau_i, ddof=1)/sqrt(n) (the HC1 SE
of a constant-only model coincides with this), t = ATT/SE, CI = ATT +- t(.975, n-1)*SE.
Each tau_i is estimated per treated NSN against a donor pool that excludes all
waived NSNs, so it does not depend on the treated-set composition; filtering the
per-NSN tau file and recomputing is therefore identical to re-running matchit on
the restricted sample. A self-check proves this by reproducing the published
summaries from the full tau files before any filtering.

Writes, per re-aggregation variant, into results/<variant>/matched/tables/:
  nn_did_tau_per_nsn_fsg_{off,on}.csv   filtered per-NSN tau
  nn_did_summary_fsg_{off,on}.csv       recomputed pooled ATT (mainline schema)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats as _stats

sys.path.insert(0, str(Path(__file__).parent))
import variants as C

FSG_VARIANTS = ("off", "on")
SUMMARY_COLS = ["outcome", "post_label", "post_threshold", "n_treated",
                "att", "se", "ci_low", "ci_high", "t_stat"]


def aggregate(tau: pl.DataFrame) -> pl.DataFrame:
    """Collapse per-NSN tau to the pooled summary (mainline schema)."""
    rows = []
    for (outcome, post_label), g in tau.group_by(["outcome", "post_label"]):
        t = g["tau"].to_numpy()
        n = len(t)
        att = float(t.mean())
        se = float(t.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        tcrit = float(_stats.t.ppf(0.975, df=n - 1)) if n > 1 else float("nan")
        rows.append({
            "outcome": outcome,
            "post_label": post_label,
            "post_threshold": int(g["post_threshold"][0]),
            "n_treated": n,
            "att": att,
            "se": se,
            "ci_low": att - tcrit * se,
            "ci_high": att + tcrit * se,
            "t_stat": att / se if se and np.isfinite(se) else float("nan"),
        })
    return pl.DataFrame(rows).select(SUMMARY_COLS).sort(["post_label", "outcome"])


def self_check(fsg: str, tau: pl.DataFrame, tol: float = 1e-9) -> None:
    """Aggregating the FULL tau file must reproduce the published summary."""
    got = aggregate(tau)
    pub = (pl.read_csv(C.MATCHED_TABLES / f"nn_did_summary_fsg_{fsg}.csv")
           .sort(["post_label", "outcome"]))
    j = got.join(pub, on=["outcome", "post_label"], suffix="_pub")
    worst = 0.0
    for col in ["n_treated", "att", "se", "ci_low", "ci_high", "t_stat"]:
        d = (j[col].cast(pl.Float64) - j[f"{col}_pub"].cast(pl.Float64)).abs().max()
        worst = max(worst, float(d))
    if worst > tol:
        raise SystemExit(
            f"[FAIL] matched fsg_{fsg} re-aggregation does not reproduce the "
            f"published summary (max abs diff {worst:.3e} > {tol:.0e}). "
            f"The re-aggregation-equals-re-run assumption is violated; stop."
        )
    print(f"  [self-check] matched fsg_{fsg}: reproduces published summary "
          f"(max abs diff {worst:.2e})")


def run() -> None:
    print("reaggregate_matched: loading per-NSN tau files")
    taus = {
        fsg: pl.read_csv(C.MATCHED_TABLES / f"nn_did_tau_per_nsn_fsg_{fsg}.csv",
                         schema_overrides={"treated_nsn": pl.Utf8})
        for fsg in FSG_VARIANTS
    }
    for fsg in FSG_VARIANTS:
        self_check(fsg, taus[fsg])

    for variant in C.REAGG_VARIANTS:
        keep = C.treated_keep_set(variant)
        paths = C.ensure_dirs(variant)
        for fsg in FSG_VARIANTS:
            sub = taus[fsg].filter(pl.col("treated_nsn").is_in(list(keep)))
            sub.write_csv(paths["matched_tables"] / f"nn_did_tau_per_nsn_fsg_{fsg}.csv")
            aggregate(sub).write_csv(paths["matched_tables"] / f"nn_did_summary_fsg_{fsg}.csv")
        n_kept = sub["treated_nsn"].n_unique()
        print(f"  {variant}: matched re-aggregated "
              f"({n_kept} treated NSNs present in fsg_on)")


if __name__ == "__main__":
    run()
