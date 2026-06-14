"""reproduce.py — runs the full thesis pipeline end to end in one command.

    uv run python reproduce.py --list            # show all stages
    uv run python reproduce.py                   # run everything, in order
    uv run python reproduce.py --to matched      # cleaning through matched controls
    uv run python reproduce.py --only synth --workers 5
    uv run python reproduce.py --from variant_artifacts

Stages run in dependency order (see README.md for the full DAG and
runtimes). Two stages are marked HEAVY — `synth` and `dla_only_synth` are
~13-18 hour synthdid fits (resumable; checkpoints under data/_partial/).
They can run concurrently in two terminals on a >=16-core machine:

    terminal 1:  uv run python reproduce.py --only synth --workers 5
    terminal 2:  uv run python reproduce.py --only dla_only_panels,dla_only_fast,dla_only_synth --workers 5

The `backfill_replay` stage is an OFFLINE replay of the archived USAspending
API pull (raw_data/procurement/api_backfill/). It must run: the bulk FY zips
alone are missing ~169K recent DLA rows that the archived pull supplies.
Never re-run the backfill live — FPDS records have been revised since the
thesis snapshot and a fresh pull would change every downstream number.

Failure stops the run and prints the resume command. Per-stage wall times are
appended to run_log.csv (gitignored).
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pipeline" / "lib"))
sys.path.insert(0, str(ROOT / "pipeline" / "07_robustness"))
import paths as P  # noqa: E402
import variants as V  # noqa: E402

PY = sys.executable
RUN_LOG = ROOT / "run_log.csv"


@dataclass
class Stage:
    key: str
    desc: str
    cmds: list[list[str]]
    env: dict[str, str] = field(default_factory=dict)
    needs: list[Path] = field(default_factory=list)
    heavy: bool = False
    note: str = ""


def _py(rel: str, *args: str) -> list[str]:
    return [PY, str(ROOT / rel), *args]


def _r(rel: str, *args: str) -> list[str]:
    return ["Rscript", str(ROOT / rel), *args]


def _matched_cmds(variant: str) -> list[list[str]]:
    """Covariates, then matching + collapsed DiD for both FSG settings."""
    return [
        _py("pipeline/05_matched/compute_covariates.py", "--variant", variant),
        _r("pipeline/05_matched/match_nn.R", "--fsg=off"),
        _r("pipeline/05_matched/match_nn.R", "--fsg=on"),
        _r("pipeline/05_matched/estimate_did.R", "--fsg=off"),
        _r("pipeline/05_matched/estimate_did.R", "--fsg=on"),
    ]


def build_stages(workers: int) -> list[Stage]:
    w = f"--workers={workers}"
    backfill_zips = sorted(P.RAW_API_BACKFILL.glob("*.zip"))

    return [
        Stage("parquement", "FPDS bulk zips -> procurement parquet",
              [_py("pipeline/01_clean/parquement.py")],
              needs=[P.RAW_PROCUREMENT]),
        Stage("backfill_replay", "merge archived API rows into the parquet (OFFLINE)",
              [_py("pipeline/01_clean/usaspending_backfill.py", "--reuse-cached-zip",
                   str(backfill_zips[0]) if backfill_zips else "MISSING")],
              needs=[P.PROCUREMENT_PARQUET],
              note=("OFFLINE REPLAY of the archived API pull — authoritative for "
                    "reproduction. Never re-run live: FPDS has revised records "
                    "since the thesis snapshot; results would drift.")),
        Stage("waivering", "clean waiver portal CSV (+ PIID join)",
              [_py("pipeline/01_clean/waivering.py")],
              needs=[P.RAW_WAIVERS, P.PROCUREMENT_PARQUET]),
        Stage("bidlink", "stack per-NSN transaction exports",
              [_py("pipeline/01_clean/bidlink.py")],
              needs=[P.RAW_BIDLINK]),
        Stage("dla_foia", "DLA FOIA zips -> enriched DLA parquets",
              [_py("pipeline/01_clean/dla_foia.py")],
              needs=[P.RAW_DLA_FOIA, P.PROCUREMENT_PARQUET]),
        Stage("treatment_id", "identify 28 treated NSNs + first-waiver dates",
              [_py("pipeline/02_treatment/identify_treatment_nsns.py")],
              needs=[P.PROCUREMENT_PARQUET, P.RAW_BIDLINK, P.NSN_REFERENCE,
                     P.UNIT_COST_CLIN_TAGGING]),
        Stage("combined_panel", "treatment + control BidLink panel + FPDS enrichment",
              [_py("pipeline/03_panels/build_combined_bidlink.py")],
              needs=[P.BIDLINK_NSN_CSV, P.PROCUREMENT_PARQUET, P.TREATMENT_DATES,
                     P.DLA_ENRICHED_LATEST, P.RAW_BIDLINK]),
        Stage("event_panel", "NSN x FY event-study panels (enriched + dla_only)",
              [_py("pipeline/03_panels/build_event_panel.py")],
              needs=[P.DLA_ENRICHED_LATEST, P.COMBINED_BIDLINK_PANEL, P.TREATMENT_DATES,
                     P.RAW_BIDLINK]),
        Stage("donor_universe", "donor universe (main; read by matched + synth)",
              [_py("pipeline/06_synth/build_donor_universe.py", "--variant", "main")],
              needs=[P.PANEL_ENRICHED, P.TREATMENT_DATES, P.NSN_REFERENCE, P.WAIVERS_CLEANED]),
        Stage("anchored_panel", "per-treated anchored panel (main variant)",
              [_py("pipeline/03_panels/build_anchored_panel.py", "--variant", "main")],
              needs=[P.donor_universe_path("main"), P.DLA_ENRICHED_LATEST,
                                P.COMBINED_BIDLINK_PANEL, P.TREATMENT_DATES,
                                P.PROCUREMENT_PARQUET, P.RAW_BIDLINK,
                                P.UNIT_COST_CLIN_TAGGING]),
        Stage("event_study", "TWFE fits + table export + main event_study.json",
              [_py("pipeline/04_event_study/estimate_twfe.py"),
               _py("pipeline/04_event_study/export_tables.py"),
               _py("pipeline/07_robustness/run_event_study_refit.py", "--variant", "main")],
              needs=[P.PANEL_ENRICHED, P.PANEL_DLA_ONLY]),
        Stage("matched", "NN matching + collapsed DiD, FSG off+on (main)",
              _matched_cmds("main"),
              env={"WIA_VARIANT": "main"},
              needs=[P.anchored_panel_path("main"), P.TREATMENT_DATES,
                     P.donor_universe_path("main")]),
        Stage("synth", "synthdid per-NSN fits + aggregation + plots (main)",
              [_py("pipeline/06_synth/build_donor_pools.py", "--variant", "main"),
               _r("pipeline/06_synth/synthdid_fits.R", w),
               _r("pipeline/06_synth/aggregate.R"),
               _py("pipeline/06_synth/plot.py", "--variant", "main")],
              env={"WIA_VARIANT": "main"}, heavy=True,
              needs=[P.PANEL_ENRICHED, P.anchored_panel_path("main"), P.donor_universe_path("main"),
                     P.TREATMENT_DATES, P.WAIVERS_CLEANED, P.NSN_REFERENCE]),
        Stage("dla_only_panels", "anchored panel + donor universe/pools (dla_only)",
              [_py("pipeline/03_panels/build_anchored_panel.py", "--variant", "dla_only"),
               _py("pipeline/06_synth/build_donor_universe.py", "--variant", "dla_only"),
               _py("pipeline/06_synth/build_donor_pools.py", "--variant", "dla_only")],
              needs=[P.PANEL_DLA_ONLY, P.donor_universe_path("main"),
                                P.TREATMENT_DATES, P.COMBINED_BIDLINK_PANEL,
                                P.DLA_ENRICHED_LATEST, P.RAW_BIDLINK, P.WAIVERS_CLEANED]),
        Stage("dla_only_fast", "event-study refit + matched controls (dla_only)",
              [_py("pipeline/07_robustness/run_event_study_refit.py", "--variant", "dla_only")]
              + _matched_cmds("dla_only"),
              env={"WIA_VARIANT": "dla_only"},
              needs=[P.PANEL_DLA_ONLY, P.anchored_panel_path("dla_only"),
                     P.TREATMENT_DATES, P.donor_universe_path("main")]),
        Stage("dla_only_synth", "synthdid fits + aggregation (dla_only)",
              [_r("pipeline/06_synth/synthdid_fits.R", w),
               _r("pipeline/06_synth/aggregate.R"),
               _py("pipeline/06_synth/plot.py", "--variant", "dla_only")],
              env={"WIA_VARIANT": "dla_only"}, heavy=True,
              needs=[P.DONOR_POOLS / "dla_only"]),
        Stage("reagg", "uniform_sample + non_container re-aggregations + ES refits",
              [_py("pipeline/07_robustness/reaggregate_matched.py"),
               _py("pipeline/07_robustness/reaggregate_synth.py"),
               _py("pipeline/07_robustness/run_event_study_refit.py")],
              needs=[V.SYNTH_ATT, V.SYNTH_SUMMARY,
                     V.MATCHED_TABLES / "nn_did_tau_per_nsn_fsg_off.csv"]),
        Stage("variant_artifacts", "thesis-style figures/tables for every variant",
              [_py("pipeline/07_robustness/build_artifacts.py")],
              needs=[P.event_study_json("uniform_sample"), P.event_study_json("main")]),
        Stage("appendix", "robustness appendix tables A7-A25 + figures A13-A14",
              [_py("pipeline/07_robustness/build_appendix_artifacts.py")],
              needs=[V.variant_paths("non_container")["tables"] / "T0_comparison_vs_main.html",
                     P.event_study_json("main")]),
        Stage("descriptives", "panel-overview + procurement-context descriptives",
              [_py("pipeline/08_artifacts/panel_overview_descriptives.py"),
               _py("pipeline/08_artifacts/procurement_context_descriptives.py")],
              needs=[P.PANEL_ENRICHED, P.PROCUREMENT_PARQUET]),
        Stage("thesis_artifacts", "main thesis tables T1-T4/TA + figures F1-F28",
              [_py("pipeline/08_artifacts/build_thesis_tables.py"),
               _py("pipeline/08_artifacts/build_thesis_figures.py")],
              needs=[V.MATCHED_TABLES / "nn_did_summary_fsg_off.csv",
                     V.SYNTH_SUMMARY, P.event_study_json("main")]),
        Stage("latex", "thesis PDF via pandoc + xelatex",
              [_py("thesis/build.py")],
              needs=[ROOT / "thesis" / "Waived in America - Honors Thesis.md",
                     P.APPENDIX / "robustness_appendix_tables.html"]),
    ]


HEAVY_BANNER = r"""
{line}
!!  THIS IS THE HEAVY ONE: stage '{key}' is a ~13-18 HOUR synthdid run.   !!
!!  It is resumable (checkpoints in data/_partial/); a kill costs only    !!
!!  the in-flight fit. See README.md for the two-terminal overlap.   !!
{line}
"""


def run_stage(st: Stage, no_backfill: bool) -> None:
    line = "=" * 74
    print(f"\n{line}\nSTAGE {st.key} — {st.desc}\n{line}")
    if st.heavy:
        print(HEAVY_BANNER.format(line="!" * 74, key=st.key))
    if st.note:
        print(f"  NOTE: {st.note}\n")
    if st.key == "backfill_replay":
        zips = sorted(P.RAW_API_BACKFILL.glob("*.zip"))
        if no_backfill:
            print("  !! SKIPPED via --no-backfill. OUTPUTS WILL NOT MATCH THE THESIS:")
            print("  !! the parquet will be missing ~169K archived DLA rows.")
            return
        if len(zips) != 1:
            sys.exit(f"FATAL: expected exactly 1 zip in {P.RAW_API_BACKFILL}, found {len(zips)}.")
    for missing in (p for p in st.needs if not p.exists()):
        sys.exit(f"FATAL: stage '{st.key}' input missing: {missing}\n"
                 f"Run the producing stage first (see --list).")

    # PYTHONIOENCODING: stage scripts print polars frames (Unicode box chars);
    # with stdout piped on Windows, Python falls back to cp1252 and crashes.
    # stdio-only — does not affect file I/O or output bytes.
    env = {**os.environ, "WIA_ROOT": str(ROOT), "MPLBACKEND": "Agg",
           "PYTHONIOENCODING": "utf-8", **st.env}
    t0 = time.perf_counter()
    for cmd in st.cmds:
        print(f"  $ {' '.join(str(c) for c in cmd)}")
        rc = subprocess.run(cmd, cwd=str(ROOT), env=env).returncode
        if rc != 0:
            elapsed = time.perf_counter() - t0
            _log(st.key, elapsed, rc)
            sys.exit(f"\nFAILED in stage '{st.key}' (exit {rc}) after {elapsed:,.0f}s.\n"
                     f"Resume with:  uv run python reproduce.py --from {st.key}")
    elapsed = time.perf_counter() - t0
    _log(st.key, elapsed, 0)
    print(f"  OK — {st.key} finished in {elapsed:,.0f}s")


def _log(key: str, seconds: float, exit_code: int) -> None:
    new = not RUN_LOG.exists()
    with open(RUN_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["stage", "seconds", "exit"])
        w.writerow([key, f"{seconds:.1f}", exit_code])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    ap.add_argument("--from", dest="from_", metavar="KEY", help="start at this stage")
    ap.add_argument("--to", metavar="KEY", help="stop after this stage")
    ap.add_argument("--only", metavar="KEYS", help="comma-separated stage keys to run")
    ap.add_argument("--skip", metavar="KEYS", default="", help="comma-separated stage keys to skip")
    ap.add_argument("--workers", type=int, default=4, help="R workers for the synth stages")
    ap.add_argument("--no-backfill", action="store_true",
                    help="skip the offline backfill replay (results WILL drift)")
    args = ap.parse_args()

    stages = build_stages(args.workers)
    keys = [s.key for s in stages]

    if args.list:
        print(f"{'stage':<18}description")
        for s in stages:
            tag = "  <-- HEAVY (~13-18h)" if s.heavy else ""
            print(f"{s.key:<18}{s.desc}{tag}")
        return

    for k in filter(None, [args.from_, args.to] + (args.only or "").split(",") + args.skip.split(",")):
        if k not in keys:
            sys.exit(f"Unknown stage '{k}'. Use --list.")

    if args.only:
        selected = [s for s in stages if s.key in args.only.split(",")]
    else:
        i = keys.index(args.from_) if args.from_ else 0
        j = keys.index(args.to) + 1 if args.to else len(stages)
        selected = stages[i:j]
    selected = [s for s in selected if s.key not in args.skip.split(",")]
    if not selected:
        sys.exit("Selection is empty (check --from/--to order and --skip). Use --list.")

    print(f"reproduce.py — running {len(selected)} stage(s): {', '.join(s.key for s in selected)}")
    heavies = [s.key for s in selected if s.heavy]
    if heavies:
        print(f"\n  >> Heads-up: this selection includes HEAVY stage(s): {', '.join(heavies)}"
              f" — each ~13-18 hours. <<\n")
    for st in selected:
        run_stage(st, args.no_backfill)
    print("\nAll selected stages complete.")


if __name__ == "__main__":
    main()
