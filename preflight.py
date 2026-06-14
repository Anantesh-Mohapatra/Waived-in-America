"""preflight.py — verify the environment before running the pipeline.

    uv run python reproduce.py --list     # what would run
    uv run python preflight.py            # is this machine ready?

Checks Python + R toolchains, raw data presence, stale checkpoints, and disk
space. Writes preflight_report.md (gitignored) and exits nonzero on any FAIL.
Run this before anything long-running — the synth stages are 13-18 hours each
and deserve a clean runway.
"""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pipeline" / "lib"))
import paths as P  # noqa: E402

REPORT = ROOT / "preflight_report.md"

PY_PACKAGES = ["polars", "pyarrow", "pyfixest", "seaborn",
               "jupytext", "joblib", "bs4", "matplotlib",
               "pandas", "scipy", "numpy"]
R_PACKAGES = ["arrow", "MatchIt", "dplyr", "synthdid", "tidyr",
              "future", "future.apply", "digest"]
# Verified versions (see r_requirements.md). Mismatch = WARN, not FAIL.
R_VERIFIED_INTERPRETER = "4.5.1"
R_VERIFIED = {"arrow": "23.0.1.2", "MatchIt": "4.7.2", "dplyr": "1.1.4",
              "synthdid": "0.0.9", "tidyr": "1.3.1",
              "future": "1.69.0", "future.apply": "1.20.2", "digest": "0.6.37"}

rows: list[tuple[str, str, str]] = []  # (status, check, detail)


def add(status: str, check: str, detail: str = "") -> None:
    rows.append((status, check, detail))
    print(f"  [{status}] {check}" + (f" — {detail}" if detail else ""))


def check_python() -> None:
    print("\n== Python ==")
    v = sys.version_info
    add("PASS" if v >= (3, 13) else "FAIL", "Python >= 3.13", f"{v.major}.{v.minor}.{v.micro}")
    for name in PY_PACKAGES:
        try:
            m = importlib.import_module(name)
            add("PASS", f"import {name}", getattr(m, "__version__", "?"))
        except ImportError as e:
            add("FAIL", f"import {name}", f"{e} — run `uv sync`")


def check_r() -> None:
    print("\n== R ==")
    if not shutil.which("Rscript"):
        add("FAIL", "Rscript on PATH", "install R and add to PATH")
        return
    script = ('cat(R.version.string, "\\n"); '
              'for (p in c(' + ",".join(f'"{p}"' for p in R_PACKAGES) + ')) '
              'cat(p, tryCatch(as.character(packageVersion(p)), error=function(e) "MISSING"), "\\n")')
    out = subprocess.run(["Rscript", "-e", script], capture_output=True, text=True)
    if out.returncode != 0:
        add("FAIL", "Rscript -e", out.stderr.strip()[:200])
        return
    lines = out.stdout.strip().splitlines()
    r_version = lines[0] if lines else "?"
    if R_VERIFIED_INTERPRETER in r_version:
        add("PASS", "R", r_version)
    else:
        add("WARN", "R", f"{r_version} != verified R {R_VERIFIED_INTERPRETER} "
            "(see r_requirements.md; results are version-sensitive)")
    for line in lines[1:]:
        parts = line.split()
        if len(parts) != 2:
            continue
        pkg, ver = parts
        if ver == "MISSING":
            add("FAIL", f"R package {pkg}", "run `Rscript install_r_packages.R`")
        elif R_VERIFIED.get(pkg) and ver != R_VERIFIED[pkg]:
            add("WARN", f"R package {pkg}", f"{ver} != verified {R_VERIFIED[pkg]} (see r_requirements.md)")
        else:
            add("PASS", f"R package {pkg}", ver)


def check_tex() -> None:
    print("\n== TeX toolchain (thesis PDF) ==")
    for tool in ["pandoc", "xelatex"]:
        add("PASS" if shutil.which(tool) else "FAIL", f"{tool} on PATH")


def check_raw_data() -> None:
    print("\n== Raw data ==")
    fy = sorted(P.RAW_PROCUREMENT.glob("FY*_All_Contracts_Full_*.zip")) if P.RAW_PROCUREMENT.exists() else []
    add("PASS" if len(fy) == 10 else "FAIL", "FY bulk zips (expect 10)", str(len(fy)))
    bz = sorted(P.RAW_API_BACKFILL.glob("*.zip")) if P.RAW_API_BACKFILL.exists() else []
    add("PASS" if len(bz) == 1 else "FAIL", "api_backfill zip (expect exactly 1)",
        ", ".join(z.name for z in bz) or "none")
    dla = list(P.RAW_DLA_FOIA.glob("*")) if P.RAW_DLA_FOIA.exists() else []
    add("PASS" if len(dla) >= 13 else "FAIL", "DLA_FOIA files (expect >= 13)", str(len(dla)))
    cc = list((P.RAW_BIDLINK / "control_contracts").glob("*.csv")) if P.RAW_BIDLINK.exists() else []
    add("PASS" if len(cc) == 63 else "FAIL",
        "bidlink control_contracts CSVs (loader asserts exactly 63)", str(len(cc)))
    nsn = list((P.RAW_BIDLINK / "nsn").rglob("*.csv")) if P.RAW_BIDLINK.exists() else []
    add("PASS" if len(nsn) >= 30 else "FAIL", "bidlink nsn exports (expect >= 30)", str(len(nsn)))
    for f in [P.RAW_WAIVERS, P.UNIT_COST_CLIN_TAGGING]:
        add("PASS" if f.exists() else "FAIL", f"{f.relative_to(P.REPO_ROOT)}")


def check_checkpoints() -> None:
    print("\n== Stale checkpoints (a populated _partial dir makes the next run skip those fits) ==")
    # data/_partial/synth_<variant> gates the full runs; data/synth_trial/<variant>/_partial
    # gates only --trial reruns (trial and full checkpoints are kept separate by design).
    partials = [p for p in (P.DATA).rglob("_partial") if p.is_dir() and any(p.iterdir())] \
        if P.DATA.exists() else []
    add("PASS" if not partials else "WARN", "no populated _partial dirs",
        "; ".join(str(p.relative_to(P.REPO_ROOT)) for p in partials)
        or "clean (a fresh run will fit every pool)")


def check_resources() -> None:
    print("\n== Resources ==")
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    add("PASS" if free_gb >= 80 else "FAIL", "free disk >= 80 GB", f"{free_gb:,.0f} GB")
    add("INFO", "logical cores", str(os.cpu_count()))


def main() -> None:
    print("preflight — waived-in-america")
    check_python()
    check_r()
    check_tex()
    check_raw_data()
    check_checkpoints()
    check_resources()

    fails = [r for r in rows if r[0] == "FAIL"]
    warns = [r for r in rows if r[0] == "WARN"]
    REPORT.write_text(
        "# Preflight report\n\n| status | check | detail |\n|---|---|---|\n"
        + "\n".join(f"| {s} | {c} | {d} |" for s, c, d in rows)
        + f"\n\n**{len(fails)} FAIL, {len(warns)} WARN**\n",
        encoding="utf-8")
    print(f"\n{'=' * 50}\n{len(fails)} FAIL, {len(warns)} WARN  ->  {REPORT.name}")
    if fails:
        sys.exit(1)
    print("READY.")


if __name__ == "__main__":
    main()
