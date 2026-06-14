# R environment

The matched-controls and synthetic-control stages run in R, invoked as
`Rscript` subprocesses by `reproduce.py`. R package versions are **not**
lockfile-managed (no renv); the versions below are the exact environment the
published results were produced — and verified — with. `preflight.py` prints
your installed versions and warns on mismatches.

## Verified environment

- **R 4.5.1 (2025-06-13 ucrt)**, platform `x86_64-w64-mingw32` (Windows 11)

| Package      | Version  | Source |
|--------------|----------|--------|
| arrow        | 23.0.1.2 | CRAN   |
| MatchIt      | 4.7.2    | CRAN   |
| dplyr        | 1.1.4    | CRAN   |
| synthdid     | 0.0.9    | **GitHub** (`synth-inference/synthdid`; not on CRAN) |
| tidyr        | 1.3.1    | CRAN   |
| future       | 1.69.0   | CRAN   |
| future.apply | 1.20.2   | CRAN   |
| digest       | 0.6.37   | CRAN   |

Install everything with:

```r
source("install_r_packages.R")
```

## Version-sensitivity notes

All randomness is seeded, so results are deterministic *within* a version of
this environment:

- **MatchIt** nearest-neighbor matching runs under `set.seed(20260430)`
  (match_nn.R). Tie-breaking behavior is deterministic within a MatchIt
  version but is not guaranteed across major versions — if your matched
  pairs differ from the committed `matched_pairs_nn_*.csv`, check
  `packageVersion("MatchIt")` first.
- **synthdid** placebo inference uses a per-fit deterministic seed
  (`digest::digest2int(paste(treated_nsn, outcome))`), making results
  independent of worker count and run order — but not of synthdid version.
- For strict pinning, `remotes::install_version("MatchIt", "4.7.2")` etc.
  reproduces the verified environment exactly.
