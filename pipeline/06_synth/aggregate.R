# Stage 4 - Aggregate Stage 3 outputs into summary tables.
#
# Reads results/tables/synth_att.csv (or trial/synth_att.csv) and produces
# results/tables/synth_summary.csv with per-outcome aggregates:
#   - n_fits, mean_att, median_att, sd_att
#   - n_asymp_p_lt_05, n_empirical_p_lt_05 (counts of fits with p < 0.05)
#   - distribution of pre_rmspe (median, q25, q75)
#   - median n_active, weight_hhi.
#
# Stage 3 already writes the per-fit synth_att.csv directly, so this stage is
# purely a cross-fit summary. No re-computation.

suppressPackageStartupMessages({
  library(dplyr)
})

# Repo root: WIA_ROOT env override, else walk up (script dir, then cwd) until
# pyproject.toml is found. Uniform block across all pipeline R scripts.
find_root <- function() {
  env <- Sys.getenv("WIA_ROOT", unset = "")
  if (nzchar(env)) return(normalizePath(env))
  cargs <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cargs, value = TRUE)
  starts <- c(if (length(file_arg) > 0) dirname(normalizePath(sub("^--file=", "", file_arg[1]))), getwd())
  for (s in starts) {
    p <- s
    while (!file.exists(file.path(p, "pyproject.toml")) && dirname(p) != p) p <- dirname(p)
    if (file.exists(file.path(p, "pyproject.toml"))) return(p)
  }
  stop("Could not locate repo root (pyproject.toml) from: ", paste(starts, collapse = ", "))
}
ROOT <- find_root()

# Shared layout derivation + variant routing (WIA_VARIANT env, validated).
source(file.path(ROOT, "pipeline", "lib", "paths.R"))
VARIANT <- wia_variant()

args <- commandArgs(trailingOnly = TRUE)
trial_mode <- "--trial" %in% args

TABLES_DIR <- if (trial_mode) synth_trial_dir(ROOT, VARIANT) else synth_tables_dir(ROOT, VARIANT)

ATT_PATH <- file.path(TABLES_DIR, "synth_att.csv")
SUMMARY_PATH <- file.path(TABLES_DIR, "synth_summary.csv")

main <- function() {
  if (!file.exists(ATT_PATH)) stop("Missing input: ", ATT_PATH)
  att <- read.csv(ATT_PATH, stringsAsFactors = FALSE)
  cat("Loaded", nrow(att), "fit rows from", ATT_PATH, "\n")

  ok <- att[is.na(att$error), , drop = FALSE]
  if (!nrow(ok)) stop("No successful fits in synth_att.csv")

  summ <- ok %>%
    group_by(outcome) %>%
    summarise(
      n_fits = n(),
      mean_att = mean(att, na.rm = TRUE),
      median_att = median(att, na.rm = TRUE),
      sd_att = sd(att, na.rm = TRUE),
      n_asymp_p_lt_05 = sum(asymp_p < 0.05, na.rm = TRUE),
      n_empirical_p_lt_05 = sum(empirical_p < 0.05, na.rm = TRUE),
      median_pre_rmspe = median(pre_rmspe, na.rm = TRUE),
      q25_pre_rmspe = quantile(pre_rmspe, 0.25, na.rm = TRUE),
      q75_pre_rmspe = quantile(pre_rmspe, 0.75, na.rm = TRUE),
      median_n_active = median(n_active, na.rm = TRUE),
      median_weight_hhi = median(weight_hhi, na.rm = TRUE),
      median_balanced_pool = median(n_balanced_pool, na.rm = TRUE),
      .groups = "drop"
    )

  write_csv_lf(summ, SUMMARY_PATH, row.names = FALSE)
  cat("Wrote:", SUMMARY_PATH, "\n\n")
  print(summ)

  n_err <- sum(!is.na(att$error))
  if (n_err) cat("\nNote:", n_err, "fits had errors (see synth_att.csv `error` column)\n")
}

main()
