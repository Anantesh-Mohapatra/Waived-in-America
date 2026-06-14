# Stage 3 - synthdid fits per (treated, outcome).
#
# TIME DIMENSION = day-precise event_year (per-fit donor pools already
# aggregated by 02_build_donor_pools.py). Each pool has rows = balanced units
# at the SAME set of event_years (treated's observed EYs, no fixed window).
#
# INFERENCE NOTE: synthdid's `jackknife_se` and `bootstrap_se` explicitly return
# NA for single-treated panels (see synthdid:::jackknife_se line 7 and
# bootstrap_sample line 6). Only `placebo_se` is defined for N1=1. We use
# placebo inference exclusively, mirroring synthdid::placebo_se internally so
# we can recover the placebo distribution (for empirical p-value + diagnostic
# plot) in addition to the SE.
#
# For each pool:
#   1. Build Y matrix (units x event_year), donors first, treated last.
#   2. synthdid_estimate(Y, N0, T0) where T0 = # pre-period EYs (EY <= -1).
#   3. Placebo inference (N_PLACEBO replications, seed = digest of treated+outcome):
#        For each replication: random permutation of controls; treat the last
#        position as placebo "treated"; renormalize original omega weights to
#        the subsample; re-evaluate synthdid_estimate with `update.omega=FALSE`
#        (fast, no SDP re-solve). Collect placebo ATT distribution.
#      SE = sqrt((R-1)/R) * sd(placebo_atts) (matches synthdid placebo_se).
#      CI = est +/- 1.96*SE; asymptotic p = 2*(1-Phi(|est/SE|)).
#      Empirical p = mean(|placebo_atts| >= |est|).
#   4. Save weights (unit), effect path (per-EY treated - synth), placebo dist.
#   5. Append summary row to synth_att.csv.
#
# Parallelization: future::multisession with N_WORKERS (default 4). Memory per
# worker is modest because synthdid_estimate uses the data matrix directly
# (no NxN covariance form); the placebo loop reuses pre-computed weights.
#
# CLI:
#   Rscript 03_synthdid_fits.R              # full run
#   Rscript 03_synthdid_fits.R --trial      # trial run, 2 NSNs only
#   Rscript 03_synthdid_fits.R --workers=N  # override worker count

suppressPackageStartupMessages({
  library(synthdid)
  library(arrow)
  library(dplyr)
  library(tidyr)
  library(future)
  library(future.apply)
  library(digest)
})

# ---------- paths ----------
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

POOL_DIR <- donor_pools_dir(ROOT, VARIANT)

# ---------- constants ----------
N_PLACEBO_DEFAULT <- 200
N_PLACEBO_TRIAL <- 50  # quick validation, can override with --n-placebo
DEFAULT_WORKERS <- 4
PRE_THRESHOLD <- -1  # event_year <= -1 is pre-period

OUTCOMES <- c("max_log_unit_price", "mean_offers", "domestic_share")

# ---------- CLI parsing ----------
args <- commandArgs(trailingOnly = TRUE)
trial_mode <- "--trial" %in% args
workers <- DEFAULT_WORKERS
w_arg <- grep("^--workers=", args, value = TRUE)
if (length(w_arg)) workers <- as.integer(sub("^--workers=", "", w_arg[1]))

N_PLACEBO <- if (trial_mode) N_PLACEBO_TRIAL else N_PLACEBO_DEFAULT
np_arg <- grep("^--n-placebo=", args, value = TRUE)
if (length(np_arg)) N_PLACEBO <- as.integer(sub("^--n-placebo=", "", np_arg[1]))

results_base <- if (trial_mode) {
  synth_trial_dir(ROOT, VARIANT)
} else {
  synth_tables_dir(ROOT, VARIANT)
}
OUT_TABLES <- results_base
OUT_WEIGHTS <- file.path(results_base, "weights")
OUT_EFFECT <- file.path(results_base, "effect_path")
OUT_PLACEBO <- file.path(results_base, "placebo_dist")
# Resume checkpoints live under gitignored data/ so a copied repo can never
# fake a completed run (each fit's checkpoint must be created by this script).
# Trial runs (--trial: fewer placebos) checkpoint into their own directory so
# they can never be mistaken for full-run fits and silently skipped on resume.
OUT_PARTIAL <- if (trial_mode) {
  file.path(synth_trial_dir(ROOT, VARIANT), "_partial")
} else {
  synth_partial_dir(ROOT, VARIANT)
}

for (d in c(OUT_TABLES, OUT_WEIGHTS, OUT_EFFECT, OUT_PLACEBO, OUT_PARTIAL)) {
  if (!dir.exists(d)) dir.create(d, recursive = TRUE)
}

# ---------- helpers ----------
parse_pool_filename <- function(path) {
  base <- tools::file_path_sans_ext(basename(path))
  parts <- strsplit(base, "__", fixed = TRUE)[[1]]
  list(treated_nsn = parts[1], outcome = parts[2])
}

build_Y_matrix <- function(pool_df) {
  # Returns list(Y, N0, T0, donor_ids, treated_id, event_years).
  ey_sorted <- sort(unique(pool_df$event_year))
  pre_mask <- ey_sorted <= PRE_THRESHOLD
  T0 <- sum(pre_mask)

  donor_ids <- sort(unique(pool_df$unit_id[!pool_df$is_treated]))
  treated_id <- unique(pool_df$unit_id[pool_df$is_treated])
  stopifnot(length(treated_id) == 1L)

  wide <- pool_df %>%
    select(unit_id, event_year, y) %>%
    pivot_wider(names_from = event_year, values_from = y)

  ey_cols <- as.character(ey_sorted)
  wide <- wide[, c("unit_id", ey_cols)]
  unit_order <- c(donor_ids, treated_id)
  wide <- wide[match(unit_order, wide$unit_id), ]

  Y <- as.matrix(wide[, ey_cols, drop = FALSE])
  rownames(Y) <- wide$unit_id
  colnames(Y) <- ey_cols

  list(Y = Y, N0 = length(donor_ids), T0 = T0,
       donor_ids = donor_ids, treated_id = treated_id,
       event_years = ey_sorted, pre_mask = pre_mask)
}

# Effect path for synthdid:
#   synth[t] = omega' Y_donors[, t] + omega_intercept
#   raw_effect[t] = treated[t] - synth[t]
# synthdid is a DID estimator, so it tolerates constant pre-period level
# gaps via the time-invariant intercept absorbed in the estimator. The
# *fit quality* of the synth is therefore the variance of the pre-period
# raw_effect around its own mean, NOT the absolute level of raw_effect.
# We expose both:
#   - effect = raw level difference (useful for plotting trajectories)
#   - effect_demeaned = raw_effect minus pre-period mean of raw_effect
#                       (the synthdid-relevant fit residual; tiny values
#                        in pre-period = good fit)
# Pre-RMSPE upstream uses effect_demeaned.
compute_effect_path <- function(est, mat) {
  w <- attr(est, "weights")
  omega <- w$omega
  Y <- mat$Y
  N0 <- mat$N0
  Y_donors <- Y[1:N0, , drop = FALSE]
  Y_treated <- Y[N0 + 1, ]
  synth_traj <- as.numeric(omega %*% Y_donors)
  effect <- Y_treated - synth_traj
  pre_mean_effect <- mean(effect[mat$pre_mask])
  effect_demeaned <- effect - pre_mean_effect
  data.frame(
    event_year = mat$event_years,
    is_pre = mat$pre_mask,
    y_treated = Y_treated,
    y_synth = synth_traj,
    effect = effect,
    effect_demeaned = effect_demeaned,
    pre_mean_effect = pre_mean_effect
  )
}

placebo_subsample <- function(est, n_placebo, seed) {
  # Mirrors synthdid::placebo_se but returns the placebo ATT distribution
  # so we can compute both empirical p and asymptotic SE.
  # For each replication:
  #   - permute control indices (no replacement)
  #   - treat position N0 as placebo "treated", rest as controls
  #   - renormalize the original omega weights to the new control subset
  #   - call synthdid_estimate with these weights (no SDP re-solve)
  setup <- attr(est, "setup")
  opts <- attr(est, "opts")
  weights <- attr(est, "weights")
  N0 <- setup$N0
  N1 <- nrow(setup$Y) - N0
  if (N0 <= N1) return(numeric(0))

  set.seed(seed)
  reps <- replicate(n_placebo, {
    ind <- sample.int(N0)  # permutation 1..N0
    new_N0 <- length(ind) - N1
    wb <- weights
    wb$omega <- synthdid:::sum_normalize(weights$omega[ind[seq_len(new_N0)]])
    Y_sub <- setup$Y[ind, , drop = FALSE]
    X_sub <- if (length(dim(setup$X)) == 3) setup$X[ind, , , drop = FALSE] else setup$X
    fit <- tryCatch(
      do.call(synthdid_estimate, c(
        list(Y = Y_sub, N0 = new_N0, T0 = setup$T0, X = X_sub, weights = wb),
        opts
      )),
      error = function(e) NULL
    )
    if (is.null(fit)) NA_real_ else as.numeric(fit)
  })
  reps[!is.na(reps)]
}

empty_row <- function(tnsn, outcome, N0, T0, n_post, t0, msg) {
  row <- data.frame(
    treated_nsn = tnsn, outcome = outcome,
    att = NA_real_, placebo_se = NA_real_,
    ci_lo = NA_real_, ci_hi = NA_real_,
    asymp_p = NA_real_, empirical_p = NA_real_,
    pre_rmspe = NA_real_, weight_hhi = NA_real_,
    n_balanced_pool = N0, n_active = NA_integer_,
    n_pre_eys = T0, n_post_eys = n_post,
    n_placebo_used = 0L,
    wall_sec = as.numeric(Sys.time() - t0, units = "secs"),
    error = msg
  )
  tag <- paste0(tnsn, "__", outcome)
  write_csv_lf(row, file.path(OUT_PARTIAL, paste0(tag, ".csv")), row.names = FALSE)
  row
}

fit_one <- function(pool_path) {
  meta <- parse_pool_filename(pool_path)
  tnsn <- meta$treated_nsn
  outcome <- meta$outcome
  t0 <- Sys.time()

  pool <- as.data.frame(read_parquet(pool_path))
  mat <- build_Y_matrix(pool)
  Y <- mat$Y
  N0 <- mat$N0
  T0 <- mat$T0
  n_post <- length(mat$event_years) - T0

  if (T0 < 1L || n_post < 1L || N0 < 2L) {
    return(empty_row(tnsn, outcome, N0, T0, n_post, t0,
                     "insufficient pre/post/donor counts"))
  }

  est <- tryCatch(synthdid_estimate(Y, N0, T0), error = function(e) e)
  if (inherits(est, "error")) {
    return(empty_row(tnsn, outcome, N0, T0, n_post, t0, conditionMessage(est)))
  }

  att <- as.numeric(est)

  # Placebo-based inference (only viable method for single-treated synthdid).
  seed_int <- abs(digest::digest2int(paste(tnsn, outcome)))
  placebo_atts <- placebo_subsample(est, N_PLACEBO, seed_int)
  if (length(placebo_atts) >= 2) {
    # synthdid placebo_se convention: sqrt((R-1)/R) * sd(reps)
    R <- length(placebo_atts)
    se <- sqrt((R - 1) / R) * sd(placebo_atts)
    ci_lo <- att - 1.96 * se
    ci_hi <- att + 1.96 * se
    asymp_p <- 2 * (1 - pnorm(abs(att / se)))
    emp_p <- mean(abs(placebo_atts) >= abs(att))
  } else {
    se <- NA_real_; ci_lo <- NA_real_; ci_hi <- NA_real_
    asymp_p <- NA_real_; emp_p <- NA_real_
  }

  # Effect path + pre-RMSPE (DID-adjusted, see compute_effect_path notes).
  ep <- compute_effect_path(est, mat)
  pre_rmspe <- sqrt(mean(ep$effect_demeaned[ep$is_pre]^2))

  # Weights
  w <- attr(est, "weights")
  omega <- w$omega
  active_threshold <- 1 / (2 * N0)
  n_active <- sum(omega > active_threshold)
  weight_hhi <- sum(omega^2)
  weights_df <- data.frame(
    donor_id = mat$donor_ids,
    omega = omega,
    active = omega > active_threshold
  )

  # Save per-fit artifacts
  tag <- paste0(tnsn, "__", outcome)
  write_csv_lf(weights_df, file.path(OUT_WEIGHTS, paste0(tag, ".csv")), row.names = FALSE)
  write_csv_lf(ep, file.path(OUT_EFFECT, paste0(tag, ".csv")), row.names = FALSE)
  write_csv_lf(data.frame(placebo_att = placebo_atts),
            file.path(OUT_PLACEBO, paste0(tag, ".csv")), row.names = FALSE)

  row <- data.frame(
    treated_nsn = tnsn,
    outcome = outcome,
    att = att,
    placebo_se = se,
    ci_lo = ci_lo,
    ci_hi = ci_hi,
    asymp_p = asymp_p,
    empirical_p = emp_p,
    pre_rmspe = pre_rmspe,
    weight_hhi = weight_hhi,
    n_balanced_pool = N0,
    n_active = as.integer(n_active),
    n_pre_eys = T0,
    n_post_eys = n_post,
    n_placebo_used = length(placebo_atts),
    wall_sec = as.numeric(Sys.time() - t0, units = "secs"),
    error = NA_character_
  )
  # Checkpoint: write this fit's row to _partial/ so a killed run can resume.
  write_csv_lf(row, file.path(OUT_PARTIAL, paste0(tag, ".csv")), row.names = FALSE)
  row
}

# ---------- main ----------
main <- function() {
  pool_files <- list.files(POOL_DIR, pattern = "\\.parquet$", full.names = TRUE)
  if (!length(pool_files)) stop("No pool files in ", POOL_DIR)

  if (trial_mode) {
    # Trial NSN selection: container = smallest-pool 8150-01-* that survives
    # Stage 2 (fast trial). Non-container = 1680016229189 (FSG-16, only
    # domestic_share survives — tests outcome-specific eligibility).
    pool_sizes_path <- file.path(synth_logs_dir(ROOT, VARIANT), "stage2_pool_sizes.csv")
    if (file.exists(pool_sizes_path)) {
      ps <- read.csv(pool_sizes_path, stringsAsFactors = FALSE,
                     colClasses = c(treated_nsn = "character"))
      ps_8150 <- ps[startsWith(ps$treated_nsn, "8150"), ]
      max_per_nsn <- aggregate(n_balanced_pool ~ treated_nsn, ps_8150, max)
      container_nsn <- max_per_nsn$treated_nsn[which.min(max_per_nsn$n_balanced_pool)]
    } else {
      container_nsn <- "8150014638553"  # known fallback
    }
    trial_set <- c(container_nsn, "1680016229189")
    pool_files <- pool_files[
      vapply(pool_files, function(p) {
        parse_pool_filename(p)$treated_nsn %in% trial_set
      }, logical(1L))
    ]
    cat("Trial mode: fitting", length(pool_files), "pools (",
        paste(trial_set, collapse=", "), ")\n")
  }

  # Resume support: skip pools whose _partial/<tag>.csv already exists.
  already_done <- list.files(OUT_PARTIAL, pattern = "\\.csv$", full.names = FALSE)
  already_done <- sub("\\.csv$", "", already_done)
  todo <- pool_files[
    vapply(pool_files, function(p) {
      meta <- parse_pool_filename(p)
      tag <- paste0(meta$treated_nsn, "__", meta$outcome)
      !(tag %in% already_done)
    }, logical(1L))
  ]
  cat("Pool files total:", length(pool_files),
      "| already done:", length(pool_files) - length(todo),
      "| to fit:", length(todo), "\n")
  cat("Workers:", workers, "| N_PLACEBO:", N_PLACEBO, "\n\n")

  if (length(todo)) {
    plan(multisession, workers = workers)
    on.exit(plan(sequential), add = TRUE)

    t_start <- Sys.time()
    results <- future_lapply(todo, fit_one,
                             future.seed = TRUE,
                             future.packages = c("synthdid", "arrow", "dplyr", "tidyr", "digest"))
    cat("\nNew fits done in", round(as.numeric(Sys.time() - t_start, units = "mins"), 2),
        "min wall.\n")
  } else {
    cat("Nothing to fit (all pools already in _partial/).\n")
  }

  # Assemble synth_att.csv from all _partial files (covers both this run and
  # any prior runs that already wrote rows).
  partial_files <- list.files(OUT_PARTIAL, pattern = "\\.csv$", full.names = TRUE)
  all_rows <- do.call(rbind, lapply(partial_files, function(p) {
    read.csv(p, stringsAsFactors = FALSE, colClasses = c(treated_nsn = "character"))
  }))
  out_df <- all_rows
  out_path <- file.path(OUT_TABLES, "synth_att.csv")
  write_csv_lf(out_df, out_path, row.names = FALSE)
  cat("Wrote:", out_path, "(", nrow(out_df), "rows assembled from _partial/)\n")

  # Console summary
  cat("\nSummary:\n")
  for (o in OUTCOMES) {
    sub <- out_df[out_df$outcome == o, ]
    if (!nrow(sub)) next
    cat(sprintf("  %s: n=%d, mean ATT=%.4f, median asymp p=%.3f, median emp p=%.3f\n",
                o, nrow(sub),
                mean(sub$att, na.rm = TRUE),
                median(sub$asymp_p, na.rm = TRUE),
                median(sub$empirical_p, na.rm = TRUE)))
  }
}

main()
