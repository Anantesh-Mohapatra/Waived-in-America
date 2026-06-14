# Stage 3 - Nearest-neighbor matching with fixed-cov Mahalanobis distance.
#
# Design notes:
#   * Matching covariates: 4 (log_n_transactions_per_fy + 3 pre-period means).
#   * Global Mahalanobis covariance computed once from the full set of unique
#     donor rows with all 4 covariates finite.
#   * Per (treated, donor) Mahalanobis distance precomputed via the global cov;
#     treated's own distance set to 0; vector passed to matchit().
#   * USE_FSG_EXACT toggles FSG exact-match. Default FALSE; flip to TRUE
#     for a sensitivity run. When TRUE, treated NSNs whose same-FSG donor
#     pool is empty (or too small to provide K matches after filtering)
#     are reported as skips in `results/<variant>/matched/logs/stage3_match_log*.csv`.
#   * method = "nearest", distance = vec, replace = TRUE, ratio = 3.

suppressPackageStartupMessages({
    library(arrow)
    library(MatchIt)
    library(dplyr)
})

set.seed(20260430)

# ---------- paths ----------
# Repo root: WIA_ROOT env override, else walk up (script dir, then cwd) until
# pyproject.toml is found. Uniform block across all pipeline R scripts.
find_root <- function() {
    env <- Sys.getenv("WIA_ROOT", unset = "")
    if (nzchar(env)) return(normalizePath(env))
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("^--file=", args, value = TRUE)
    starts <- c(if (length(file_arg) > 0) dirname(normalizePath(sub("^--file=", "", file_arg[1]))), getwd())
    for (s in starts) {
        p <- s
        while (!file.exists(file.path(p, "pyproject.toml")) && dirname(p) != p) p <- dirname(p)
        if (file.exists(file.path(p, "pyproject.toml"))) return(p)
    }
    stop("Could not locate repo root (pyproject.toml) from: ", paste(starts, collapse = ", "))
}
ROOT <- find_root()
cat("  project root:", ROOT, "\n")

# Shared layout derivation + variant routing (WIA_VARIANT env, validated).
source(file.path(ROOT, "pipeline", "lib", "paths.R"))
VARIANT <- wia_variant()
v_suffix <- v_suffix_for(VARIANT)
DATA_DIR     <- anchored_data_dir(ROOT)
MATCHING_DIR <- matched_matching_dir(ROOT, VARIANT)
TABLES_DIR   <- matched_tables_dir(ROOT, VARIANT)
LOGS_DIR     <- matched_logs_dir(ROOT, VARIANT)

# ---------- FSG exact-match toggle (--fsg=off|on, default off) ----------
# off (default) lets Mahalanobis pick neighbors across product groups.
# on adds an exact-match constraint on FSG (2-digit FSC) at MatchIt.
# reproduce.py runs this script once per setting.
cli <- commandArgs(trailingOnly = TRUE)
fsg_arg <- sub("^--fsg=", "", grep("^--fsg=", cli, value = TRUE))
FSG <- if (length(fsg_arg)) fsg_arg[1] else "off"
if (!FSG %in% c("off", "on")) stop("--fsg must be 'off' or 'on'; got: ", FSG)
USE_FSG_EXACT <- FSG == "on"
fsg_suffix <- paste0("fsg_", FSG)
cat("  FSG exact-match:", USE_FSG_EXACT, "(suffix:", fsg_suffix, ")\n")

MC_PATH <- file.path(DATA_DIR, sprintf("matching_covariates%s.parquet", v_suffix))
# Outputs carry the FSG suffix so both variants can co-exist for Stage 5/6.
PAIRS_OUT <- file.path(MATCHING_DIR, sprintf("matched_pairs_nn_%s.csv", fsg_suffix))
BALANCE_OUT <- file.path(TABLES_DIR, sprintf("balance_nn_%s.csv", fsg_suffix))
MATCH_LOG_OUT <- file.path(LOGS_DIR, sprintf("stage3_match_log_%s.csv", fsg_suffix))

# ---------- constants ----------
COVS <- c(
    "log_n_transactions_per_fy",
    "mean_max_log_unit_price",
    "mean_domestic_share",
    "mean_pre_offers"
)

K_NEIGHBORS <- 3

# ---------- main ----------
cat("Stage 3 - NN matching (fixed-cov Mahalanobis)\n")
cat("  reading:", MC_PATH, "\n")
mc <- read_parquet(MC_PATH)
cat("  rows:", format(nrow(mc), big.mark = ","), "\n")

# Drop rows with any NaN/NA/Inf in the 4 matching covariates.
mc_complete <- mc %>% filter(if_all(all_of(COVS), ~ is.finite(.)))
cat("  rows after dropping non-finite covariates:", format(nrow(mc_complete), big.mark = ","), "\n")

# Global covariance from unique donor rows (donor NSNs may appear under multiple
# treated_nsn groups; cov on uniques avoids over-weighting popular donors).
donor_rows_unique <- mc_complete %>%
    filter(!is_treated) %>%
    distinct(nsn, .keep_all = TRUE)
cat("  unique donors with full covariates:", format(nrow(donor_rows_unique), big.mark = ","), "\n")

X_donor <- as.matrix(donor_rows_unique[, COVS])
global_cov <- cov(X_donor)
cat("  global cov computed (", length(COVS), "x", length(COVS),
    "); det =", format(det(global_cov), digits = 4), "\n", sep = "")

dir.create(TABLES_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(MATCHING_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(LOGS_DIR, showWarnings = FALSE, recursive = TRUE)

# Numerical safety: ridge-regularize if near-singular.
if (rcond(global_cov) < 1e-10) {
    cat("  WARNING: global_cov is near-singular; adding ridge 1e-6 * I\n")
    global_cov <- global_cov + diag(1e-6, ncol(global_cov))
}
inv_global_cov <- solve(global_cov)

# ---------- per-treated MatchIt loop ----------
treated_nsns <- sort(unique(mc_complete$treated_nsn))
# All treated NSNs that came out of Stage 2 (before the is.finite drop) so we
# can report *why* a treated NSN got dropped (which covariate is non-finite).
treated_nsns_all <- sort(unique(mc$treated_nsn[mc$is_treated]))
cat("  treated NSNs in Stage 2 output:", length(treated_nsns_all), "\n")
cat("  treated NSNs after non-finite drop:", length(treated_nsns), "\n\n")

all_matched <- list()
balance_rows <- list()
match_log <- list()

for (i in seq_along(treated_nsns_all)) {
    tnsn <- treated_nsns_all[i]
    sub_full <- mc %>% filter(treated_nsn == tnsn)
    treated_row_full <- sub_full %>% filter(is_treated)
    treated_fsg <- if (nrow(treated_row_full) == 1) treated_row_full$fsg[1] else NA_character_

    # If the treated NSN's row has any non-finite covariate, report which.
    if (nrow(treated_row_full) == 1) {
        treated_vals <- unlist(treated_row_full[1, COVS], use.names = TRUE)
        bad_covs <- names(treated_vals)[!is.finite(treated_vals)]
    } else {
        bad_covs <- character(0)
    }

    sub <- mc_complete %>% filter(treated_nsn == tnsn)
    treated_row <- sub %>% filter(is_treated)
    n_donors_complete <- sum(!sub$is_treated)
    donors_same_fsg <- if (!is.na(treated_fsg)) {
        sum(!sub$is_treated & sub$fsg == treated_fsg, na.rm = TRUE)
    } else 0

    if (nrow(treated_row) != 1) {
        reason <- if (length(bad_covs) > 0) {
            paste0("treated_row_has_non_finite_covariate:",
                   paste(bad_covs, collapse = ","))
        } else {
            "treated_row_missing"
        }
        cat(sprintf("  [%02d/%02d] %s: SKIP - %s\n",
                    i, length(treated_nsns_all), tnsn, reason))
        match_log[[tnsn]] <- data.frame(
            treated_nsn = tnsn, fsg = treated_fsg,
            n_donors_complete = n_donors_complete,
            n_donors_same_fsg = donors_same_fsg,
            n_matched = 0L, status = reason)
        next
    }

    treated_x <- unlist(treated_row[1, COVS], use.names = FALSE)
    X <- as.matrix(sub[, COVS])
    diffs <- sweep(X, 2, treated_x, "-")
    d2 <- rowSums((diffs %*% inv_global_cov) * diffs)
    distance_vec <- sqrt(pmax(d2, 0))

    # Defensive: cap any non-finite distances (overflow, etc.) to a large
    # finite value so MatchIt's input validation passes. Log if encountered.
    bad <- !is.finite(distance_vec)
    if (any(bad)) {
        cat(sprintf("  [%02d] %s: %d non-finite distances; capping to 1e8\n",
                    i, tnsn, sum(bad)))
        distance_vec[bad] <- 1e8
    }


    sub$treated_int <- as.integer(sub$is_treated)
    sub$._mhd <- distance_vec  # captured into match output

    fmla <- as.formula(paste(
        "treated_int ~",
        paste(COVS, collapse = " + ")
    ))

    matchit_args <- list(
        formula = fmla,
        data = sub,
        method = "nearest",
        distance = distance_vec,
        replace = TRUE,
        ratio = K_NEIGHBORS
    )
    if (USE_FSG_EXACT) {
        matchit_args$exact <- ~ fsg
    }

    m <- tryCatch(
        do.call(matchit, matchit_args),
        error = function(e) {
            message(sprintf("  matchit error for %s: %s", tnsn, e$message))
            NULL
        }
    )
    if (is.null(m)) {
        match_log[[tnsn]] <- data.frame(
            treated_nsn = tnsn, fsg = treated_fsg,
            n_donors_complete = n_donors_complete,
            n_donors_same_fsg = donors_same_fsg,
            n_matched = 0L, status = "matchit_error")
        next
    }

    md <- match.data(m, drop.unmatched = TRUE)
    matched_donors <- md %>%
        filter(treated_int == 0) %>%
        mutate(
            treated_nsn = tnsn,
            donor_nsn = nsn,
            mahalanobis_distance = ._mhd,
            match_weight = weights
        ) %>%
        select(any_of(c("treated_nsn", "donor_nsn", "fsg",
                        "mahalanobis_distance", "match_weight", "subclass")))

    n_matched <- nrow(matched_donors)
    all_matched[[tnsn]] <- matched_donors

    # Balance via summary(m, standardize=TRUE)
    s <- summary(m, standardize = TRUE)
    sm <- as.data.frame(s$sum.matched)
    sm$treated_nsn <- tnsn
    sm$covariate <- rownames(sm)
    sa <- as.data.frame(s$sum.all)
    sa$treated_nsn <- tnsn
    sa$covariate <- rownames(sa)
    sa$stage <- "before"
    sm$stage <- "after"
    balance_rows[[tnsn]] <- bind_rows(sa, sm)

    match_log[[tnsn]] <- data.frame(
        treated_nsn = tnsn, fsg = treated_fsg,
        n_donors_complete = n_donors_complete,
        n_donors_same_fsg = donors_same_fsg,
        n_matched = n_matched,
        status = "ok")

    if (i %% 5 == 0 || i == length(treated_nsns_all)) {
        cat(sprintf("  [%02d/%02d] %s (fsg=%s): n_donors_same_fsg=%d -> matched=%d\n",
                    i, length(treated_nsns_all), tnsn, treated_fsg, donors_same_fsg, n_matched))
    }
}

# ---------- write ----------
# Sort matched pairs on the unique (treated, donor) key before writing: the
# per-treated match.data() rows come out in the covariate frame's (uncontracted)
# order, which reshuffled this committed diagnostic run-to-run. Order does not
# affect the downstream collapsed-DiD estimate (a weighted sum over pairs).
matched_pairs <- bind_rows(all_matched) %>% arrange(treated_nsn, donor_nsn)
# Round the balance SMDs to 6 dp before writing: the diagnostic statistics
# carry summation-order float noise that flips a low digit run-to-run. 6 dp is
# far below any reported precision for a standardized mean difference yet far
# above the noise, so the rounding boundary is never straddled. Diagnostics
# only; the estimates are computed separately and are byte-stable unrounded.
balance_df <- bind_rows(balance_rows) %>%
    mutate(across(where(is.numeric), \(x) round(x, 6)))
match_log_df <- bind_rows(match_log)

write_csv_lf(matched_pairs, PAIRS_OUT, row.names = FALSE)
# Only the FSG-off balance feeds the appendix balance table; the FSG-on balance
# is diagnostic-only (printed below), so it is not written.
if (FSG == "off") write_csv_lf(balance_df, BALANCE_OUT, row.names = FALSE)
write_csv_lf(match_log_df, MATCH_LOG_OUT, row.names = FALSE)

cat("\n")
cat("  wrote:", PAIRS_OUT, "(", format(nrow(matched_pairs), big.mark = ","), "rows)\n")
if (FSG == "off") cat("  wrote:", BALANCE_OUT, "(", format(nrow(balance_df), big.mark = ","), "rows)\n")
cat("  wrote:", MATCH_LOG_OUT, "(", nrow(match_log_df), "treated NSNs)\n")

n_matched_treated <- length(unique(matched_pairs$treated_nsn))
cat("  treated NSNs matched: ", n_matched_treated, " / ", length(treated_nsns_all), "\n", sep = "")
if (n_matched_treated < length(treated_nsns_all)) {
    failed <- match_log_df %>% filter(status != "ok")
    cat("  failures:\n")
    print(failed)
}

# Post-match balance summary: median |SMD| per covariate across matched treated NSNs.
post <- balance_df %>% filter(stage == "after")
worst <- post %>%
    group_by(covariate) %>%
    summarise(
        median_abs_smd = median(abs(`Std. Mean Diff.`), na.rm = TRUE),
        max_abs_smd = max(abs(`Std. Mean Diff.`), na.rm = TRUE),
        n_above_0_25 = sum(abs(`Std. Mean Diff.`) > 0.25, na.rm = TRUE),
        .groups = "drop"
    ) %>%
    arrange(desc(median_abs_smd))
cat(sprintf("\n  post-match SMD summary (median across %d matched treated NSNs):\n",
            n_matched_treated))
print(worst, n = 20)
