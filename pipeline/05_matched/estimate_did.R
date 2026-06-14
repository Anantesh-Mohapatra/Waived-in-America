# Stage 5 - Single-ATT DiD on the NN matched sample.
#
# Estimator: Path 1 - 2x2 collapsed DiD on the matched sample, per outcome.
#   For each (treated_nsn, donor_nsn) pair, compute
#     delta = mean(Y, post) - mean(Y, pre)
#   using the treated's event-year clock on both sides. Per treated NSN,
#     tau_i = delta_treated_i - sum_j (w_ij * delta_donor_ij)
#   ATT = mean(tau_i). SE = sd(tau_i) / sqrt(n_treated) (clustered on
#   treated_nsn by construction; tau_i is the only thing varying).
#   Per-NSN tau_i is exported for the histogram / strip-plot deliverable.
#
# Sensitivity: rerun with post_it = 1[event_year >= 2]. NSNs whose latest
# observation is < EY 2 (waivers in 2024-2025) drop out under this rule.
#
# Pre-period: EY [-5, -1] (anchored to each treated's first_waiver_date).
# Post-period: thresholds defined below.
#
# Per-outcome NSN eligibility:
#   * Price (max_log_unit_price): NSNs whose pre-period max_log_unit_price
#     was NaN at Stage 2 are already absent from matched_pairs_nn.csv. No
#     extra filter needed here.
#   * Domestic share / offers: same matched sample. mean_offers is permitted
#     to have NA post-period cells; pre-mean and post-mean compute only over
#     observed cells.

suppressPackageStartupMessages({
    library(arrow)
    library(dplyr)
})

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

# Shared layout derivation + variant routing (WIA_VARIANT env, validated).
source(file.path(ROOT, "pipeline", "lib", "paths.R"))
VARIANT <- wia_variant()
v_suffix <- v_suffix_for(VARIANT)
DATA_DIR     <- anchored_data_dir(ROOT)
MATCHING_DIR <- matched_matching_dir(ROOT, VARIANT)
TABLES_DIR   <- matched_tables_dir(ROOT, VARIANT)

# ---------- FSG variant selector (--fsg=off|on, default off) ----------
# Picks the Stage 3 pairs file and tags all outputs so both settings can
# co-exist for Stage 6.
cli <- commandArgs(trailingOnly = TRUE)
fsg_arg <- sub("^--fsg=", "", grep("^--fsg=", cli, value = TRUE))
FSG <- if (length(fsg_arg)) fsg_arg[1] else "off"
if (!FSG %in% c("off", "on")) stop("--fsg must be 'off' or 'on'; got: ", FSG)
FSG_SUFFIX <- paste0("fsg_", FSG)
cat("  FSG variant:", FSG_SUFFIX, "\n")

# Anchored panel keyed by (anchor_nsn, nsn, fy, event_year) with day-precise
# event_year. Built by pipeline/03_panels/build_anchored_panel.py.
PANEL_PATH <- file.path(DATA_DIR, sprintf("anchored_panel%s.parquet", v_suffix))
PAIRS_PATH <- file.path(MATCHING_DIR, sprintf("matched_pairs_nn_%s.csv", FSG_SUFFIX))
TREATED_CSV <- file.path(ROOT, "results", "treatment", "treatment_nsn_first_waiver_dates.csv")

OUT_PANEL <- file.path(DATA_DIR, sprintf("nn_matched_analysis_panel%s_%s.parquet", v_suffix, FSG_SUFFIX))
OUT_TAU <- file.path(TABLES_DIR, sprintf("nn_did_tau_per_nsn_%s.csv", FSG_SUFFIX))
OUT_SUMMARY <- file.path(TABLES_DIR, sprintf("nn_did_summary_%s.csv", FSG_SUFFIX))

# ---------- knobs ----------
OUTCOMES <- c("max_log_unit_price", "mean_offers", "domestic_share")
PRE_WINDOW <- c(-5, -1)

# Each entry defines a post-period threshold. `headline` = full post window
# (EY >= 0). `ey2plus` = restrict to EY >= 2 to check whether short-run
# pipeline effects drive the ATT.
POST_THRESHOLDS <- c(headline = 0L, ey2plus = 2L)

dir.create(TABLES_DIR, showWarnings = FALSE, recursive = TRUE)

# ---------- load ----------
cat("Stage 5 - NN matched-sample DiD (single ATT)\n")

# Anchored panel is keyed by (anchor_nsn, nsn, fy, event_year) with day-precise
# event_year. We use cells directly: no further (nsn, fy) aggregation. When an
# event_year boundary falls inside an FY, that FY appears as multiple cells in
# the panel; each cell is one observation in pre/post means.
panel <- read_parquet(PANEL_PATH)
cat("  panel rows:", format(nrow(panel), big.mark = ","),
    " | anchors:", length(unique(panel$anchor_nsn)),
    " | NSNs:", length(unique(panel$nsn)), "\n")

pairs <- read.csv(PAIRS_PATH, stringsAsFactors = FALSE)
pairs$treated_nsn <- as.character(pairs$treated_nsn)
pairs$donor_nsn <- as.character(pairs$donor_nsn)
cat("  matched donor rows:", nrow(pairs), "across",
    length(unique(pairs$treated_nsn)), "treated NSNs\n")

treated <- read.csv(TREATED_CSV, stringsAsFactors = FALSE)
treated$nsn <- gsub("-", "", treated$nsn)
treated$first_waiver_date <- as.Date(treated$first_waiver_date)
treated_dates <- setNames(treated$first_waiver_date, treated$nsn)

# ---------- build row-level analysis panel ----------
# One row per (treated_nsn, donor_nsn, fy, event_year) cell, with event_year
# already stamped day-precise relative to the treated NSN's first_waiver_date
# by the anchored panel build. Donors can appear under multiple treated NSNs
# (replace=TRUE matching); each appearance is a distinct counterfactual with
# its own event-year clock (selected via the anchor_nsn filter).

build_treated_rows <- function(tnsn, fwd) {
    panel %>%
        filter(anchor_nsn == tnsn, nsn == tnsn) %>%
        mutate(
            treated_nsn = tnsn,
            donor_nsn = tnsn,
            is_treated_unit = 1L,
            match_weight = 1.0,
            pair_id = paste0(tnsn, "::self")
        )
}

build_donor_rows <- function(tnsn, donor_id, fwd, mw) {
    panel %>%
        filter(anchor_nsn == tnsn, nsn == donor_id) %>%
        mutate(
            treated_nsn = tnsn,
            donor_nsn = donor_id,
            is_treated_unit = 0L,
            match_weight = mw,
            pair_id = paste0(tnsn, "::", donor_id)
        )
}

matched_treated <- unique(pairs$treated_nsn)
chunks <- list()
for (tnsn in matched_treated) {
    fwd <- treated_dates[[tnsn]]
    if (is.na(fwd)) {
        warning(sprintf("Missing waiver date for %s; skipping", tnsn))
        next
    }
    chunks[[paste0(tnsn, "::treated")]] <- build_treated_rows(tnsn, fwd)
    pdf <- pairs %>% filter(treated_nsn == tnsn)
    for (k in seq_len(nrow(pdf))) {
        chunks[[paste0(tnsn, "::donor::", k)]] <- build_donor_rows(
            tnsn = tnsn, donor_id = pdf$donor_nsn[k],
            fwd = fwd, mw = pdf$match_weight[k]
        )
    }
}

analysis <- bind_rows(chunks) %>%
    filter(!is.na(event_year))

cat("  analysis panel rows:", format(nrow(analysis), big.mark = ","), "\n")

write_parquet(analysis, OUT_PANEL)
cat("  wrote:", OUT_PANEL, "\n\n")

# ---------- Path 1: 2x2 collapsed DiD ----------
# For each (treated, donor) pair, compute pre-mean and post-mean. Then
# per treated NSN, tau_i = delta_treated_i - sum_j (w_ij * delta_donor_ij).

per_unit_means <- function(df, outcome, post_threshold) {
    df %>%
        filter(.data[[outcome]] |> is.finite()) %>%
        mutate(
            in_pre = event_year >= PRE_WINDOW[1] & event_year <= PRE_WINDOW[2],
            in_post = event_year >= post_threshold
        ) %>%
        group_by(treated_nsn, donor_nsn, is_treated_unit, match_weight) %>%
        summarise(
            pre_mean = mean(.data[[outcome]][in_pre], na.rm = TRUE),
            post_mean = mean(.data[[outcome]][in_post], na.rm = TRUE),
            n_pre = sum(in_pre),
            n_post = sum(in_post),
            .groups = "drop"
        ) %>%
        mutate(
            pre_mean = ifelse(is.finite(pre_mean), pre_mean, NA_real_),
            post_mean = ifelse(is.finite(post_mean), post_mean, NA_real_)
        )
}

path1_results_tau <- list()
path1_results_summary <- list()

for (label in names(POST_THRESHOLDS)) {
    post_thr <- POST_THRESHOLDS[[label]]
    cat(sprintf("Path 1 (collapsed DiD) - post=%s (EY >= %d)\n", label, post_thr))

    for (outcome in OUTCOMES) {
        means <- per_unit_means(analysis, outcome, post_thr)

        # delta per (treated, donor): require both pre and post observed.
        pair_deltas <- means %>%
            filter(!is.na(pre_mean), !is.na(post_mean)) %>%
            mutate(delta = post_mean - pre_mean)

        # Treated's own delta (one row per treated_nsn).
        t_delta <- pair_deltas %>%
            filter(is_treated_unit == 1L) %>%
            transmute(treated_nsn, delta_treated = delta,
                      pre_treated = pre_mean, post_treated = post_mean)

        # Weighted donor-side counterfactual delta per treated_nsn.
        d_delta <- pair_deltas %>%
            filter(is_treated_unit == 0L) %>%
            group_by(treated_nsn) %>%
            summarise(
                delta_donor = sum(match_weight * delta) / sum(match_weight),
                n_donors = n(),
                sum_w = sum(match_weight),
                .groups = "drop"
            )

        tau <- t_delta %>%
            inner_join(d_delta, by = "treated_nsn") %>%
            mutate(
                tau = delta_treated - delta_donor,
                outcome = outcome,
                post_label = label,
                post_threshold = post_thr
            )

        path1_results_tau[[paste(outcome, label, sep = "::")]] <- tau

        if (nrow(tau) > 0) {
            att <- mean(tau$tau, na.rm = TRUE)
            n_used <- sum(!is.na(tau$tau))
            se <- if (n_used > 1) sd(tau$tau, na.rm = TRUE) / sqrt(n_used) else NA_real_
            # t-based CI (small n_treated); a 95% t-CI excludes zero iff the
            # two-sided t-p < 0.05, matching the table's significance stars.
            tcrit <- if (n_used > 1) qt(0.975, df = n_used - 1) else NA_real_
            ci_low <- att - tcrit * se
            ci_high <- att + tcrit * se
            t_stat <- if (!is.na(se) && se > 0) att / se else NA_real_
        } else {
            att <- NA_real_; n_used <- 0L; se <- NA_real_
            ci_low <- NA_real_; ci_high <- NA_real_; t_stat <- NA_real_
        }
        path1_results_summary[[paste(outcome, label, sep = "::")]] <- data.frame(
            outcome = outcome, post_label = label, post_threshold = post_thr,
            n_treated = n_used, att = att, se = se,
            ci_low = ci_low, ci_high = ci_high, t_stat = t_stat
        )
        cat(sprintf("  %-22s n=%2d  ATT=%+.4f  SE=%.4f  t=%+.2f\n",
                    outcome, n_used, att, se, t_stat))
    }
    cat("\n")
}

tau_df <- bind_rows(path1_results_tau)
summary_df <- bind_rows(path1_results_summary)
write_csv_lf(tau_df, OUT_TAU, row.names = FALSE)
write_csv_lf(summary_df, OUT_SUMMARY, row.names = FALSE)
cat("  wrote:", OUT_TAU, "(", nrow(tau_df), "rows )\n")
cat("  wrote:", OUT_SUMMARY, "(", nrow(summary_df), "rows )\n\n")

# ---------- console summary ----------
cat("==== Headline ATTs (collapsed DiD, post = EY >= 0) ====\n")
print(summary_df %>% filter(post_label == "headline") %>%
      select(outcome, n_treated, att, se, ci_low, ci_high, t_stat))

cat("\n==== Sensitivity (collapsed DiD, post = EY >= 2) ====\n")
print(summary_df %>% filter(post_label == "ey2plus") %>%
      select(outcome, n_treated, att, se, ci_low, ci_high, t_stat))
