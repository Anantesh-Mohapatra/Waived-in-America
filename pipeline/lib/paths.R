# Results/data layout derivation shared by all pipeline R scripts.
# Mirrors pipeline/lib/paths.py (the Python owner of the same layout).
#
# Usage — AFTER the inline find_root() block (which cannot be shared because
# a script must know the root before it can source anything):
#
#   ROOT <- find_root()
#   source(file.path(ROOT, "pipeline", "lib", "paths.R"))
#   VARIANT <- wia_variant()

wia_variant <- function() {
  v <- Sys.getenv("WIA_VARIANT", "main")
  if (!v %in% c("main", "dla_only")) {
    stop("WIA_VARIANT must be 'main' or 'dla_only'; got: ", v)
  }
  cat("  variant:", v, "\n")
  v
}

# Filename-suffix rule: main is unsuffixed, every other variant is _<variant>.
v_suffix_for <- function(variant) if (variant == "main") "" else paste0("_", variant)

anchored_data_dir    <- function(root) file.path(root, "data", "anchored")
donor_pools_dir      <- function(root, variant) file.path(root, "data", "donor_pools", variant)
synth_partial_dir    <- function(root, variant) file.path(root, "data", "_partial", paste0("synth_", variant))
synth_trial_dir      <- function(root, variant) file.path(root, "data", "synth_trial", variant)

matched_tables_dir   <- function(root, variant) file.path(root, "results", variant, "matched", "tables")
matched_matching_dir <- function(root, variant) file.path(root, "results", variant, "matched", "matching")
matched_logs_dir     <- function(root, variant) file.path(root, "results", variant, "matched", "logs")
synth_tables_dir     <- function(root, variant) file.path(root, "results", variant, "synth", "tables")
synth_logs_dir       <- function(root, variant) file.path(root, "results", variant, "synth", "logs")

# LF-output writers. On Windows a text-mode connection translates \n -> \r\n,
# so R's write.csv/writeLines produce CRLF by default; a binary ("wb")
# connection writes the bytes verbatim, giving LF on every platform. This keeps
# the committed text outputs byte-identical to the LF reference regardless of OS.
write_csv_lf <- function(x, path, ...) {
  con <- file(path, "wb"); on.exit(close(con))
  write.csv(x, con, ...)
}
write_lines_lf <- function(lines, path) {
  con <- file(path, "wb"); on.exit(close(con))
  writeLines(lines, con, sep = "\n")
}
