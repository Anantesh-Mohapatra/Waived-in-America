# Install the R packages the pipeline needs (see r_requirements.md for the
# exact verified versions). Installs only what is missing.

cran_pkgs <- c("arrow", "MatchIt", "dplyr", "tidyr",
               "future", "future.apply", "digest", "remotes")
missing <- setdiff(cran_pkgs, rownames(installed.packages()))
if (length(missing)) {
  install.packages(missing, repos = "https://cloud.r-project.org")
}

# synthdid is NOT on CRAN; install the verified 0.0.9 from GitHub, pinned to
# the commit whose DESCRIPTION reads Version 0.0.9 (the repo has no tags).
if (!"synthdid" %in% rownames(installed.packages())) {
  remotes::install_github("synth-inference/synthdid@70c1ce3eac58e28c30b67435ca377bb48baa9b8a")
}

cat("R package check complete.\n")
for (p in c(setdiff(cran_pkgs, "remotes"), "synthdid")) {
  cat(sprintf("  %-14s %s\n", p,
              tryCatch(as.character(packageVersion(p)), error = function(e) "MISSING")))
}
