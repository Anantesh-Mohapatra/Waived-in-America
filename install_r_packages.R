# Install the R packages the pipeline needs (see r_requirements.md for the
# exact verified versions). Installs only what is missing.

cran_pkgs <- c("arrow", "MatchIt", "dplyr", "tidyr",
               "future", "future.apply", "digest", "remotes")
missing <- setdiff(cran_pkgs, rownames(installed.packages()))
if (length(missing)) {
  install.packages(missing, repos = "https://cloud.r-project.org")
}

# synthdid is NOT on CRAN; the verified version (0.0.9) comes from GitHub.
if (!"synthdid" %in% rownames(installed.packages())) {
  remotes::install_github("synth-inference/synthdid")
}

# Strict pinning to the verified versions (optional; uncomment to use):
# remotes::install_version("arrow",        "23.0.1.2")
# remotes::install_version("MatchIt",      "4.7.2")
# remotes::install_version("dplyr",        "1.1.4")
# remotes::install_version("tidyr",        "1.3.1")
# remotes::install_version("future",       "1.69.0")
# remotes::install_version("future.apply", "1.20.2")
# remotes::install_version("digest",       "0.6.37")

cat("R package check complete.\n")
for (p in c(setdiff(cran_pkgs, "remotes"), "synthdid")) {
  cat(sprintf("  %-14s %s\n", p,
              tryCatch(as.character(packageVersion(p)), error = function(e) "MISSING")))
}
