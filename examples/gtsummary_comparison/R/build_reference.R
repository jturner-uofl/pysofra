#!/usr/bin/env Rscript
# ----------------------------------------------------------------------
# gtsummary::trial reference output for PySofra positioning.
#
# This script exports:
#   trial.csv        — the dataset used by both sides (200 rows × 8 cols)
#   gtsummary.json   — the table values gtsummary::tbl_summary(by=trt)
#                      produces, decomposed into per-cell summary stats
#                      so the Python side can be compared row-by-row.
#   gtsummary.html   — the rendered gtsummary HTML for visual inspection.
#
# Run:  Rscript R/build_reference.R
# ----------------------------------------------------------------------

suppressPackageStartupMessages({
  library(gtsummary)
  library(dplyr)
  library(jsonlite)
})

script_args <- commandArgs(trailingOnly = FALSE)
file_arg <- script_args[grepl("^--file=", script_args)]
script_dir <- if (length(file_arg) > 0) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1])))
} else getwd()
proj_dir <- normalizePath(file.path(script_dir, ".."))

data(trial, package = "gtsummary")
trial$response <- as.integer(trial$response)
trial$death    <- as.integer(trial$death)

# Save the dataset so the Python side reads the EXACT same numbers.
write.csv(trial, file.path(proj_dir, "trial.csv"), row.names = FALSE)

# Build the canonical gtsummary table — the default "tbl_summary by=trt"
# call that every gtsummary tutorial opens with.
tbl <- trial |>
  tbl_summary(by = trt, missing = "no") |>
  add_p()

# Save the rendered HTML for visual inspection.
gt::gtsave(as_gt(tbl), file.path(proj_dir, "gtsummary.html"))

# Extract the per-cell summary numbers into a structured JSON so the
# Python comparison can assert agreement.  We pull from tbl$table_body
# (gtsummary's internal data frame) which has one row per cell.
tb <- tbl$table_body
ref_rows <- list()
for (i in seq_len(nrow(tb))) {
  ref_rows[[i]] <- list(
    variable     = tb$variable[i],
    label        = tb$label[i],
    row_type     = tb$row_type[i],
    stat_drug_a  = if ("stat_1" %in% names(tb)) tb$stat_1[i] else NA,
    stat_drug_b  = if ("stat_2" %in% names(tb)) tb$stat_2[i] else NA,
    p_value      = if ("p.value" %in% names(tb))
                     suppressWarnings(as.numeric(tb$p.value[i])) else NA
  )
}

out <- list(
  meta = list(
    gtsummary_version = as.character(packageVersion("gtsummary")),
    R_version         = R.version.string,
    n_total           = nrow(trial),
    n_drug_a          = sum(trial$trt == "Drug A"),
    n_drug_b          = sum(trial$trt == "Drug B"),
    timestamp_utc     = format(Sys.time(), tz = "UTC",
                               "%Y-%m-%dT%H:%M:%SZ")
  ),
  table_body = ref_rows
)

writeLines(toJSON(out, pretty = TRUE, auto_unbox = TRUE, na = "null"),
           file.path(proj_dir, "gtsummary.json"))

cat(sprintf("Wrote: %s\n", file.path(proj_dir, "trial.csv")))
cat(sprintf("Wrote: %s\n", file.path(proj_dir, "gtsummary.json")))
cat(sprintf("Wrote: %s\n", file.path(proj_dir, "gtsummary.html")))

# Print a compact human-readable summary
cat("\n==== gtsummary table_body ====\n")
print(tb |> select(variable, label, row_type,
                   any_of(c("stat_1", "stat_2", "p.value"))),
      n = 50)
