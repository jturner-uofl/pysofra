#!/usr/bin/env Rscript
# ----------------------------------------------------------------------
# NHANES 2017-2018 cross-validation reference values for PySofra.
#
# This script reproduces, in R via the `survey` package (Lumley 2004,
# JSS 9(8); 2010 book), the key survey-weighted statistics that the
# PySofra narrative-audit notebook computes in Python.  Output is a
# JSON blob the notebook reads in Step 12 to display side-by-side
# agreement.
#
# Run:  Rscript R/cross_validate.R [cache_dir] [output_path]
#
#   cache_dir   default: ../_nhanes_cache (same XPT cache the notebook uses)
#   output_path default: ../R_reference.json
#
# Requirements: survey, haven, jsonlite, dplyr.
# ----------------------------------------------------------------------

suppressPackageStartupMessages({
  library(survey)
  library(haven)
  library(jsonlite)
  library(dplyr)
})

args <- commandArgs(trailingOnly = TRUE)

# Resolve defaults relative to this script's own location so the
# script works whether you invoke it from the examples/jss_case_study
# directory, from R/, or via an absolute path.
script_args <- commandArgs(trailingOnly = FALSE)
file_arg    <- script_args[grepl("^--file=", script_args)]
if (length(file_arg) > 0) {
  script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg[1])))
} else {
  script_dir <- getwd()
}
proj_dir <- normalizePath(file.path(script_dir, ".."))

cache_dir   <- if (length(args) >= 1) args[1] else file.path(proj_dir, "_nhanes_cache")
output_path <- if (length(args) >= 2) args[2] else file.path(proj_dir, "R_reference.json")

if (!dir.exists(cache_dir)) {
  stop(sprintf(
    "NHANES cache not found at %s. Run the notebook once (it auto-downloads).",
    cache_dir
  ))
}

# ----------------------------------------------------------------------
# Load and merge — mirrors Step 1 of the notebook exactly.
# ----------------------------------------------------------------------
read_xpt <- function(name) {
  haven::read_xpt(file.path(cache_dir, paste0(name, ".XPT")))
}
demo <- read_xpt("DEMO_J")
bmx  <- read_xpt("BMX_J")
bpx  <- read_xpt("BPX_J")
diq  <- read_xpt("DIQ_J")
ghb  <- read_xpt("GHB_J")
inq  <- read_xpt("INQ_J")
hiq  <- read_xpt("HIQ_J")

df <- demo %>%
  left_join(bmx, by = "SEQN") %>%
  left_join(bpx, by = "SEQN") %>%
  left_join(diq, by = "SEQN") %>%
  left_join(ghb, by = "SEQN") %>%
  left_join(inq, by = "SEQN") %>%
  left_join(hiq, by = "SEQN") %>%
  filter(RIDAGEYR >= 20, !is.na(LBXGH))

# Drop pregnant women if the column is present.
if ("RIDEXPRG" %in% names(df)) {
  df <- df %>% filter(is.na(RIDEXPRG) | RIDEXPRG != 1)
}

# Outcome definition — ADA criterion + self-report. (DIQ010 codes:
# 1 yes, 2 no, 3 borderline, 7/9 refused/don't-know.)
df$diabetes <- as.integer((df$LBXGH >= 6.5) | (df$DIQ010 == 1))

cat(sprintf("Analytic n = %d\n", nrow(df)))

# ----------------------------------------------------------------------
# Survey design — same strata / PSU / weights as PySofra.
# ----------------------------------------------------------------------
options(survey.lonely.psu = "adjust")  # mirrors PySofra's contribute-zero
des <- svydesign(
  ids     = ~SDMVPSU,
  strata  = ~SDMVSTRA,
  weights = ~WTMEC2YR,
  nest    = TRUE,
  data    = df
)

# ----------------------------------------------------------------------
# 1. svymean(~RIDAGEYR)  →  mean + Taylor-linearised SE.
# ----------------------------------------------------------------------
m_age <- svymean(~RIDAGEYR, des, na.rm = TRUE)
mean_age <- as.numeric(coef(m_age))
se_age   <- as.numeric(SE(m_age))

# ----------------------------------------------------------------------
# 2. svymean(~BMXBMI)  →  same for BMI (independent of outcome).
# ----------------------------------------------------------------------
m_bmi <- svymean(~BMXBMI, des, na.rm = TRUE)
mean_bmi <- as.numeric(coef(m_bmi))
se_bmi   <- as.numeric(SE(m_bmi))

# ----------------------------------------------------------------------
# 3. svyttest(BMXBMI ~ diabetes) — design-adjusted Welch-type t.
# BMI is independent of the outcome definition, so this is a real test.
# ----------------------------------------------------------------------
des_bmi <- subset(des, !is.na(BMXBMI))
tt_bmi <- svyttest(BMXBMI ~ diabetes, des_bmi)
tt_bmi_t  <- as.numeric(tt_bmi$statistic)
tt_bmi_p  <- as.numeric(tt_bmi$p.value)
tt_bmi_df <- as.numeric(tt_bmi$parameter)

# ----------------------------------------------------------------------
# 4. svyttest(BPXSY1 ~ diabetes) — second independent comparison.
# ----------------------------------------------------------------------
des_sbp <- subset(des, !is.na(BPXSY1))
tt_sbp <- svyttest(BPXSY1 ~ diabetes, des_sbp)
tt_sbp_t  <- as.numeric(tt_sbp$statistic)
tt_sbp_p  <- as.numeric(tt_sbp$p.value)
tt_sbp_df <- as.numeric(tt_sbp$parameter)

# ----------------------------------------------------------------------
# 5. svychisq(~diabetes + RIDRETH3) — full second-order Rao-Scott.
# PySofra uses the first-order Kish-DEFF approximation; this is the
# reference for the "10-15% disagreement" claim in the docstring.
# ----------------------------------------------------------------------
chi_race <- svychisq(~diabetes + RIDRETH3, des, statistic = "Chisq")
chi_race_stat <- as.numeric(chi_race$statistic)
chi_race_p    <- as.numeric(chi_race$p.value)

# ----------------------------------------------------------------------
# 6. svyglm(diabetes ~ ...) — survey-weighted logistic regression.
# Compare ORs and SEs to PySofra Step 7.
# ----------------------------------------------------------------------
mod_df <- df %>%
  mutate(
    sex_male = as.integer(RIAGENDR == 1),
    race_NHW = as.integer(RIDRETH3 == 3),
    pir = INDFMPIR,
    bmi = BMXBMI,
    insured = as.integer(HIQ011 == 1)
  ) %>%
  filter(!is.na(bmi), !is.na(pir), !is.na(insured))

# Rebuild the design on the complete-case subset; this matches PySofra's
# Step 7 (which drops rows missing any predictor before calling .fit()).
des_mod <- svydesign(
  ids = ~SDMVPSU, strata = ~SDMVSTRA, weights = ~WTMEC2YR,
  nest = TRUE, data = mod_df
)

fit <- svyglm(
  diabetes ~ RIDAGEYR + sex_male + bmi + pir + insured + race_NHW,
  design = des_mod,
  family = quasibinomial()
)

coefs <- summary(fit)$coefficients
# `svyglm` with quasibinomial reports t-statistics (not z) because the
# residual df is finite under the design (n_PSU − n_strata).  The
# column name in `summary()` is `"t value"` and the p-value column
# is `"Pr(>|t|)"`.
ors <- exp(coefs[, "Estimate"])
ses <- coefs[, "Std. Error"]
ts  <- coefs[, "t value"]
ps  <- coefs[, "Pr(>|t|)"]

reg <- list(
  variable = rownames(coefs),
  estimate = unname(coefs[, "Estimate"]),
  std_error = unname(ses),
  odds_ratio = unname(ors),
  t_value = unname(ts),
  p_value = unname(ps)
)

# ----------------------------------------------------------------------
# Assemble + write the JSON.
# ----------------------------------------------------------------------
out <- list(
  meta = list(
    R_version       = R.version.string,
    survey_version  = as.character(packageVersion("survey")),
    haven_version   = as.character(packageVersion("haven")),
    analytic_n      = nrow(df),
    analytic_n_mod  = nrow(mod_df),
    cache_dir       = normalizePath(cache_dir),
    timestamp_utc   = format(Sys.time(), tz = "UTC",
                             "%Y-%m-%dT%H:%M:%SZ")
  ),
  svymean = list(
    age_mean = mean_age, age_se = se_age,
    bmi_mean = mean_bmi, bmi_se = se_bmi
  ),
  svyttest = list(
    bmi_t = tt_bmi_t,   bmi_p = tt_bmi_p,   bmi_df = tt_bmi_df,
    sbp_t = tt_sbp_t,   sbp_p = tt_sbp_p,   sbp_df = tt_sbp_df
  ),
  svychisq = list(
    race_diabetes_stat = chi_race_stat,
    race_diabetes_p    = chi_race_p
  ),
  svyglm = reg
)

writeLines(toJSON(out, pretty = TRUE, auto_unbox = TRUE, digits = NA),
           output_path)
cat(sprintf("\nWrote reference values to: %s\n", output_path))

# ----------------------------------------------------------------------
# Human-readable summary
# ----------------------------------------------------------------------
cat("\n==== R reference values ====\n")
cat(sprintf("  svymean(~RIDAGEYR)          = %.6f   (SE %.6f)\n",
            mean_age, se_age))
cat(sprintf("  svymean(~BMXBMI)            = %.6f   (SE %.6f)\n",
            mean_bmi, se_bmi))
cat(sprintf("  svyttest(BMXBMI ~ diabetes) = %.6f   (p = %.3g, df = %.1f)\n",
            tt_bmi_t, tt_bmi_p, tt_bmi_df))
cat(sprintf("  svyttest(BPXSY1 ~ diabetes) = %.6f   (p = %.3g, df = %.1f)\n",
            tt_sbp_t, tt_sbp_p, tt_sbp_df))
cat(sprintf("  svychisq(diabetes + race)   = %.4f   (p = %.4f)\n",
            chi_race_stat, chi_race_p))
cat("\n  svyglm logistic regression coefficients:\n")
for (i in seq_along(reg$variable)) {
  cat(sprintf("    %-13s  beta=%8.4f  SE=%6.4f  OR=%6.3f  t=%7.3f  p=%.3g\n",
              reg$variable[i], reg$estimate[i], reg$std_error[i],
              reg$odds_ratio[i], reg$t_value[i], reg$p_value[i]))
}
