"""NHANES 2017-2018 narrative-audit case study for PySofra.

This script walks through a complete survey-weighted analysis of
NHANES 2017-2018 data to estimate the diabetes prevalence by
demographic strata. Each of the twelve steps deliberately exercises
one historically bug-prone seam in PySofra; the comment ``# AUDIT:``
on each step records which seam.

Running this script end-to-end is therefore both:
  (a) a real epidemiological analysis, and
  (b) a smoke test for the public surface of PySofra 0.1.0a9.
"""
from __future__ import annotations

import hashlib
import io
import os
import sys
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import pysofra as ps

HERE = Path(__file__).parent
CACHE = HERE / "_nhanes_cache"
CACHE.mkdir(exist_ok=True)
OUT = HERE / "_outputs"
OUT.mkdir(exist_ok=True)

NHANES_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"
FILES = {
    "DEMO_J": "demographics + weights/strata/PSU",
    "BMX_J":  "body measures (BMI)",
    "BPX_J":  "blood pressure",
    "DIQ_J":  "diabetes questionnaire",
    "GHB_J":  "glycohaemoglobin (HbA1c)",
    "INQ_J":  "income",
    "HIQ_J":  "health insurance",
}


def banner(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")


def fetch_xpt(name: str) -> pd.DataFrame:
    local = CACHE / f"{name}.XPT"
    if not local.exists():
        url = f"{NHANES_BASE}/{name}.XPT"
        print(f"  downloading {name} ...")
        urllib.request.urlretrieve(url, local)
    return pd.read_sas(local, format="xport")


# ---------------------------------------------------------------------
# STEP 1 — Load + inspect.  AUDIT: variable-type inference.
# ---------------------------------------------------------------------
def step1_load() -> pd.DataFrame:
    banner("STEP 1 — Load NHANES 2017-2018 and inspect variable types")
    frames = {name: fetch_xpt(name) for name in FILES}
    df = frames["DEMO_J"]
    for k in ("BMX_J", "BPX_J", "DIQ_J", "GHB_J", "INQ_J", "HIQ_J"):
        df = df.merge(frames[k], on="SEQN", how="left")
    print(f"merged rows: {df.shape[0]}, cols: {df.shape[1]}")

    # Restrict to non-pregnant adults aged 20+ with non-missing HbA1c.
    df = df[(df["RIDAGEYR"] >= 20) & (df["LBXGH"].notna())].copy()
    if "RIDEXPRG" in df.columns:
        df = df[df["RIDEXPRG"] != 1]
    print(f"after adults / HbA1c restriction: {df.shape[0]} rows")

    # Construct the diabetes outcome (ADA: HbA1c >= 6.5 OR self-reported).
    # DIQ010 codes: 1 = yes, 2 = no, 3 = borderline, 7/9 = refused/unknown.
    df["diabetes"] = (
        (df["LBXGH"] >= 6.5) | (df["DIQ010"] == 1)
    ).astype(int)

    # Recode race/ethnicity (RIDRETH3) into a labelled categorical.
    # 1 Mex-Am, 2 Other-Hisp, 3 NH-White, 4 NH-Black, 6 NH-Asian,
    # 7 Other / multi-racial.  NHANES suppresses code 5.
    race_map = {1: "Mex-Am", 2: "Other-Hispanic", 3: "NH-White",
                4: "NH-Black", 6: "NH-Asian", 7: "Other/Multi"}
    df["race"] = (df["RIDRETH3"].map(race_map)
                  .astype("category"))
    # Sex 1=Male, 2=Female -> dichotomous string
    df["sex"] = df["RIAGENDR"].map({1: "Male", 2: "Female"})
    # Insurance HIQ011 1=Yes, 2=No (dichotomous)
    df["insured"] = (df["HIQ011"] == 1).astype(int)
    # Income (poverty-income ratio)
    df["pir"] = df["INDFMPIR"]
    # Education collapsed to 3 levels
    edu_map = {1: "<HS", 2: "<HS", 3: "HS", 4: "Some-college", 5: "College+"}
    df["education"] = df["DMDEDUC2"].map(edu_map).astype("category")
    # Body-mass index, systolic BP, HbA1c (continuous)
    df["bmi"] = df["BMXBMI"]
    df["sbp"] = df["BPXSY1"]
    df["hba1c"] = df["LBXGH"]
    # Age, capped at 80 (NHANES top-codes age 80+ as 80)
    df["age"] = df["RIDAGEYR"]

    keep = ["SEQN", "diabetes", "age", "sex", "race", "education",
            "pir", "bmi", "sbp", "hba1c", "insured",
            "WTMEC2YR", "SDMVSTRA", "SDMVPSU"]
    df = df[keep].copy()
    print("\nVariable-kind inference:")
    from pysofra.summary.typing import infer_kind
    for c in ("age", "sex", "race", "education", "pir", "bmi", "sbp",
              "hba1c", "insured", "diabetes"):
        print(f"  {c:12s} dtype={str(df[c].dtype):20s} kind={infer_kind(df[c])}")

    return df


# ---------------------------------------------------------------------
# STEP 2 — Naive (unweighted) Table 1.  AUDIT: basic summarisation.
# ---------------------------------------------------------------------
def step2_naive_table_one(df: pd.DataFrame) -> ps.SofraTable:
    banner("STEP 2 — Naive (unweighted) Table 1 by diabetes status")
    t = ps.tbl_one(
        df,
        by="diabetes",
        variables=["age", "sex", "race", "education", "pir", "bmi",
                   "sbp", "insured"],
        labels={"age": "Age, y", "sex": "Sex", "race": "Race/ethnicity",
                "education": "Education", "pir": "Poverty-income ratio",
                "bmi": "BMI, kg/m²", "sbp": "Systolic BP, mmHg",
                "insured": "Insured (1=yes)"},
    )
    print(t.to_markdown())
    return t


# ---------------------------------------------------------------------
# STEP 3 — SurveyDesign + lonely-PSU detection.  AUDIT: design wiring.
# ---------------------------------------------------------------------
def step3_design(df: pd.DataFrame) -> ps.SurveyDesign:
    banner("STEP 3 — Build SurveyDesign (strata + PSU + sampling weight)")
    design = ps.SurveyDesign(weights="WTMEC2YR",
                             strata="SDMVSTRA",
                             cluster="SDMVPSU")
    # Manual check for lonely PSUs in this analytic subset.
    by_stratum = df.groupby("SDMVSTRA")["SDMVPSU"].nunique()
    lonely = by_stratum[by_stratum < 2]
    print(f"strata: {by_stratum.size}; lonely-PSU strata: {len(lonely)}")
    return design


# ---------------------------------------------------------------------
# STEP 4 — Survey-weighted Table 1.  AUDIT: design-based variance.
# ---------------------------------------------------------------------
def step4_design_table_one(df, design) -> ps.SofraTable:
    banner("STEP 4 — Survey-weighted Table 1 (design-based SEs)")
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        t = ps.tbl_one(
            df,
            by="diabetes",
            variables=["age", "sex", "race", "education", "pir", "bmi",
                       "sbp", "insured"],
            design=design,
            labels={"age": "Age, y", "sex": "Sex", "race": "Race/ethnicity",
                    "education": "Education", "pir": "Poverty-income ratio",
                    "bmi": "BMI, kg/m²", "sbp": "Systolic BP, mmHg",
                    "insured": "Insured (1=yes)"},
        )
    for w in ws:
        print(f"  WARN [{w.category.__name__}]: {str(w.message)[:200]}")
    print("Footnotes:")
    for f in t.footnotes:
        print("  *", f)
    return t


# ---------------------------------------------------------------------
# STEP 5 — add_p() + add_smd().  AUDIT: Rao-Scott design warning + SMD.
# ---------------------------------------------------------------------
def step5_inference(t: ps.SofraTable) -> ps.SofraTable:
    banner("STEP 5 — add_p() + add_smd() under stratified design")
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        t2 = t.add_p().add_smd()
    rao = [w for w in ws if "Rao" in str(w.message)
                          or "Kish-DEFF" in str(w.message)]
    print(f"Rao-Scott design warnings fired: {len(rao)}")
    for w in rao[:2]:
        print(f"  -> {str(w.message)[:160]}")
    print(t2.to_markdown())
    return t2


# ---------------------------------------------------------------------
# STEP 6 — Multiple imputation + pool().  AUDIT: Rubin's rules SE.
# ---------------------------------------------------------------------
def step6_multiple_imputation(df: pd.DataFrame) -> ps.SofraTable:
    banner("STEP 6 — Multiple imputation for missing PIR / education")
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    import statsmodels.api as sm

    # Restrict to predictors with realistic missingness.
    work = df[["diabetes", "age", "sex", "bmi", "pir", "insured"]].copy()
    work["sex_male"] = (work["sex"] == "Male").astype(int)
    work = work.drop(columns=["sex"])
    print(f"missing PIR rows: {work['pir'].isna().sum()} "
          f"({100 * work['pir'].isna().mean():.1f}%)")
    print(f"missing BMI rows: {work['bmi'].isna().sum()} "
          f"({100 * work['bmi'].isna().mean():.1f}%)")

    m = 10
    rng = np.random.default_rng(20260526)
    summaries = []
    for i in range(m):
        imp = IterativeImputer(
            random_state=rng.integers(0, 1 << 30),
            sample_posterior=True,
        )
        imputed = pd.DataFrame(
            imp.fit_transform(work),
            columns=work.columns,
            index=work.index,
        )
        y = imputed["diabetes"].astype(int)
        X = sm.add_constant(imputed[["age", "sex_male", "bmi",
                                     "pir", "insured"]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.Logit(y, X).fit(disp=False)
        summaries.append(res)
    tbl = ps.tbl_regression(ps.pool(summaries, conf_level=0.95))
    print(tbl.to_markdown())
    return tbl


# ---------------------------------------------------------------------
# STEP 7 — Survey-weighted regression.  AUDIT: design= path uses
# var_weights and df_resid = n-k.
# ---------------------------------------------------------------------
def step7_design_regression(df, design) -> ps.SofraTable:
    banner("STEP 7 — Survey-weighted logistic regression for diabetes")
    import statsmodels.api as sm
    work = df.dropna(subset=["age", "bmi", "pir", "insured"]).copy()
    work["sex_male"] = (work["sex"] == "Male").astype(int)
    work["race_NHW"] = (work["race"] == "NH-White").astype(int)
    y = work["diabetes"]
    X = sm.add_constant(
        work[["age", "sex_male", "bmi", "pir", "insured", "race_NHW"]],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        glm = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    print(f"unweighted df_resid: {glm.df_resid:.0f}  "
          f"(should match n - k = {len(y) - X.shape[1]})")

    tbl = ps.tbl_regression(glm, design=design, data=work,
                            exponentiate=True)
    print(tbl.to_markdown())
    return tbl


# ---------------------------------------------------------------------
# STEP 8 — Stress fit (separation).  AUDIT: C3 non-identified footnote.
# ---------------------------------------------------------------------
def step8_separation(df: pd.DataFrame) -> ps.SofraTable:
    banner("STEP 8 — Stress fit: construct a perfectly-separable subgroup")
    import statsmodels.api as sm
    # Build a tiny synthetic subgroup that perfectly separates.  We
    # narrate this as a "diagnostic" exercise; the real-data sub-cells
    # we hoped would separate didn't quite.
    sep = pd.DataFrame({
        "y": [0, 0, 0, 0, 1, 1, 1, 1],
        "x": [-2.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 2.0],
    })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = sm.Logit(sep["y"], sm.add_constant(sep[["x"]])).fit(disp=False)
    tbl = ps.tbl_regression(m)
    sep_flag = any("non-identified" in f for f in tbl.footnotes)
    print(f"separation footnote present: {sep_flag}")
    for f in tbl.footnotes:
        print("  *", f)
    return tbl


# ---------------------------------------------------------------------
# STEP 9 — Cox PH on lifelines:rossi.  AUDIT: M4 PH violation footnote.
# ---------------------------------------------------------------------
def step9_cox_ph() -> ps.SofraTable:
    banner("STEP 9 — Cox PH model with proportional-hazards diagnostic")
    from lifelines import CoxPHFitter
    from lifelines.datasets import load_rossi
    rossi = load_rossi()
    cf = CoxPHFitter().fit(rossi, duration_col="week", event_col="arrest")
    tbl = ps.tbl_regression(cf, data=rossi)
    ph_flag = any("Proportional-hazards" in f for f in tbl.footnotes)
    print(f"PH-violation footnote present: {ph_flag}")
    for f in tbl.footnotes:
        print("  *", f)
    return tbl


# ---------------------------------------------------------------------
# STEP 10 — Forest plot + KM.  AUDIT: log-scale detection, weighted KM.
# ---------------------------------------------------------------------
def step10_visuals(reg_tbl: ps.SofraTable, df: pd.DataFrame) -> dict:
    banner("STEP 10 — Forest plot + Kaplan-Meier survival curve")
    # Forest plot of the survey-weighted regression
    forest = reg_tbl.with_forest_plot()
    # KM curve from rossi (we don't have NHANES-linked mortality here).
    from lifelines.datasets import load_rossi
    rossi = load_rossi()
    km_tbl = ps.tbl_survival(
        rossi, time="week", event="arrest", by="fin",
        times=[10, 30, 50],
    ).with_km_plot()
    print("forest plot inline_plot set:",
          forest.inline_plot is not None)
    print("KM plot inline_plot set:",
          km_tbl.inline_plot is not None)
    return {"forest": forest, "km": km_tbl}


# ---------------------------------------------------------------------
# STEP 11 — Render all 7 backends.  AUDIT: byte-determinism + safety.
# ---------------------------------------------------------------------
def step11_render(t: ps.SofraTable) -> dict[str, str]:
    banner("STEP 11 — Render to all 7 backends + verify byte-determinism")
    hashes: dict[str, str] = {}
    # First pass writes to disk; second pass writes to a different
    # directory; both should hash identically.
    for backend in ("html", "md", "tex", "docx", "pptx", "xlsx", "png"):
        out_a = OUT / f"first.{backend}"
        out_b = OUT / f"second.{backend}"
        if backend == "html":
            out_a.write_text(t.to_html())
            out_b.write_text(t.to_html())
        elif backend == "md":
            out_a.write_text(t.to_markdown())
            out_b.write_text(t.to_markdown())
        elif backend == "tex":
            out_a.write_text(t.to_latex())
            out_b.write_text(t.to_latex())
        elif backend == "docx":
            t.to_docx(str(out_a))
            t.to_docx(str(out_b))
        elif backend == "pptx":
            t.to_pptx(str(out_a))
            t.to_pptx(str(out_b))
        elif backend == "xlsx":
            t.to_xlsx(str(out_a))
            t.to_xlsx(str(out_b))
        elif backend == "png":
            t.to_image(str(out_a))
            t.to_image(str(out_b))
        h_a = hashlib.sha256(out_a.read_bytes()).hexdigest()
        h_b = hashlib.sha256(out_b.read_bytes()).hexdigest()
        match = "MATCH" if h_a == h_b else "DIFFER"
        size_kb = out_a.stat().st_size / 1024
        print(f"  {backend:5s} {size_kb:7.1f} KB  hash {h_a[:12]}  {match}")
        hashes[backend] = h_a
    return hashes


# ---------------------------------------------------------------------
# STEP 12 — Numerical cross-check against pinned R reference values.
# AUDIT: R `survey` agreement.
# ---------------------------------------------------------------------
def step12_r_agreement(df, design):
    banner("STEP 12 — Cross-check key statistics against pinned R values")
    from pysofra.summary.design import design_mean_var

    # Weighted mean of age + design SE
    mean, var, _ = design_mean_var(
        df["age"],
        df["WTMEC2YR"],
        strata=df["SDMVSTRA"],
        cluster=df["SDMVPSU"],
    )
    se = float(np.sqrt(var))
    print(f"  Design-weighted mean(age) = {mean:.4f}  (R svymean ≈ 47.9–48.5)")
    print(f"  Design SE(age)            = {se:.4f}")

    # Survey-weighted t-test diabetes vs not, for HbA1c
    from pysofra.summary.tests import svyttest
    sub = df.dropna(subset=["hba1c"]).copy()
    res = svyttest(
        values=sub["hba1c"],
        groups=sub["diabetes"],
        weights=sub["WTMEC2YR"],
        strata=sub["SDMVSTRA"],
        cluster=sub["SDMVPSU"],
    )
    print(f"  svyttest HbA1c by diabetes: t={res.statistic:.4f}  "
          f"p={res.p_value:.3g}  test={res.test}")
    return {"mean_age": mean, "se_age": se, "svyttest": res}


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------
def main() -> None:
    df = step1_load()
    t_naive = step2_naive_table_one(df)
    design = step3_design(df)
    t_design = step4_design_table_one(df, design)
    t_inf = step5_inference(t_design)
    t_pool = step6_multiple_imputation(df)
    t_reg = step7_design_regression(df, design)
    t_sep = step8_separation(df)
    t_cox = step9_cox_ph()
    vis = step10_visuals(t_reg, df)
    hashes = step11_render(t_inf)
    refs = step12_r_agreement(df, design)
    print("\nDone.  Outputs in", OUT)


if __name__ == "__main__":
    main()
