# Quickstart

## Install

```bash
pip install pysofra
```

PySofra requires Python ≥ 3.11. Optional extras:

```bash
pip install pysofra[pptx]   # PowerPoint export
pip install pysofra[dev]    # testing + linting
```

## Your first Table 1

```python
import pandas as pd
import pysofra as ps

df = pd.read_csv("trial.csv")

table = (
    ps.tbl_one(df, by="arm")
      .add_p()
      .add_smd()
      .add_overall()
      .theme("clinical")
)

table                          # renders in Jupyter / Colab / VS Code
table.to_docx("table1.docx")   # publication-quality Word
table.to_latex()               # booktabs LaTeX
table.to_html()                # standalone HTML fragment
```

## A regression table

```python
import statsmodels.api as sm

X = sm.add_constant(df[["age", "bmi"]])
fit = sm.Logit(df["event"], X).fit(disp=False)

(
    ps.tbl_regression(fit, exponentiate=True)
      .bold_p()
      .theme("jama")
      .to_docx("table2.docx")
)
```

## Multi-model side-by-side

```python
fit_uni = sm.Logit(df["event"], sm.add_constant(df[["age"]])).fit(disp=False)
fit_adj = sm.Logit(df["event"], sm.add_constant(df[["age", "bmi"]])).fit(disp=False)

ps.tbl_regression(
    [fit_uni, fit_adj],
    exponentiate=True,
    model_labels=["Unadjusted", "Adjusted"],
).theme("jama")
```

## polars works too

```python
import polars as pl

pl_df = pl.read_csv("trial.csv")
ps.tbl_one(pl_df, by="arm").add_p()
```

## Multiplicity adjustment

```python
ps.tbl_one(df, by="arm").add_p().add_q(method="fdr_bh")
```

See the [guides](guides/tbl_one.md) for in-depth coverage.
