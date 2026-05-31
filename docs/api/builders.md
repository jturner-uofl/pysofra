# Builders

The six top-level "builder" functions construct a fresh
[`SofraTable`][pysofra.core.table.SofraTable] from a DataFrame plus
some configuration. Every builder returns an immutable object that
can be chained through presentational modifiers (`.bold_p()`,
`.theme()`, `.set_caption()`, …) and rendered to any supported format.

Most builders also support statistical re-computation modifiers
(`.add_p()`, `.add_smd()`, …) that rebuild the table with additional
columns. **Exception:** `tbl_uvregression` bakes p-values and
confidence intervals in at build time. Use the `conf_level=` and
`digits=` constructor arguments to control formatting; calling
`.add_p()` on a `tbl_uvregression` result raises `NotImplementedError`
with an explanatory message.

::: pysofra.tbl_one

::: pysofra.tbl_summary

::: pysofra.tbl_cross

::: pysofra.tbl_survival

::: pysofra.tbl_regression

::: pysofra.tbl_uvregression
