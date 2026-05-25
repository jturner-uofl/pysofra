# Builders

The six top-level "builder" functions construct a fresh
[`SofraTable`][pysofra.core.table.SofraTable] from a DataFrame plus
some configuration. Every builder returns an immutable object that
can be chained through statistical modifiers (`.add_p()`,
`.add_smd()`, …) and rendered to any supported format.

::: pysofra.tbl_one

::: pysofra.tbl_summary

::: pysofra.tbl_cross

::: pysofra.tbl_survival

::: pysofra.tbl_regression

::: pysofra.tbl_uvregression
