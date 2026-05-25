# Table 1 — `tbl_one()`

The bread-and-butter call: stratified baseline characteristics with
auto-selected tests, SMDs, missing summaries, and an overall column.

## Anatomy

```python
ps.tbl_one(
    data,                   # pandas or polars DataFrame
    by="arm",               # stratification column (optional)
    variables=[...],        # which columns to include + order
    labels={...},           # display labels
    types={...},            # override automatic variable typing
    nonnormal=[...],        # use median (Q1, Q3) + rank tests
    tests={...},            # per-variable test overrides
    digits=2,
    pct_digits=1,
    missing="ifany",        # "ifany" | "always" | "never"
)
```

## Variable typing

PySofra classifies each column as `continuous`, `categorical`,
`dichotomous`, or `ordinal`:

- `bool` → dichotomous
- exactly `{0, 1}` ints → dichotomous
- few-level small ints → categorical
- string with 2 levels → dichotomous, more → categorical
- ordered `pd.Categorical` → ordinal
- numeric → continuous

Override with `types={'smoker': 'categorical'}`.

## Test selection

| Variable kind | Default test |
|---|---|
| Continuous, 2 groups | Welch's t-test |
| Continuous, 3+ groups | One-way ANOVA |
| Continuous + `nonnormal=[...]` | Wilcoxon / Kruskal–Wallis |
| Categorical, 2×2 | Fisher's exact |
| Categorical, larger | Pearson χ² |

Override per variable: `tests={'age': 'wilcoxon', 'race': 'fisher'}`.

## Modifiers

| Method | Effect |
|---|---|
| `.add_p()` | p-value column |
| `.add_q(method='fdr_bh')` | q-value column (multiplicity-adjusted) |
| `.add_smd()` | standardised mean difference column |
| `.add_overall(label='Overall')` | unstratified column |
| `.bold_p(threshold=0.05)` | bold rows below threshold |
| `.theme('clinical')` | apply a theme |
| `.set_caption(text)` / `.add_footnote(text)` | annotations |

All modifiers return a new `SofraTable` (immutable).
