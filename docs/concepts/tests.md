# Statistical tests

PySofra auto-selects per-variable tests following the `tableone` /
`gtsummary` conventions. You can override per variable with `tests=`.

## Defaults

| Variable kind | 2 groups | 3+ groups |
|---|---|---|
| Continuous | Welch's t-test | One-way ANOVA |
| Continuous + `nonnormal=` | Wilcoxon rank-sum | Kruskal–Wallis |
| Dichotomous / categorical (2×2) | Fisher's exact | — |
| Categorical (larger) | — | Pearson χ² (flagged sparse if any expected < 5) |

## Named-test registry

```python
import pysofra as ps
ps.available_tests()
```

| Continuous | Categorical |
|---|---|
| `welch` / `welch_t` / `ttest` / `t` | `fisher` / `fisher_exact` |
| `student` / `student_t` / `equal_var_t` | `chisq` / `chi_square` / `chi2` / `pearson` |
| `wilcoxon` / `mannwhitney` / `mwu` / `rank_sum` | |
| `anova` / `oneway_anova` | |
| `kruskal` / `kruskal_wallis` | |

Override per variable:

```python
ps.tbl_one(df, by='arm',
           tests={'hba1c': 'wilcoxon', 'race': 'fisher'}).add_p()
```

The footnote automatically names both the overrides and any defaults
that fired elsewhere.

## Multiplicity adjustment

`.add_q(method='fdr_bh')` adds a q-value column. Methods come from
`statsmodels.stats.multitest.multipletests`:

| Method | Description |
|---|---|
| `fdr_bh` (default) | Benjamini–Hochberg |
| `fdr_by` | Benjamini–Yekutieli |
| `bonferroni` | Bonferroni |
| `holm` | Holm–Bonferroni |
| `hommel` | Hommel |
| `sidak` | Šidák |
