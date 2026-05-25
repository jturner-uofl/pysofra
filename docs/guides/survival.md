# Survival — `tbl_survival()`

Produces a Kaplan–Meier summary table with median survival, N at risk
at fixed time points, and the multivariate log-rank test. Requires
the optional `lifelines` dependency.

```python
import pysofra as ps

ps.tbl_survival(
    df,
    time='followup_months',
    event='died',
    by='arm',
    times=[12, 24, 36],
    times_label='mo',
).theme('clinical')
```

## Output rows

| Row | Description |
|---|---|
| **N** | Sample size per group |
| **Events** | Number of event observations |
| **Censored** | Number of censored observations |
| **Median survival (CI)** | Kaplan–Meier median with confidence interval |
| **S(t = X)** | Survival probability at each requested time point, with N at risk |

When `by=` is provided and contains 2+ groups, the median-survival row
also carries a log-rank p-value.

## Customisation

| Parameter | Effect |
|---|---|
| `times=[6, 12, 24]` | Survival probability at fixed time points |
| `times_label='mo'` | Unit label appended to each S(t) header |
| `conf_level=0.95` | CI level for the median |
| `digits=2`, `pct_digits=1` | Display rounding |
| `labels={'A': 'Arm A — placebo'}` | Per-group display labels |
| `show_logrank=False` | Skip the log-rank computation |
