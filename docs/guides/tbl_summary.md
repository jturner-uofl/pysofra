# Summary — `tbl_summary()`

`tbl_summary()` is the general-purpose descriptive summary builder.
It shares the engine of [`tbl_one`](tbl_one.md); the two functions
exist separately because the *intent* differs:

- `tbl_one` is for stratified baseline characteristics (a "Table 1").
- `tbl_summary` is for any descriptive summary — typically unstratified.

```python
ps.tbl_summary(
    df,
    variables=['age', 'bmi', 'sex', 'race'],
    labels={'age': 'Age (years)', 'bmi': 'BMI (kg/m²)'},
)
```

All `tbl_one` modifiers (`.add_p()`, `.add_q()`, `.add_smd()`,
`.add_overall()`, `.theme()`, ...) work on `tbl_summary` output.
