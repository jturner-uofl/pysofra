# Composition — `tbl_merge()` & `tbl_stack()`

## Side-by-side (`tbl_merge`)

Two tables glued horizontally, sharing the first label column when
identical:

```python
t_female = ps.tbl_one(df[df.sex == 'F'], by='arm').add_p()
t_male   = ps.tbl_one(df[df.sex == 'M'], by='arm').add_p()

ps.tbl_merge([t_female, t_male], tab_spanners=['Female', 'Male'])
```

Requirements:

- All inputs must have the same number of body rows.
- If the first column of every row is identical across tables, it's
  collapsed to a single label column (`share_first_column=True`,
  default).

## Vertical (`tbl_stack`)

Multiple tables concatenated under one header:

```python
ps.tbl_stack(
    [t_cohort_a, t_cohort_b],
    group_labels=['Cohort A', 'Cohort B'],
)
```

Requirements:

- All inputs must have the same column count and header structure.
- Optional `group_labels` insert a group-header row between blocks.
