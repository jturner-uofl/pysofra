# The `SofraTable` object

A `SofraTable` is a backend-agnostic representation of a publication-ready
statistical table. Every builder (`tbl_one`, `tbl_summary`, `tbl_regression`)
produces one; every renderer (HTML, Markdown, DOCX, LaTeX, PPTX) consumes one.

## Immutable & chainable

`SofraTable` is a frozen dataclass. Every modifier method returns a *new*
instance. The original is untouched.

```python
base = ps.tbl_one(df, by='arm')
with_p = base.add_p()       # new SofraTable
with_p_smd = with_p.add_smd()
# `base` is unchanged
```

This makes pipelines deterministic and notebook-friendly.

## Anatomy

```python
@dataclass(frozen=True)
class SofraTable:
    rows: tuple[Row, ...]
    headers: tuple[HeaderRow, ...]
    spanning_headers: tuple[SpanningHeader, ...]
    caption: str | None
    footnotes: tuple[str, ...]
    theme_name: str
    metadata: dict
    _spec: TableSpec | None        # carries builder spec for recomputation
    _rebuild: Callable | None      # rebuilds the table under a new spec
```

## Recomputation modifiers

Some modifiers — `.add_p()`, `.add_q()`, `.add_smd()`, `.add_overall()` —
need to recompute statistics from the original data. The builder
attaches a `_rebuild` closure that closes over the source dataframe;
`_with_option` calls it with an updated spec and returns a fresh
`SofraTable`.

## Presentational modifiers

Modifiers that don't need recomputation (`.theme()`, `.set_caption()`,
`.bold_p()`, `.bold_if()`, `.highlight_if()`) operate on the rendered
rows directly. They work on any `SofraTable` regardless of how it was
constructed.

## `to_dict()` — structural snapshot

```python
d = table.to_dict()
d.keys()
# {'caption', 'footnotes', 'theme', 'headers', 'spanning_headers', 'rows'}
```

Cheap and stable across versions — used for snapshot testing.
