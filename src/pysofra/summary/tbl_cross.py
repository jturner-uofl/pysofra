"""Cross-tabulation tables — equivalent to ``gtsummary::tbl_cross``.

``tbl_cross`` builds a two-way contingency table with selectable cell
content:

* ``n``         — raw count (default)
* ``row_pct``   — row-percent
* ``col_pct``   — column-percent
* ``total_pct`` — overall percent
* ``n_row_pct`` — n with row-% in parens (the "n (row %)" style)
* ``n_col_pct`` — n with col-% in parens
* ``n_total_pct`` — n with total-% in parens

Margins (row totals / column totals / grand total) are added by default
and can be turned off with ``margins=False``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..core.format import fmt_n_pct, fmt_p_value
from ..core.frames import to_pandas
from ..core.schema import HeaderCell, HeaderRow, Row, make_cell
from ..core.table import SofraTable, TableSpec

_CELL_STYLES = (
    "n", "row_pct", "col_pct", "total_pct",
    "n_row_pct", "n_col_pct", "n_total_pct",
)


def tbl_cross(
    data: Any,
    *,
    row: str,
    column: str,
    cell: str = "n_col_pct",
    margins: bool = True,
    digits: int = 1,
    labels: dict[str, str] | None = None,
) -> SofraTable:
    """Cross-tabulate ``row`` against ``column``.

    Parameters
    ----------
    data
        Source dataframe.
    row
        Variable name for the rows.
    column
        Variable name for the columns.
    cell
        How to display each interior cell. See module docstring.
    margins
        Include row / column / grand totals.
    digits
        Decimal places for the percent.
    labels
        Optional mapping of level → display label, applied to both row
        and column labels.

    Notes
    -----
    The returned :class:`SofraTable` carries a rebuild closure over the
    source ``data`` so the statistical modifiers ``.add_p()`` and
    ``.add_overall()`` work directly:

    * ``.add_p()`` re-runs the cross-tab and appends a *p*-value
      footnote based on the auto-selected categorical test (Fisher's
      exact for 2x2, Pearson χ² otherwise).
    * ``.add_overall()`` toggles ``margins=True`` so the row, column,
      and grand totals are rendered (no-op when margins are already on,
      which is the default).
    * ``.add_smd()`` raises :class:`NotImplementedError` — SMD is a
      between-group effect-size on a single distribution and is
      undefined on a contingency table. Use :func:`tbl_one` for SMD
      between two arms.
    """
    if cell not in _CELL_STYLES:
        raise ValueError(
            f"cell must be one of {_CELL_STYLES}; got {cell!r}."
        )
    df = to_pandas(data)
    if row not in df.columns:
        raise KeyError(f"row column {row!r} not in data")
    if column not in df.columns:
        raise KeyError(f"column column {column!r} not in data")

    spec = TableSpec(
        builder="tbl_cross",
        options={
            "row": row,
            "column": column,
            "cell": cell,
            "margins": margins,
            "digits": digits,
            "labels": dict(labels or {}),
            # Modifier flags — toggled by .add_p() / .add_overall() /
            # .add_smd() via SofraTable._with_option().
            "p_value": False,
            "overall": False,
            "smd": False,
        },
    )
    return _build_cross(df, spec)


def _build_cross(df: pd.DataFrame, spec: TableSpec) -> SofraTable:
    """Build a tbl_cross SofraTable from a frozen source df and spec."""
    row: str = spec.options["row"]
    column: str = spec.options["column"]
    cell: str = spec.options["cell"]
    # add_overall() forces margins on; the user-passed margins= remains
    # otherwise.
    margins: bool = bool(spec.options["margins"] or spec.options.get("overall"))
    digits: int = int(spec.options["digits"])
    labels: dict[str, str] = dict(spec.options.get("labels") or {})

    # add_smd() is meaningless on a cross-tab. Surface that explicitly
    # rather than emitting a spurious "SMD" column or silently dropping
    # the flag.
    if spec.options.get("smd"):
        raise NotImplementedError(
            "add_smd() is not defined for tbl_cross — SMD measures the "
            "standardised difference between two distributions of a "
            "single variable. For SMD between two arms on the same "
            "variable, use tbl_one(df, by=...).add_smd().",
        )

    # Drop rows missing either dimension.
    sub = df[[row, column]].dropna()
    if sub.empty:
        # Emit a minimal placeholder table. Carry the spec + rebuild so
        # the table can still respond to modifiers (no-ops in practice,
        # but the contract is "rebuild always works").
        def _empty_rebuild(new_spec: TableSpec) -> SofraTable:  # pragma: no cover — modifier on empty cross-tab
            return _build_cross(df, new_spec)
        return SofraTable(
            rows=(Row(cells=(make_cell(row, align="left", bold=True),
                             make_cell("—", value=None,
                                       kind="numeric", align="right"))),),
            headers=(HeaderRow(cells=(
                HeaderCell(text=labels.get(row, row), align="left"),
                HeaderCell(text=labels.get(column, column)),
            )),),
            footnotes=("No non-missing rows for the requested cross-tabulation.",),
            metadata={"builder": "tbl_cross"},
            _spec=spec,
            _rebuild=_empty_rebuild,
        )

    # Preserve categorical ordering where present.
    if isinstance(df[row].dtype, pd.CategoricalDtype):
        row_levels = [lvl for lvl in df[row].cat.categories if lvl in set(sub[row])]
    else:
        row_levels = sorted(sub[row].unique(), key=_sort_key)
    if isinstance(df[column].dtype, pd.CategoricalDtype):
        col_levels = [lvl for lvl in df[column].cat.categories if lvl in set(sub[column])]
    else:
        col_levels = sorted(sub[column].unique(), key=_sort_key)

    ctab = pd.crosstab(sub[row], sub[column])
    ctab = ctab.reindex(index=row_levels, columns=col_levels, fill_value=0)

    row_totals = ctab.sum(axis=1)
    col_totals = ctab.sum(axis=0)
    grand_total = float(ctab.values.sum())

    # ------------------------------------------------------------------
    # Headers
    # ------------------------------------------------------------------
    header_cells = [HeaderCell(text=labels.get(row, row), align="left")]
    for lvl in col_levels:
        header_cells.append(HeaderCell(text=labels.get(lvl, str(lvl))))
    if margins:
        header_cells.append(HeaderCell(text="Total"))
    headers = (HeaderRow(cells=tuple(header_cells)),)

    # spanning header naming the column variable
    from ..core.schema import SpanningHeader
    spanning: tuple[SpanningHeader, ...]
    if len(col_levels) > 0:
        spanning = (SpanningHeader(
            label=labels.get(column, column),
            start=1,
            end=len(col_levels) + (1 if margins else 0),
        ),)
    else:  # pragma: no cover — guarded by the empty-sub short-circuit above
        spanning = ()

    # ------------------------------------------------------------------
    # Body rows
    # ------------------------------------------------------------------
    rows: list[Row] = []
    for r_lvl in row_levels:
        body = [make_cell(labels.get(r_lvl, str(r_lvl)), align="left")]
        for c_lvl in col_levels:
            n = int(ctab.loc[r_lvl, c_lvl])
            body.append(_fmt_cross_cell(
                n=n,
                row_total=int(row_totals.loc[r_lvl]),
                col_total=int(col_totals.loc[c_lvl]),
                grand_total=grand_total,
                style=cell,
                digits=digits,
            ))
        if margins:
            rt = int(row_totals.loc[r_lvl])
            body.append(make_cell(
                _fmt_margin(rt, grand_total, style=cell, digits=digits),
                value=rt, kind="numeric", align="right",
            ))
        rows.append(Row(cells=tuple(body)))

    # Margin row
    if margins:
        body = [make_cell("Total", align="left", bold=True)]
        for c_lvl in col_levels:
            ct = int(col_totals.loc[c_lvl])
            body.append(make_cell(
                _fmt_margin(ct, grand_total, style=cell, digits=digits),
                value=ct, kind="numeric", align="right",
            ))
        body.append(make_cell(
            f"{int(grand_total):,}",
            value=int(grand_total),
            kind="numeric", align="right", bold=True,
        ))
        rows.append(Row(cells=tuple(body), is_group_header=True))

    footnotes = [_footnote_for(cell)]

    # ------------------------------------------------------------------
    # add_p() — auto-selected categorical test on the full contingency.
    # Reported as a footnote so the table body stays a clean grid.
    # ------------------------------------------------------------------
    metadata: dict[str, Any] = {"builder": "tbl_cross"}
    if spec.options.get("p_value"):
        from .tests import categorical_test
        res = categorical_test(sub[row], sub[column])
        if res.p_value is not None:
            footnotes.append(
                f"{res.test}: p = {fmt_p_value(res.p_value)}",
            )
            # Surface the raw p-value + test name in metadata for
            # programmatic consumers (e.g. golden tests, downstream
            # reports that want the numeric value).
            metadata["p_value"] = float(res.p_value)
            metadata["p_test"] = res.test

    def _rebuild(new_spec: TableSpec) -> SofraTable:
        return _build_cross(df, new_spec)

    return SofraTable(
        rows=tuple(rows),
        headers=headers,
        spanning_headers=spanning,
        footnotes=tuple(footnotes),
        metadata=metadata,
        _spec=spec,
        _rebuild=_rebuild,
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _fmt_cross_cell(
    *, n: int, row_total: int, col_total: int, grand_total: float,
    style: str, digits: int,
) -> Any:
    """Format one body cell of the cross-tab according to ``style``."""
    if style == "n":
        return make_cell(f"{n:,}", value=n, kind="numeric", align="right")
    if style == "row_pct":
        pct = 100.0 * n / row_total if row_total else float("nan")
        return make_cell(_pct(pct, digits), value=pct,
                         kind="numeric", align="right")
    if style == "col_pct":
        pct = 100.0 * n / col_total if col_total else float("nan")
        return make_cell(_pct(pct, digits), value=pct,
                         kind="numeric", align="right")
    if style == "total_pct":
        pct = 100.0 * n / grand_total if grand_total else float("nan")
        return make_cell(_pct(pct, digits), value=pct,
                         kind="numeric", align="right")
    if style == "n_row_pct":
        return make_cell(fmt_n_pct(n, row_total, digits=digits),
                         value=n, kind="numeric", align="right")
    if style == "n_col_pct":
        return make_cell(fmt_n_pct(n, col_total, digits=digits),
                         value=n, kind="numeric", align="right")
    if style == "n_total_pct":
        return make_cell(fmt_n_pct(n, int(grand_total), digits=digits),
                         value=n, kind="numeric", align="right")
    raise ValueError(f"unknown cell style {style!r}")  # pragma: no cover — guarded by top-level cell-style validation


def _fmt_margin(n: int, grand_total: float, *, style: str, digits: int) -> str:
    """Margin cell formatting — always 'n (overall %)' for n-style cells."""
    if style.startswith("n_"):
        return fmt_n_pct(n, int(grand_total), digits=digits)
    if style in ("row_pct", "col_pct", "total_pct"):
        return _pct(100.0 * n / grand_total if grand_total else float("nan"),
                    digits)
    return f"{n:,}"


def _pct(p: float, digits: int) -> str:
    import math
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "—"
    return f"{p:.{digits}f}%"


def _footnote_for(style: str) -> str:
    return {
        "n":            "Cells: raw counts.",
        "row_pct":      "Cells: row-percent.",
        "col_pct":      "Cells: column-percent.",
        "total_pct":    "Cells: overall percent.",
        "n_row_pct":    "Cells: n (row-%).",
        "n_col_pct":    "Cells: n (column-%).",
        "n_total_pct":  "Cells: n (overall-%).",
    }[style]


def _sort_key(x: Any) -> tuple[int, Any]:
    if isinstance(x, bool):
        return (0, int(x))
    if isinstance(x, (int, float)):
        return (0, float(x))
    if isinstance(x, str):
        return (1, x)
    return (2, repr(x))
