"""Table composition: :func:`tbl_merge` and :func:`tbl_stack`.

Both functions take a sequence of :class:`SofraTable` objects and combine
them into a single SofraTable. Merging glues tables side-by-side (sharing
the first / label column by default); stacking concatenates them vertically.
"""

from __future__ import annotations

from itertools import zip_longest

from .schema import Cell, HeaderCell, HeaderRow, Row, SpanningHeader
from .table import SofraTable


def tbl_merge(
    tables: list[SofraTable] | tuple[SofraTable, ...],
    *,
    tab_spanners: list[str] | None = None,
    share_first_column: bool = True,
) -> SofraTable:
    """Merge tables side-by-side.

    Parameters
    ----------
    tables
        Two or more :class:`SofraTable` objects with the same number of
        body rows.
    tab_spanners
        Optional list of spanning-header labels, one per input table.
    share_first_column
        When ``True`` (default) and every input has the same first column
        in every row, the duplicate label columns are dropped from the
        2nd-Nth tables.
    """
    tables = list(tables)
    if len(tables) < 2:
        raise ValueError("tbl_merge requires at least two tables.")
    n_rows_sets = {len(t.rows) for t in tables}
    if len(n_rows_sets) != 1:
        raise ValueError(
            f"tbl_merge requires all tables to have the same number of rows; got {n_rows_sets}."
        )

    drop_first = (
        share_first_column
        and all(t.rows for t in tables)
        and all(
            all(t.rows[i].cells[0].text == tables[0].rows[i].cells[0].text
                for i in range(len(t.rows)))
            for t in tables[1:]
        )
    )

    # Headers — pick the deepest header rows across inputs.
    max_header_depth = max(len(t.headers) for t in tables)
    merged_headers: list[HeaderRow] = []
    for level in range(max_header_depth):
        header_row_cells: list[HeaderCell] = []
        for i, t in enumerate(tables):
            hr = t.headers[level] if level < len(t.headers) else None
            if hr is None:
                # pad with empty header cells
                base = tables[0].headers[level] if level < len(tables[0].headers) else None
                width = len(t.headers[0].cells) if t.headers else 1
                hr = HeaderRow(cells=tuple(HeaderCell(text="") for _ in range(width)))
                del base
            row_cells: list[HeaderCell] = list(hr.cells)
            if drop_first and i > 0 and row_cells:
                row_cells = row_cells[1:]
            header_row_cells.extend(row_cells)
        merged_headers.append(HeaderRow(cells=tuple(header_row_cells)))

    # Spanning headers from tab_spanners (if provided).
    spanning: list[SpanningHeader] = []
    if tab_spanners:
        if len(tab_spanners) != len(tables):
            raise ValueError("tab_spanners must have one entry per table.")
        col = 0
        for i, (t, label) in enumerate(zip(tables, tab_spanners, strict=True)):
            width = len(t.headers[0].cells) if t.headers else len(t.rows[0].cells)
            if drop_first and i > 0:
                width -= 1
            spanning.append(SpanningHeader(label=label, start=col, end=col + width - 1))
            col += width

    # Body rows — concatenate cells horizontally.
    n_rows = next(iter(n_rows_sets))
    merged_rows: list[Row] = []
    for i in range(n_rows):
        body_cells: list[Cell] = []
        for j, t in enumerate(tables):
            body_row_cells: list[Cell] = list(t.rows[i].cells)
            if drop_first and j > 0 and body_row_cells:
                body_row_cells = body_row_cells[1:]
            body_cells.extend(body_row_cells)
        merged_rows.append(Row(cells=tuple(body_cells),
                               is_group_header=tables[0].rows[i].is_group_header))

    footnotes: list[str] = []
    for t in tables:
        for f in t.footnotes:
            if f not in footnotes:
                footnotes.append(f)

    return SofraTable(
        rows=tuple(merged_rows),
        headers=tuple(merged_headers),
        spanning_headers=tuple(spanning),
        footnotes=tuple(footnotes),
        caption=tables[0].caption,
        theme_name=tables[0].theme_name,
        metadata={"merged_from": [t.metadata.get("builder", "?") for t in tables]},
    )


def tbl_stack(
    tables: list[SofraTable] | tuple[SofraTable, ...],
    *,
    group_labels: list[str] | None = None,
) -> SofraTable:
    """Stack tables vertically.

    All inputs must share the same column count and header structure.
    Optional ``group_labels`` introduce a group-header row between blocks.
    """
    tables = list(tables)
    if len(tables) < 2:
        raise ValueError("tbl_stack requires at least two tables.")

    ncols = len(tables[0].headers[0].cells) if tables[0].headers else len(tables[0].rows[0].cells)
    for t in tables[1:]:
        nc = len(t.headers[0].cells) if t.headers else len(t.rows[0].cells)
        if nc != ncols:
            raise ValueError(
                f"tbl_stack requires equal column counts; got {ncols} and {nc}."
            )
    if group_labels is not None and len(group_labels) != len(tables):
        raise ValueError("group_labels must have one entry per table.")

    rows: list[Row] = []
    for i, t in enumerate(tables):
        if group_labels:
            header_cells = [Cell(text=group_labels[i], bold=True, align="left")]
            header_cells.extend(Cell(text="") for _ in range(ncols - 1))
            rows.append(Row(cells=tuple(header_cells), is_group_header=True))
        rows.extend(t.rows)

    footnotes: list[str] = []
    for t in tables:
        for f in t.footnotes:
            if f not in footnotes:
                footnotes.append(f)

    return SofraTable(
        rows=tuple(rows),
        headers=tables[0].headers,
        spanning_headers=tables[0].spanning_headers,
        footnotes=tuple(footnotes),
        caption=tables[0].caption,
        theme_name=tables[0].theme_name,
        metadata={"stacked_from": [t.metadata.get("builder", "?") for t in tables]},
    )


# Silence unused
_ = zip_longest
