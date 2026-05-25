"""Render a SofraTable to a PNG image.

Uses matplotlib's ``ax.table`` to draw the structural representation of
the table on a Figure, then saves to PNG. Captions are rendered above
the table; footnotes below. Spanning headers are drawn as a separate
row at the top.

This renderer is intentionally simple — for publication-quality output
the LaTeX / DOCX / HTML backends produce better results. PNG export
exists for screenshot-style previews and for embedding inside a notebook
when an SVG isn't desired.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.schema import HeaderRow, Row, SpanningHeader
from ..core.table import SofraTable


def write_image(
    table: SofraTable,
    path: Path,
    *,
    scale: float = 2.0,
    dpi: int = 300,
) -> Path:
    try:
        from ..plot._backend import use_headless_backend
        use_headless_backend()
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "to_image() requires matplotlib. Install with `pip install matplotlib`."
        ) from e

    ncols = _ncols(table)
    n_header_rows = (1 if table.spanning_headers else 0) + len(table.headers)
    n_body_rows = len(table.rows)
    total_rows = n_header_rows + n_body_rows
    # Approximate sizing: 1.2 in per column, 0.35 in per row, plus header/footer.
    width = max(4.0, 1.2 * ncols * scale)
    height = max(2.0, 0.35 * total_rows * scale + 1.2)

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    ax.axis("off")

    # Caption above.
    if table.caption:
        ax.set_title(table.caption, loc="left", fontweight="bold", fontsize=11)

    grid: list[list[str]] = []
    spans: list[tuple[int, int, int, str]] = []  # (row, start, end, label)
    row_idx = 0
    if table.spanning_headers:
        # Reserve a row for spans.
        grid.append([""] * ncols)
        for s in table.spanning_headers:
            spans.append((row_idx, s.start, s.end, s.label))
        row_idx += 1

    for hr in table.headers:
        grid.append([c.text.replace("\n", " · ") for c in hr.cells])
        row_idx += 1
    for r in table.rows:
        prefix = "    " * (r.cells[0].indent if r.cells else 0)
        grid.append(
            [prefix + c.text if j == 0 else c.text
             for j, c in enumerate(r.cells)]
        )
        row_idx += 1

    if not grid:
        grid = [[""] * ncols]
    tbl = ax.table(
        cellText=grid,
        cellLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.4)

    # Bold header rows.
    for j in range(ncols):
        cell = tbl[(0 if table.spanning_headers else 0, j)]
        cell.set_text_props(weight="bold")
        for i_h in range(n_header_rows):
            tbl[(i_h, j)].set_text_props(weight="bold")
            tbl[(i_h, j)].set_facecolor("#f2f2f2")

    # Apply span labels as merged-looking cells (matplotlib doesn't really
    # merge; we just write the label into the leftmost cell of the span
    # and blank out the others, but the visual cue is enough for previews).
    for row_i, start, end, label in spans:
        tbl[(row_i, start)].get_text().set_text(label)
        tbl[(row_i, start)].set_text_props(weight="bold",
                                            horizontalalignment="center")
        for k in range(start + 1, end + 1):
            tbl[(row_i, k)].get_text().set_text("")
            tbl[(row_i, k)].set_facecolor("#f2f2f2")

    # Footnotes below.
    if table.footnotes:
        fn_text = "\n".join(table.footnotes)
        ax.text(
            0, -0.05, fn_text,
            transform=ax.transAxes,
            fontsize=8, style="italic",
            verticalalignment="top",
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return path


def _ncols(table: SofraTable) -> int:
    if table.headers:
        return len(table.headers[0].cells)
    if table.rows:
        return len(table.rows[0].cells)
    return 1


_ = (Any, HeaderRow, Row, SpanningHeader)  # silence unused-import linter
