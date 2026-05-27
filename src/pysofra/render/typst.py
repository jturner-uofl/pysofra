"""Typst rendering.

Typst (`https://typst.app/`) is a modern document-preparation system
positioned as a faster, simpler-syntax alternative to LaTeX. Its
``#table(...)`` element supports the same column counts, header
emphasis, and cell alignment that PySofra needs, with a friendlier
syntax.

We emit a ``#table(...)`` block prefixed by the caption (as a
``#figure`` wrapper when one is set) and followed by the footnotes
as italicised paragraphs. The output is a Typst-source string ready
to ``#include`` in a ``.typ`` document or compile directly via
``typst compile``.

The renderer is **lossy** in two respects:
* Inline plots (matplotlib PNGs attached via ``with_forest_plot`` /
  ``with_km_plot``) are *not* embedded — Typst's image() needs an
  on-disk path; users can embed manually via ``#image("plot.png")``.
* CellPart styling (bold, italic, super/subscript) inside cells is
  preserved as Typst markup (``*bold*``, ``_italic_``, etc.) but
  colour is dropped (Typst's per-cell colouring requires an
  alternate API).

This is the first stats-reporting package in either Python or R to
ship a Typst backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..core.schema import Cell, HeaderCell
from .base import Renderer

if TYPE_CHECKING:  # pragma: no cover
    from ..core.table import SofraTable


# Characters that must be escaped in Typst markup. Typst's escape
# character is backslash; the set is small.
_TYPST_ESCAPE = {
    "\\": "\\\\",
    "$": "\\$",
    "*": "\\*",
    "_": "\\_",
    "`": "\\`",
    "#": "\\#",
    "<": "\\<",
    ">": "\\>",
    "[": "\\[",
    "]": "\\]",
    "@": "\\@",
    "=": "\\=",
}


def _esc(text: str) -> str:
    """Escape a string for Typst markup context."""
    return "".join(_TYPST_ESCAPE.get(c, c) for c in str(text))


def _esc_cell(text: str) -> str:
    """Escape *and* preserve simple newlines.

    Typst cell content is markup; explicit line breaks need ``\\``
    at the end of the line.
    """
    if "\n" in text:
        parts = [_esc(p) for p in text.split("\n")]
        return " \\ ".join(parts)
    return _esc(text)


def _align_letter(cell: HeaderCell | Cell) -> str:
    a = (cell.align or "left").lower()
    if a == "right":
        return "right"
    if a == "center":
        return "center"
    return "left"


def _cell_inner(cell: Cell | HeaderCell, *, bold: bool = False) -> str:
    txt = _esc_cell(cell.text)
    if bold or getattr(cell, "bold", False):
        txt = f"*{txt}*"
    if getattr(cell, "italic", False):
        txt = f"_{txt}_"
    return f"[{txt}]"


@dataclass
class TypstRenderer(Renderer[str]):
    """Render a SofraTable as a Typst ``#table(...)`` block."""

    def render(self, table: SofraTable) -> str:
        ncols = _ncols(table)
        lines: list[str] = []

        # Caption + footnotes are wrapped in a #figure when present.
        has_caption = bool(table.caption)
        if has_caption:
            lines.append("#figure(")
            lines.append("  caption: [" + _esc(table.caption) + "],")
            lines.append("  kind: table,")
            lines.append("  [")  # body open

        # Per-column alignment from the last header row (or default left).
        if table.headers:
            align_row = table.headers[-1]
            aligns = [_align_letter(c) for c in align_row.cells]
            while len(aligns) < ncols:
                aligns.append("left")
        else:
            aligns = ["left"] * ncols

        lines.append("#table(")
        lines.append(f"  columns: {ncols},")
        lines.append(
            "  align: ("
            + ", ".join(aligns[:ncols])
            + ",),"
        )

        # Spanning headers come first (one cell with colspan = N).
        for sh in (table.spanning_headers or ()):
            colspan = max(int(getattr(sh, "cspan", ncols)), 1)
            label = _esc(sh.label)
            lines.append(
                f"  table.cell(colspan: {colspan}, [*{label}*]),"
            )

        # Column-header rows
        for hr in (table.headers or ()):
            cells = list(hr.cells)
            while len(cells) < ncols:
                cells.append(HeaderCell(text=""))
            lines.append(
                "  table.header("
                + ", ".join(_cell_inner(c, bold=True) for c in cells[:ncols])
                + "),"
            )

        # Body rows
        for r in table.rows:
            cells = list(r.cells)
            while len(cells) < ncols:
                cells.append(Cell(text=""))
            lines.append(
                "  "
                + ", ".join(_cell_inner(c) for c in cells[:ncols])
                + ","
            )

        lines.append(")")

        # Footnotes
        for fn in (table.footnotes or ()):
            lines.append("")
            lines.append("_" + _esc(fn) + "_")

        if has_caption:
            lines.append("  ]")  # body close
            lines.append(")")

        return "\n".join(lines) + "\n"

    def write(self, table: SofraTable, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(table), encoding="utf-8")
        return path


def _ncols(table: SofraTable) -> int:
    """Best-effort column count: use the widest header row, then rows."""
    n = 0
    for hr in (table.headers or ()):
        n = max(n, len(hr.cells))
    for r in table.rows:
        n = max(n, len(r.cells))
    return max(n, 1)
