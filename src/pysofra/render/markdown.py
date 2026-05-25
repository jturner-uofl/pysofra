"""Markdown rendering.

Outputs GitHub-flavored Markdown. The format is intentionally lossy:
indentation and styling do not survive across every Markdown flavour,
but the table structure, captions, footnotes, and spanning-header
labels are preserved.

Spanning headers are emitted as a **bold paragraph above the table**
(not as a pipe row) because Markdown's table grammar requires the
column-header row to be immediately followed by the alignment row;
inserting a row between them turns the spanner into the data header
and silently corrupts the table. Rendering spanners as a paragraph
keeps both the table valid and the information visible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..core.schema import Cell, HeaderCell, Row, SpanningHeader
from ..core.table import SofraTable
from .base import Renderer

_INDENT_CHAR = "    "  # 4 regular spaces — Markdown collapses these visually


@dataclass
class MarkdownRenderer(Renderer[str]):
    def render(self, table: SofraTable) -> str:
        ncols = _ncols(table)
        lines: list[str] = []

        if table.caption:
            lines.append(f"**{_escape(table.caption)}**")
            lines.append("")

        if table.spanning_headers:
            lines.append(_render_spanning_paragraph(table.spanning_headers))
            lines.append("")

        # Use the *last* header row as the column header; earlier rows become
        # part of the caption block (Markdown supports only one).
        header_row = table.headers[-1] if table.headers else None
        if header_row is None:
            header_cells = [HeaderCell(text="") for _ in range(ncols)]
        else:
            header_cells = list(header_row.cells)

        lines.append("| " + " | ".join(_header_text(c) for c in header_cells) + " |")
        lines.append("|" + "|".join(_align_marker(c) for c in header_cells) + "|")

        for r in table.rows:
            lines.append(_render_row(r))

        if table.footnotes:
            lines.append("")
            for f in table.footnotes:
                lines.append(f"_{_escape(f)}_")

        return "\n".join(lines).rstrip() + "\n"


def _render_spanning_paragraph(spans: tuple[SpanningHeader, ...]) -> str:
    """Render spanners as a single italicised paragraph above the table.

    Example: ``*Group 1 (cols 1–3) · Group 2 (cols 4–5)*``.
    """
    parts: list[str] = []
    for s in spans:
        range_str = (
            f"col {s.start + 1}" if s.start == s.end
            else f"cols {s.start + 1}–{s.end + 1}"
        )
        parts.append(f"**{_escape(s.label)}** ({range_str})")
    return "*" + " · ".join(parts) + "*"


def _header_text(c: HeaderCell) -> str:
    return _escape(c.text.replace("\n", " · "))


def _align_marker(c: HeaderCell) -> str:
    if c.align == "right":
        return " ---: "
    if c.align == "center":
        return " :---: "
    return " :--- "


def _render_row(r: Row) -> str:
    return "| " + " | ".join(_cell_text(c) for c in r.cells) + " |"


def _cell_text(c: Cell) -> str:
    indent = _INDENT_CHAR * c.indent
    text = _escape(c.text)
    if c.bold:
        text = f"**{text}**"
    if c.italic:
        text = f"*{text}*"
    return f"{indent}{text}"


# Characters that have syntactic meaning in GitHub-flavored Markdown and
# need to be backslash-escaped inside table cells. The backslash itself
# is escaped FIRST so subsequent additions don't double-escape.
_MARKDOWN_SPECIALS = ("\\", "|", "`", "*", "_", "[", "]", "<", ">", "#")
_SPECIAL_RE = re.compile("|".join(re.escape(ch) for ch in _MARKDOWN_SPECIALS))


def _escape(s: str) -> str:
    """Backslash-escape every character GitHub-flavored Markdown treats
    as syntax inside table cells.

    Without this, a cell text like ``gene*`` is rendered as italicised
    ``gene``, ``a_b_c`` becomes underlined, and a stray ``[`` opens an
    unmatched link.
    """
    return _SPECIAL_RE.sub(lambda m: "\\" + m.group(0), s)


def _ncols(table: SofraTable) -> int:
    if table.headers:
        return len(table.headers[0].cells)
    if table.rows:
        return len(table.rows[0].cells)
    return 1
