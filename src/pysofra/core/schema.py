"""Internal schema for SofraTable.

This module defines the backend-agnostic representation of a statistical
table. Every renderer (HTML, Markdown, DOCX, PPTX, LaTeX) consumes the same
schema; statistical engines (Table 1, summary, regression) produce it.

The schema is intentionally simple and immutable. Cells carry both the
*display* string and the *raw* value when known, so renderers can choose
the appropriate format (e.g., right-alignment for numeric cells).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

CellKind = Literal[
    "text", "numeric", "p_value", "q_value", "ci", "header_label", "group_label"
]
Alignment = Literal["left", "center", "right"]


@dataclass(frozen=True)
class CellPart:
    """A typographically distinct run inside a single cell.

    Used by ``SofraTable.compose()`` to embed multi-format content
    (bold + italic + colour) inside one cell. Renderers honour
    ``CellPart.bold``, ``italic``, ``superscript``, ``subscript``,
    ``code``, ``color``, and ``link`` where the backend supports them;
    unsupported flags degrade to plain text.
    """

    text: str
    bold: bool = False
    italic: bool = False
    superscript: bool = False
    subscript: bool = False
    code: bool = False
    color: str | None = None
    link: str | None = None


@dataclass(frozen=True)
class Cell:
    """One cell of a SofraTable.

    `value` carries the raw underlying value when meaningful (e.g., a float
    p-value). `text` is the rendered string used by all backends. Renderers
    may consult `kind` and `value` for additional formatting decisions.

    `style` is an optional mapping of renderer-specific overrides:

    - ``html`` — extra inline-CSS declarations applied to the ``<td>``.
    - ``docx`` — keys like ``padding_pt``, ``shading_hex``, ``borders``.
    - ``xlsx`` — keys like ``bg_color``, ``num_format`` (forwarded to xlsxwriter).
    - ``latex`` — prepended raw LaTeX (e.g. ``\\rowcolor{...}``).

    All renderers ignore keys they don't understand.
    """

    text: str
    value: Any = None
    kind: CellKind = "text"
    align: Alignment | None = None
    bold: bool = False
    italic: bool = False
    indent: int = 0
    style: dict[str, Any] | None = None
    parts: tuple[CellPart, ...] | None = None


@dataclass(frozen=True)
class Row:
    """One row of the table body."""

    cells: tuple[Cell, ...]
    is_group_header: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HeaderCell:
    text: str
    align: Alignment = "center"
    bold: bool = True


@dataclass(frozen=True)
class HeaderRow:
    """A column-header row. Tables may have multiple stacked header rows."""

    cells: tuple[HeaderCell, ...]


@dataclass(frozen=True)
class SpanningHeader:
    """A spanning header above the column headers.

    `start` and `end` are 0-indexed column indices, inclusive.
    """

    label: str
    start: int
    end: int


def make_cell(
    text: str,
    value: Any = None,
    kind: CellKind = "text",
    align: Alignment | None = None,
    bold: bool = False,
    italic: bool = False,
    indent: int = 0,
    style: dict[str, Any] | None = None,
) -> Cell:
    """Convenience constructor used internally."""
    return Cell(
        text=text,
        value=value,
        kind=kind,
        align=align,
        bold=bold,
        italic=italic,
        indent=indent,
        style=style,
    )
