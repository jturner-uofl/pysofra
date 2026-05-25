"""Excel (.xlsx) rendering via ``xlsxwriter``.

Writes a single-sheet workbook containing one table. The caller never
touches the xlsx-writer API directly — pass a :class:`SofraTable` and a
path and the renderer takes care of formatting (fonts, borders, header
shading, row indentation, column widths, captions, footnotes).
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.schema import HeaderRow, Row, SpanningHeader
from ..core.table import SofraTable
from ..themes.registry import resolve_theme

# A fixed wall-clock timestamp used in the workbook's docProps metadata.
# xlsxwriter would otherwise embed ``datetime.now()`` into core.xml's
# ``<dcterms:created>``/``<dcterms:modified>`` elements, breaking
# byte-determinism across processes (the same SofraTable would produce
# a different SHA-256 on each call). 2000-01-01T00:00:00Z is the epoch
# we pin for reproducible publication artefacts.
_DETERMINISTIC_CREATED = _dt.datetime(2000, 1, 1, 0, 0, 0)

# Matches a plain decimal — optional sign, digits, optional fractional
# part. Used to detect when ``Cell.text`` is the literal formatted
# value (so writing the numeric ``Cell.value`` to Excel will match
# what HTML/Markdown/LaTeX display).
_PLAIN_NUMBER_RE = re.compile(r"^[-+]?\d+(\.\d+)?$")


def _text_matches_number(text: str) -> bool:
    """True when ``text`` is a bare decimal (no journal-style
    threshold marker like ``<0.001`` / ``>0.99`` / em-dash)."""
    return bool(_PLAIN_NUMBER_RE.match(text.strip()))


@dataclass
class XlsxRenderer:
    """Write a SofraTable to an .xlsx file."""

    sheet_name: str = "Table"

    def write(self, table: SofraTable, path: Path) -> Path:
        try:
            import xlsxwriter
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Excel export requires xlsxwriter. Install with "
                "`pip install xlsxwriter`."
            ) from e

        theme = resolve_theme(table.theme_name)
        d = theme.docx  # reuse the docx theme dict for font / sizing
        font_name: str = d.get("font_name", "Calibri")
        font_size: int = int(d.get("font_size", 11))

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        wb = xlsxwriter.Workbook(str(path))
        # Pin the workbook's creation timestamp so the bytes are
        # reproducible across processes (see _DETERMINISTIC_CREATED).
        wb.set_properties({"created": _DETERMINISTIC_CREATED})
        ws = wb.add_worksheet(self.sheet_name)

        base_fmt = {"font_name": font_name, "font_size": font_size}
        fmt_caption = wb.add_format({**base_fmt, "bold": True,
                                      "font_size": font_size + 1})
        fmt_header = wb.add_format({**base_fmt, "bold": True,
                                     "align": "center", "valign": "vcenter",
                                     "bottom": 2, "top": 2,
                                     "text_wrap": True})
        fmt_spanning = wb.add_format({**base_fmt, "bold": True,
                                       "align": "center", "bottom": 1})
        fmt_footnote = wb.add_format({**base_fmt, "italic": True,
                                       "font_size": max(8, font_size - 1)})

        row_idx = 0
        ncols = _ncols(table)

        if table.caption:
            ws.merge_range(row_idx, 0, row_idx, ncols - 1, table.caption,
                           fmt_caption)
            row_idx += 1

        if table.spanning_headers:
            row_idx = _write_spanning_row(
                ws, row_idx, table.spanning_headers, ncols, fmt_spanning,
            )

        for hr in table.headers:
            row_idx = _write_header_row(ws, row_idx, hr, fmt_header)

        last_body_row = row_idx + len(table.rows) - 1
        for r in table.rows:
            _write_body_row(ws, row_idx, r, wb, base_fmt,
                            is_last=(row_idx == last_body_row))
            row_idx += 1

        if table.footnotes:
            for fn in table.footnotes:
                ws.merge_range(row_idx, 0, row_idx, ncols - 1, fn, fmt_footnote)
                row_idx += 1

        # Sensible column widths based on cell-text length.
        _autosize_columns(ws, table, ncols)

        # xlsxwriter writes the actual .xlsx on ``close()``; a permission
        # failure (read-only directory) surfaces here as
        # ``xlsxwriter.exceptions.FileCreateError``. Every other PySofra
        # renderer raises an ``OSError`` subclass on the same failure
        # mode (``PermissionError`` / ``IsADirectoryError`` / etc.). To
        # let callers handle "couldn't write the file" with a single
        # ``except OSError`` regardless of backend, re-raise as
        # ``OSError`` (chaining the original).
        try:
            wb.close()
        except xlsxwriter.exceptions.FileCreateError as e:
            raise OSError(str(e)) from e
        # xlsxwriter also stamps ZIP entry mtimes with the current
        # wall-clock; rewrite with fixed entry mtimes for cross-process
        # byte-determinism.
        from ._zip_determinism import make_zip_deterministic
        make_zip_deterministic(path)
        return path

    def render(self, table: SofraTable) -> str:  # pragma: no cover
        raise NotImplementedError("XlsxRenderer writes to disk; use .write(table, path).")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _ncols(table: SofraTable) -> int:
    if table.headers:
        return len(table.headers[0].cells)
    if table.rows:
        return len(table.rows[0].cells)
    return 1


def _write_spanning_row(ws: Any, row_idx: int,
                        spans: tuple[SpanningHeader, ...],
                        ncols: int, fmt: Any) -> int:
    for span in spans:
        if span.end > span.start:
            ws.merge_range(row_idx, span.start, row_idx, span.end,
                           span.label, fmt)
        else:
            ws.write(row_idx, span.start, span.label, fmt)
    del ncols
    return row_idx + 1


def _write_header_row(ws: Any, row_idx: int, hr: HeaderRow, fmt: Any) -> int:
    ws.set_row(row_idx, 30)  # taller for wrapped headers
    for col_idx, c in enumerate(hr.cells):
        text = c.text.replace("\n", "\n")
        ws.write_string(row_idx, col_idx, text, fmt)
    return row_idx + 1


def _write_body_row(ws: Any, row_idx: int, r: Row, wb: Any,
                    base_fmt: dict[str, Any], *, is_last: bool) -> None:
    for col_idx, c in enumerate(r.cells):
        props: dict[str, Any] = dict(base_fmt)
        if c.align == "right":
            props["align"] = "right"
        elif c.align == "center":
            props["align"] = "center"
        else:
            props["align"] = "left"
        if c.bold or r.is_group_header:
            props["bold"] = True
        if c.italic:
            props["italic"] = True
        if c.indent > 0 and col_idx == 0:
            props["indent"] = c.indent
        if is_last:
            props["bottom"] = 2
        highlight = (r.metadata or {}).get("highlight")
        if highlight:
            props["bg_color"] = str(highlight)
        # Cell-level style['xlsx'] overrides are forwarded to xlsxwriter.
        cell_xlsx = (c.style or {}).get("xlsx") if c.style else None
        if isinstance(cell_xlsx, dict):
            props.update(cell_xlsx)
        fmt = wb.add_format(props)

        # Try to preserve numeric type when the cell carries a number,
        # but only when the rendered text is a plain decimal. When the
        # journal-style p-value threshold has fired (``"<0.001"``,
        # ``">0.99"``, em-dash for NA), the rendered text no longer
        # matches the float, and writing the float would make Excel
        # disagree with HTML / Markdown / LaTeX. Detect that case and
        # write the formatted string instead.
        if (
            isinstance(c.value, (int, float))
            and c.kind in ("numeric", "p_value", "q_value")
            and _text_matches_number(c.text)
        ):
            try:
                ws.write_number(row_idx, col_idx, float(c.value), fmt)
                continue
            except Exception:  # pragma: no cover — xlsxwriter accepts every float
                pass
        ws.write_string(row_idx, col_idx, c.text, fmt)


def _autosize_columns(ws: Any, table: SofraTable, ncols: int) -> None:
    widths = [10] * ncols
    for hr in table.headers:
        for j, hc in enumerate(hr.cells):
            for line in hc.text.split("\n"):
                widths[j] = max(widths[j], min(40, len(line) + 2))
    for r in table.rows:
        for j, bc in enumerate(r.cells[:ncols]):
            widths[j] = max(widths[j], min(40, len(bc.text) + 2 + bc.indent * 2))
    for j, w in enumerate(widths):
        ws.set_column(j, j, w)
