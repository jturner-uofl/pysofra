"""LaTeX rendering — booktabs style.

Outputs a self-contained ``table`` float with ``\\caption``, ``\\toprule``,
``\\midrule``, ``\\bottomrule``, and an ``\\addlinespace``-style spanning
header. The output is publication-ready and compiles with any modern
``pdflatex`` / ``lualatex`` / ``xelatex`` engine, given a ``\\usepackage{booktabs}``
in the document preamble (also ``\\usepackage{array}`` for the alignment
column types we use).

The renderer is deliberately minimal: we do not try to replicate every
HTML styling decision in LaTeX. Captions, footnotes, spanning headers,
column alignment, indentation, and bold/italic cells are all preserved;
fonts and colors come from the surrounding LaTeX document, not from our
themes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.schema import Cell, HeaderCell, Row, SpanningHeader
from ..core.table import SofraTable
from .base import Renderer

# LaTeX special characters that must be escaped in text mode.
_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    # ``<`` and ``>`` render as ¡ / ¿ under the default OT1 font
    # encoding (only correct under T1); escape to the encoding-robust
    # text commands so category labels like "<HS" / ">=College"
    # typeset correctly regardless of the document's fontenc setup.
    "<": r"\textless{}",
    ">": r"\textgreater{}",
}


def _escape(text: str) -> str:
    """Escape LaTeX special characters in plain text."""
    out: list[str] = []
    for ch in text:
        out.append(_LATEX_ESCAPES.get(ch, ch))
    return "".join(out)


def _align_char(c: HeaderCell | Cell) -> str:
    if getattr(c, "align", None) == "right":
        return "r"
    if getattr(c, "align", None) == "center":
        return "c"
    return "l"


@dataclass
class LatexRenderer(Renderer[str]):
    """Render a :class:`SofraTable` as a LaTeX ``table`` float."""

    booktabs: bool = True
    float_position: str = "ht"
    centering: bool = True
    image_basename: str | None = None  # if set, the inline plot is written here.pdf

    def render(self, table: SofraTable) -> str:
        ncols = _ncols(table)
        col_spec = _column_spec(table)

        toprule = r"\toprule" if self.booktabs else r"\hline"
        midrule = r"\midrule" if self.booktabs else r"\hline"
        bottomrule = r"\bottomrule" if self.booktabs else r"\hline"

        plot = getattr(table, "inline_plot", None)
        plot_command: str | None = None
        if (
            plot is not None
            and getattr(plot, "pdf_bytes", None)
            and self.image_basename
        ):
            plot_command = (
                rf"\includegraphics[width={plot.width_in:.2f}in]"
                rf"{{{self.image_basename}}}"
            )

        out: list[str] = []
        out.append(rf"\begin{{table}}[{self.float_position}]")
        if self.centering:
            out.append(r"\centering")
        if table.caption:
            out.append(rf"\caption{{{_escape(table.caption)}}}")

        if plot_command and table.inline_svg_position == "above":
            out.append(plot_command)
            out.append(r"\par\vspace{0.5em}")

        out.append(rf"\begin{{tabular}}{{{col_spec}}}")
        out.append(toprule)

        # Spanning headers
        for span_row in self._render_spanning_rows(table.spanning_headers, ncols):
            out.append(span_row)

        # Column headers
        for hr in table.headers:
            out.append(self._render_header_row(hr))
        if table.headers:
            out.append(midrule)

        # Body rows
        for r in table.rows:
            out.append(self._render_row(r))

        out.append(bottomrule)
        out.append(r"\end{tabular}")

        if plot_command and table.inline_svg_position == "below":
            out.append(r"\par\vspace{0.5em}")
            out.append(plot_command)

        # Footnotes — emitted *outside* the tabular as small italic paragraphs.
        if table.footnotes:
            out.append(r"\vspace{0.25em}")
            for fn in table.footnotes:
                out.append(
                    rf"\par\noindent\small\textit{{{_escape(fn)}}}"
                )

        out.append(r"\end{table}")
        return "\n".join(out) + "\n"

    def write(self, table: SofraTable, path: Any) -> Any:
        """Write LaTeX source to ``path``, plus a sidecar PDF if the table carries a plot.

        Use this when you want a self-contained LaTeX deliverable with
        the inline plot embedded. The sidecar PDF is named
        ``<stem>_plot.pdf`` and referenced from the LaTeX source via
        ``\\includegraphics``.
        """
        from pathlib import Path
        path = Path(path)
        plot = getattr(table, "inline_plot", None)
        if plot is not None and getattr(plot, "pdf_bytes", None):
            stem = path.stem
            pdf_path = path.with_name(f"{stem}_plot.pdf")
            pdf_path.write_bytes(plot.pdf_bytes)
            self.image_basename = pdf_path.name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(table))
        return path

    # ------------------------------------------------------------------
    def _render_spanning_rows(
        self, spans: tuple[SpanningHeader, ...], ncols: int
    ) -> list[str]:
        if not spans:
            return []
        # Build the row of \multicolumn cells in column order, padding gaps.
        ordered = sorted(spans, key=lambda s: s.start)
        cells: list[str] = []
        cline_parts: list[str] = []
        col = 0
        for span in ordered:
            while col < span.start:
                cells.append("")
                col += 1
            size = span.end - span.start + 1
            cells.append(
                rf"\multicolumn{{{size}}}{{c}}{{{_escape(span.label)}}}"
            )
            cline_parts.append(rf"\cmidrule(lr){{{span.start + 1}-{span.end + 1}}}")
            col = span.end + 1
        while col < ncols:
            cells.append("")
            col += 1
        return [" & ".join(cells) + r" \\", "".join(cline_parts)]

    def _render_header_row(self, hr: Any) -> str:
        parts = [self._render_header_cell(c) for c in hr.cells]
        return " & ".join(parts) + r" \\"

    def _render_header_cell(self, c: HeaderCell) -> str:
        parts = c.text.split("\n")
        if len(parts) > 1:
            body = r" \\ ".join(_escape(p) for p in parts)
            text = rf"\shortstack{{{body}}}"
        else:
            text = _escape(c.text)
        if c.bold:
            return rf"\textbf{{{text}}}"
        return text

    def _render_row(self, r: Row) -> str:
        cells = [self._render_cell(c) for c in r.cells]
        line = " & ".join(cells) + r" \\"
        if r.is_group_header:
            line = r"\addlinespace[0.25em]" + " " + line
        return line

    def _render_cell(self, c: Cell) -> str:
        text = (
            "".join(_render_part_tex(p) for p in c.parts)
            if c.parts
            else _escape(c.text)
        )
        if c.indent > 0:
            text = rf"\hspace{{{c.indent * 1.2:.2f}em}}{text}"
        if c.bold:
            text = rf"\textbf{{{text}}}"
        if c.italic:
            text = rf"\textit{{{text}}}"
        return text


def _render_part_tex(part: Any) -> str:
    """Render a CellPart as LaTeX."""
    s = _escape(part.text)
    if part.code:
        s = rf"\texttt{{{s}}}"
    if part.superscript:
        s = rf"\textsuperscript{{{s}}}"
    if part.subscript:
        s = rf"\textsubscript{{{s}}}"
    if part.italic:
        s = rf"\textit{{{s}}}"
    if part.bold:
        s = rf"\textbf{{{s}}}"
    if part.link:
        s = rf"\href{{{part.link}}}{{{s}}}"
    return s


def _ncols(table: SofraTable) -> int:
    if table.headers:
        return len(table.headers[0].cells)
    if table.rows:
        return len(table.rows[0].cells)
    return 1


def _column_spec(table: SofraTable) -> str:
    """Derive the tabular column alignment spec from the header row."""
    if not table.headers:
        return "l" * _ncols(table)
    cells = table.headers[0].cells
    aligns: list[str] = []
    # First column is conventionally left-aligned (the label column).
    for i, c in enumerate(cells):
        if i == 0:
            aligns.append("l")
        else:
            aligns.append(_align_char(c) if c.align else "c")
    return "".join(aligns)
