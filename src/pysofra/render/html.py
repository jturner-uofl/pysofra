"""HTML rendering, including the rich ``_repr_html_`` notebook output.

The HTML renderer emits a self-contained fragment:

* a unique wrapper class so styles do not leak across multiple tables in
  the same notebook;
* a scoped ``<style>`` block built from the active theme;
* every theme-driven structural style (padding, borders, font) ALSO
  emitted as an inline ``style="..."`` attribute on the affected
  element, so themes stay visibly distinct even in renderers that
  sanitize away inline ``<style>`` blocks (GitHub's notebook viewer,
  some markdown previewers, some PDF/email clients).

Notebook mode and standalone mode produce the same HTML; only a wrapper
``<div class="pysofra-wrap">`` is added in notebook mode so the table can
scroll horizontally inside narrow output cells.
"""

from __future__ import annotations

import hashlib
import html
from dataclasses import dataclass

from ..core.schema import Cell, HeaderCell, HeaderRow, Row, SpanningHeader
from ..core.table import SofraTable
from ..themes.registry import Theme, resolve_theme
from .base import Renderer


def _scope_id_for(table: SofraTable) -> str:
    """Stable per-content scope id.

    Derived from the table's textual content (rows, headers, spans,
    caption, footnotes, theme) so that re-rendering the same table —
    in the same or a different process — always produces an identical
    HTML string. This is required for deterministic snapshot tests
    and reproducible publications.
    """
    h = hashlib.sha256()
    h.update((table.theme_name or "").encode("utf-8"))
    h.update((table.caption or "").encode("utf-8"))
    for fn in table.footnotes:
        h.update(b"\x00f")
        h.update(fn.encode("utf-8"))
    for hr in table.headers:
        h.update(b"\x00h")
        for hc in hr.cells:
            h.update(hc.text.encode("utf-8"))
    for s in table.spanning_headers:
        h.update(f"\x00s{s.label}|{s.start}|{s.end}".encode())
    for r in table.rows:
        h.update(b"\x00r")
        for rc in r.cells:
            h.update(rc.text.encode("utf-8"))
    return f"pysofra-{h.hexdigest()[:10]}"


# ----------------------------------------------------------------------
# Inline-style precomputation
# ----------------------------------------------------------------------

# Properties safe to inline on <table>: they all cascade via CSS
# inheritance to descendant <th> and <td>, so we get themed text
# without bloating every cell with its own style attribute.
_INHERITABLE_TABLE_PROPS = (
    "font-family",
    "font-size",
    "line-height",
    "color",
    "border-collapse",
)

# Per-element structural properties that DO NOT cascade. These get
# inlined onto each affected element so they survive <style>-block
# stripping by sanitizers. The lists are deliberately conservative —
# only the visually significant axes that themes actually vary.
_TH_INLINE_PROPS = (
    "padding",
    "border-top",
    "border-bottom",
    "background",
    "text-align",
    "vertical-align",
)
_TD_INLINE_PROPS = (
    "padding",
    "border-bottom",
    "vertical-align",
)
_LAST_ROW_TD_INLINE_PROPS = (
    "border-bottom",
)
_CAPTION_INLINE_PROPS = (
    "font-family",
    "font-size",
    "font-weight",
    "font-style",
    "padding",
    "text-align",
)
_TFOOT_TD_INLINE_PROPS = (
    "font-family",
    "font-size",
    "color",
    "padding-top",
    "border-bottom",
)


def _css_to_inline(decls: dict[str, str], props: tuple[str, ...]) -> str:
    """Serialise selected declarations as an inline ``style`` value.

    Filters to ``props`` (in order), and rewrites any double-quoted
    CSS string literals — e.g. ``font-family: "Helvetica Neue", ...`` —
    to use single quotes, so the value is safe to drop inside a
    double-quoted HTML attribute. Single quotes are valid CSS string
    delimiters, so the rendered font lookup is unchanged.
    """
    parts: list[str] = []
    for p in props:
        if p in decls:
            parts.append(f"{p}:{decls[p].replace(chr(34), chr(39))}")
    return ";".join(parts)


@dataclass(frozen=True)
class _ThemeInlines:
    """Pre-computed inline-style strings for every themed region.

    Built once per ``render()`` call so the per-cell formatting loop
    is a simple string append rather than a dict lookup + filter on
    every cell.
    """

    table: str
    th: str
    td: str
    last_row_td: str
    caption: str
    tfoot_td: str
    spanning: str


def _theme_inlines(theme: Theme) -> _ThemeInlines:
    """Pre-compute inline style strings for each table region."""
    css = theme.css
    return _ThemeInlines(
        table=_css_to_inline(css.get("table", {}), _INHERITABLE_TABLE_PROPS),
        th=_css_to_inline(css.get("th", {}), _TH_INLINE_PROPS),
        td=_css_to_inline(css.get("td", {}), _TD_INLINE_PROPS),
        last_row_td=_css_to_inline(
            css.get("tr:last-child td", {}), _LAST_ROW_TD_INLINE_PROPS,
        ),
        caption=_css_to_inline(css.get("caption", {}), _CAPTION_INLINE_PROPS),
        tfoot_td=_css_to_inline(css.get("tfoot td", {}), _TFOOT_TD_INLINE_PROPS),
        spanning=_css_to_inline(
            css.get(".pysofra-spanning", {}),
            ("padding", "border-bottom", "text-align", "font-weight"),
        ),
    )


@dataclass
class HtmlRenderer(Renderer[str]):
    notebook: bool = False
    sticky_header: bool = False
    max_height: str | None = None  # CSS length, e.g. "60vh"; enables vertical scroll

    def render(self, table: SofraTable) -> str:
        theme = resolve_theme(table.theme_name)
        scope_id = _scope_id_for(table)
        style = _build_style(theme, scope_id, sticky=self.sticky_header)
        inlines = _theme_inlines(theme)

        head = _render_caption(table, inlines) + _render_thead(table, inlines)
        body = _render_tbody(table, inlines)
        foot = _render_tfoot(table, inlines)
        ncols = _ncols(table)

        table_html = (
            f'<table class="pysofra {scope_id}" role="table" style="{inlines.table}">'
            f"{head}{body}{foot}"
            "</table>"
        )

        wrap_styles: list[str] = []
        if self.notebook or self.max_height:
            wrap_styles.append("overflow-x:auto")
            wrap_styles.append("max-width:100%")
        if self.max_height:
            wrap_styles.append(f"max-height:{self.max_height}")
            wrap_styles.append("overflow-y:auto")
        if wrap_styles:
            wrapper = (
                f'<div class="pysofra-wrap {scope_id}" '
                f'style="{";".join(wrap_styles)};">'
                f"{table_html}</div>"
            )
        else:
            wrapper = table_html

        if table.inline_svg:
            svg_block = (
                f'<div class="pysofra-plot {scope_id}" '
                f'style="max-width:100%;">{table.inline_svg}</div>'
            )
            if table.inline_svg_position == "below":
                wrapper = f"{wrapper}{svg_block}"
            else:
                wrapper = f"{svg_block}{wrapper}"

        del ncols  # silence unused — reserved for future column sizing
        return f"{style}{wrapper}"


# ----------------------------------------------------------------------
# Style block (the <style>...</style> path; still emitted alongside the
# inline attributes because it covers things inline can't, like
# :hover, child selectors, sticky-header positioning).
# ----------------------------------------------------------------------

def _build_style(theme: Theme, scope_id: str, *, sticky: bool = False) -> str:
    """Assemble the scoped stylesheet for one rendered table.

    We deliberately do *not* set an explicit foreground colour — every
    cell inherits the surrounding context (Jupyter's notebook stylesheet,
    or the wrapping HTML document) and borders use ``currentColor`` so
    they always have the same contrast as the text. This avoids the
    common failure mode where a hard-coded dark-mode colour leaks onto a
    light page (and vice-versa).
    """
    blocks: list[str] = []
    for selector, decls in theme.css.items():
        scoped = _scope(selector, scope_id)
        decl_str = "".join(f"{k}:{v};" for k, v in decls.items())
        blocks.append(f"{scoped}{{{decl_str}}}")
    if sticky:
        blocks.append(
            f"table.{scope_id} thead th"
            "{position:sticky;top:0;background:var(--jp-cell-editor-background,#fff);"
            "z-index:1;}"
        )
    return f"<style>{''.join(blocks)}</style>"


def _scope(selector: str, scope_id: str) -> str:
    """Prefix every comma-separated selector with the unique scope class."""
    parts = [p.strip() for p in selector.split(",")]
    return ", ".join(f".{scope_id} {p}" if p != "table" else f"table.{scope_id}" for p in parts)


# ----------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------

def _merge_style(*parts: str) -> str:
    """Merge style fragments, dropping empties and ensuring a single ``;`` join."""
    cleaned = [p.rstrip(";") for p in parts if p]
    return ";".join(cleaned)


def _render_caption(table: SofraTable, inlines: _ThemeInlines) -> str:
    if not table.caption:
        return ""
    style = inlines.caption
    style_attr = f' style="{style}"' if style else ""
    return f"<caption{style_attr}>{html.escape(table.caption)}</caption>"


def _render_thead(table: SofraTable, inlines: _ThemeInlines) -> str:
    rows: list[str] = []
    if table.spanning_headers:
        rows.append(_render_spanning_row(table.spanning_headers, _ncols(table), inlines))
    for hr in table.headers:
        rows.append(_render_header_row(hr, inlines))
    if not rows:
        return ""
    return "<thead>" + "".join(rows) + "</thead>"


def _render_spanning_row(
    spans: tuple[SpanningHeader, ...], ncols: int, inlines: _ThemeInlines,
) -> str:
    # Build a placeholder list — each column is either covered by a span
    # (with the rendered <th colspan>) or rendered as an empty <th>.
    covered = [False] * ncols
    cells_by_index: dict[int, str] = {}
    span_style = (
        f' style="{inlines.spanning}"' if inlines.spanning else ""
    )
    for span in spans:
        size = span.end - span.start + 1
        cells_by_index[span.start] = (
            f'<th class="pysofra-spanning" colspan="{size}"{span_style}>'
            f"{html.escape(span.label)}</th>"
        )
        for i in range(span.start, span.end + 1):
            covered[i] = True

    out: list[str] = []
    i = 0
    while i < ncols:
        if i in cells_by_index:
            out.append(cells_by_index[i])
            # skip to end of span
            span = next(s for s in spans if s.start == i)
            i = span.end + 1
        else:
            if not covered[i]:
                out.append("<th></th>")
            i += 1
    return "<tr>" + "".join(out) + "</tr>"


def _render_header_row(hr: HeaderRow, inlines: _ThemeInlines) -> str:
    return "<tr>" + "".join(_render_header_cell(c, inlines) for c in hr.cells) + "</tr>"


def _render_header_cell(c: HeaderCell, inlines: _ThemeInlines) -> str:
    # Per-cell overrides (alignment, bold) come AFTER the theme styles
    # so they win the cascade — last declaration wins for equal
    # specificity within a single inline attribute.
    per_cell_parts: list[str] = []
    if c.align:
        per_cell_parts.append(f"text-align:{c.align}")
    if c.bold:
        per_cell_parts.append("font-weight:600")
    per_cell = ";".join(per_cell_parts)
    style = _merge_style(inlines.th, per_cell)
    style_attr = f' style="{style}"' if style else ""
    # Newlines in header text indicate stacked label (e.g. "Group A\nN=10").
    parts = c.text.split("\n")
    body = "<br>".join(html.escape(p) for p in parts) if len(parts) > 1 else html.escape(c.text)
    return f'<th{style_attr}>{body}</th>'


def _render_tbody(table: SofraTable, inlines: _ThemeInlines) -> str:
    if not table.rows:
        return "<tbody></tbody>"
    last_idx = len(table.rows) - 1
    return (
        "<tbody>"
        + "".join(
            _render_row(r, inlines, is_last=(i == last_idx))
            for i, r in enumerate(table.rows)
        )
        + "</tbody>"
    )


def _render_row(r: Row, inlines: _ThemeInlines, *, is_last: bool) -> str:
    cls = " class=\"group-header\"" if r.is_group_header else ""
    style = ""
    if r.metadata:
        highlight = r.metadata.get("highlight")
        if highlight:
            style = f' style="background:{html.escape(str(highlight))};"'
    return (
        f"<tr{cls}{style}>"
        + "".join(_render_cell(c, inlines, is_last_row=is_last) for c in r.cells)
        + "</tr>"
    )


def _render_cell(c: Cell, inlines: _ThemeInlines, *, is_last_row: bool) -> str:
    classes: list[str] = []
    per_cell_parts: list[str] = []
    if c.kind in ("numeric", "p_value", "ci"):
        classes.append("pysofra-num")
    if c.align:
        per_cell_parts.append(f"text-align:{c.align}")
    if c.bold:
        classes.append("pysofra-bold")
    if c.italic:
        per_cell_parts.append("font-style:italic")
    if c.indent > 0:
        per_cell_parts.append(f"padding-left:{0.75 + 1.0 * c.indent:.2f}em")
    # Cell-level renderer-specific overrides (style['html'] is appended as-is).
    if c.style and isinstance(c.style.get("html"), str):
        extra = c.style["html"].strip().rstrip(";")
        if extra:
            per_cell_parts.append(extra)

    # Theme inline styles first (lowest priority within the attribute);
    # per-cell overrides last so they win for equal-specificity props.
    theme_part = inlines.td
    if is_last_row and inlines.last_row_td:
        # last-row border-bottom must override the general td rule
        theme_part = _merge_style(theme_part, inlines.last_row_td)
    style = _merge_style(theme_part, ";".join(per_cell_parts))

    class_attr = f' class="{" ".join(classes)}"' if classes else ""
    style_attr = f' style="{style}"' if style else ""
    body = _render_parts(c) if c.parts else html.escape(c.text)
    return f"<td{class_attr}{style_attr}>{body}</td>"


def _render_parts(c: Cell) -> str:
    """Render a rich cell (``c.parts``) as HTML runs."""
    if not c.parts:
        return html.escape(c.text)
    out: list[str] = []
    for p in c.parts:
        s = html.escape(p.text)
        if p.code:
            s = f"<code>{s}</code>"
        if p.superscript:
            s = f"<sup>{s}</sup>"
        if p.subscript:
            s = f"<sub>{s}</sub>"
        if p.italic:
            s = f"<em>{s}</em>"
        if p.bold:
            s = f"<strong>{s}</strong>"
        if p.color:
            s = f'<span style="color:{html.escape(p.color)};">{s}</span>'
        if p.link:
            s = f'<a href="{html.escape(p.link)}">{s}</a>'
        out.append(s)
    return "".join(out)


def _render_tfoot(table: SofraTable, inlines: _ThemeInlines) -> str:
    if not table.footnotes:
        return ""
    ncols = _ncols(table)
    lines = "<br>".join(html.escape(f) for f in table.footnotes)
    style_attr = f' style="{inlines.tfoot_td}"' if inlines.tfoot_td else ""
    return (
        "<tfoot><tr>"
        f'<td colspan="{ncols}"{style_attr}>{lines}</td>'
        "</tr></tfoot>"
    )


def _ncols(table: SofraTable) -> int:
    if table.headers:
        return len(table.headers[0].cells)
    if table.rows:
        return len(table.rows[0].cells)
    return 1
