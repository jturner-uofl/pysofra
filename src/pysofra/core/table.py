"""The core :class:`SofraTable` object.

A SofraTable is a backend-agnostic representation of a publication-ready
statistical table. Every builder (``tbl_one``, ``tbl_summary``,
``tbl_regression``) produces a SofraTable; every renderer (HTML, Markdown,
DOCX) consumes one.

The object is immutable: every modifier method (``.theme()``, ``.caption()``,
``.add_p()``, ...) returns a *new* SofraTable. This keeps results
deterministic and notebook-friendly.

Some modifier methods (``.add_p``, ``.add_smd``, ``.add_overall``) need to
recompute statistics from the original data. A SofraTable produced by a
builder therefore carries a private ``_context`` callable that can rebuild
the table under an updated spec. Tables that have no recompute context
(e.g. those produced by merge/stack operations) silently ignore those
operations or raise, as appropriate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .schema import HeaderRow, Row, SpanningHeader


def _row_p_below(row: Row, threshold: float) -> bool:
    """Predicate: row has a numeric p-value cell below ``threshold``."""
    for c in row.cells:
        if c.kind == "p_value" and isinstance(c.value, (int, float)):
            return float(c.value) < threshold
    return False

if TYPE_CHECKING:
    pass


RebuildFn = Callable[["TableSpec"], "SofraTable"]


@dataclass(frozen=True)
class TableSpec:
    """Specification carried by builder-produced tables for recomputation.

    The fields are deliberately generic (a dict) because different builders
    have different knobs; each builder reads only the keys it cares about.
    """

    builder: str  # "tbl_one", "tbl_summary", "tbl_regression", ...
    options: dict[str, Any] = field(default_factory=dict)

    def updated(self, **changes: Any) -> TableSpec:
        new_opts = dict(self.options)
        new_opts.update(changes)
        return TableSpec(builder=self.builder, options=new_opts)


@dataclass(frozen=True)
class SofraTable:
    """A backend-agnostic publication-ready table.

    Attributes
    ----------
    rows
        The table body, one :class:`~pysofra.core.schema.Row` per visible line.
    headers
        Column header rows, top-to-bottom. Most tables have a single header
        row; multi-level headers are supported via additional rows.
    spanning_headers
        Optional spanning headers above the column headers.
    caption
        Optional table caption (rendered as a title above the table).
    footnotes
        Tuple of footnote strings rendered below the table.
    theme_name
        Theme key registered in :mod:`pysofra.themes`. Defaults to
        ``"default"``.
    metadata
        Free-form metadata dict carried for downstream consumers and tests
        (e.g. raw p-values, SMDs, model objects).
    """

    rows: tuple[Row, ...] = ()
    headers: tuple[HeaderRow, ...] = ()
    spanning_headers: tuple[SpanningHeader, ...] = ()
    caption: str | None = None
    footnotes: tuple[str, ...] = ()
    theme_name: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)
    inline_svg: str | None = None
    inline_svg_position: str = "above"  # "above" | "below"
    inline_plot: Any = None  # InlinePlot — typed loosely to avoid plot-import cycle
    _spec: TableSpec | None = None
    _rebuild: RebuildFn | None = None

    # ------------------------------------------------------------------
    # Pickling
    # ------------------------------------------------------------------
    #
    # ``_rebuild`` is a closure produced by the builders (``tbl_one``
    # etc.) that captures the original DataFrame and the spec, so that
    # recomputation modifiers (``add_p``, ``add_smd``, ...) can re-run
    # statistics over the source data. Closures are not pickleable.
    # We strip it from the pickled state and replace it with ``None``
    # on restore — the unpickled object keeps all rendered cell values
    # and can be re-rendered to HTML/Markdown/LaTeX/DOCX/PPTX/XLSX/PNG
    # without issue, but recomputation modifiers (the
    # ``add_p``/``add_smd``/etc. family that requires the captured
    # data) will raise a clear error explaining the limitation. The
    # presentational modifiers (``theme``, ``set_caption``,
    # ``with_footnotes``, ``bold_p``, ``with_forest_plot`` …) keep
    # working because they don't need ``_rebuild``.

    # ------------------------------------------------------------------
    # Equality
    # ------------------------------------------------------------------
    #
    # The dataclass-generated ``__eq__`` compares every field including
    # ``_rebuild``, which is a closure with per-instance identity. Two
    # tables built from the same data by the same builder therefore
    # tested unequal, and a pickled-then-unpickled table tested unequal
    # to its source (the closure is dropped on pickle).
    #
    # Define equality on the output-affecting fields only: rows,
    # headers, spanning_headers, caption, footnotes, theme_name,
    # inline_svg, inline_svg_position, inline_plot. Skip ``metadata``
    # (free-form internal carry that can hold non-equal-comparable
    # values such as model objects), ``_spec`` (build-time descriptor),
    # and ``_rebuild`` (closure).
    def __eq__(self, other: object) -> bool:
        if other is self:
            return True
        if not isinstance(other, SofraTable):
            return NotImplemented
        return (
            self.rows == other.rows
            and self.headers == other.headers
            and self.spanning_headers == other.spanning_headers
            and self.caption == other.caption
            and self.footnotes == other.footnotes
            and self.theme_name == other.theme_name
            and self.inline_svg == other.inline_svg
            and self.inline_svg_position == other.inline_svg_position
            and self.inline_plot == other.inline_plot
        )

    # Frozen-dataclass auto-generates a ``__hash__`` based on every
    # field; with our custom ``__eq__`` it would violate the eq-vs-hash
    # contract (and would fail anyway on the ``metadata: dict`` field).
    # Mark explicitly unhashable so the error is a sharp ``TypeError``,
    # not a silent inconsistency between equal-but-different-hash tables.
    __hash__ = None  # type: ignore[assignment]

    def __getstate__(self) -> dict[str, Any]:
        state = {
            "rows": self.rows,
            "headers": self.headers,
            "spanning_headers": self.spanning_headers,
            "caption": self.caption,
            "footnotes": self.footnotes,
            "theme_name": self.theme_name,
            "metadata": self.metadata,
            "inline_svg": self.inline_svg,
            "inline_svg_position": self.inline_svg_position,
            "inline_plot": self.inline_plot,
            "_spec": self._spec,
            "_rebuild": None,  # closures are not pickleable; see class docstring
        }
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        # SofraTable is frozen; bypass __setattr__ block.
        for k, v in state.items():
            object.__setattr__(self, k, v)

    # ------------------------------------------------------------------
    # Composition / styling modifiers
    # ------------------------------------------------------------------
    def theme(self, name: str) -> SofraTable:
        """Apply a theme by name. See :mod:`pysofra.themes` for available themes."""
        from ..themes.registry import resolve_theme

        resolve_theme(name)  # validates
        return replace(self, theme_name=name)

    def set_caption(self, text: str | None) -> SofraTable:
        """Set the table caption, replacing any existing one.

        The caption renders above the table in every backend
        (``<caption>`` in HTML, ``\\caption{}`` in LaTeX, a bold
        paragraph above the table in DOCX/PPTX, sheet header in
        XLSX). Pass ``None`` to clear an existing caption.

        Parameters
        ----------
        text
            The caption string, or ``None`` to clear.

        Returns
        -------
        SofraTable
            A new SofraTable with the caption set. The original is
            unchanged.
        """
        return replace(self, caption=text)

    def add_footnote(self, text: str) -> SofraTable:
        """Append a single footnote to the existing footnote list.

        Footnotes render below the table in every backend, in the
        order they were appended. Builders such as :func:`tbl_one`
        emit footnotes automatically (e.g. "Tests:", "n (%) for
        categorical variables.") describing test choices and
        formatting; user-supplied footnotes appended via this method
        appear *after* the auto-generated ones.

        Parameters
        ----------
        text
            The footnote text. Plain text; renderers escape special
            characters appropriately for their backend.

        Returns
        -------
        SofraTable
            A new SofraTable with the footnote appended. To replace
            the footnote list wholesale, use
            :meth:`with_footnotes` instead.
        """
        return replace(self, footnotes=tuple([*self.footnotes, text]))

    def with_footnotes(self, footnotes: list[str] | tuple[str, ...]) -> SofraTable:
        """Replace the footnote list entirely."""
        return replace(self, footnotes=tuple(footnotes))

    # ------------------------------------------------------------------
    # Statistical modifiers (require a rebuild context)
    # ------------------------------------------------------------------
    def add_p(self, **overrides: Any) -> SofraTable:
        """Add a p-value column.

        Behaviour depends on the builder that produced the table. For
        ``tbl_one`` / ``tbl_summary`` this triggers automatic test
        selection per row (see :mod:`pysofra.summary.tests`); for
        ``tbl_regression`` p-values are already present and this is a
        no-op.
        """
        return self._with_option(p_value=True, p_overrides=dict(overrides))

    def add_smd(self) -> SofraTable:
        """Add a standardized-mean-difference column (Table 1 only)."""
        return self._with_option(smd=True)

    def add_q(self, method: str = "fdr_bh") -> SofraTable:
        """Add a multiplicity-adjusted q-value column.

        ``method`` is passed through to
        ``statsmodels.stats.multitest.multipletests``; common choices are
        ``fdr_bh`` (Benjamini–Hochberg, default), ``fdr_by``,
        ``bonferroni``, ``holm``, ``hommel``, ``sidak``. Implicitly
        enables p-values when not already on.
        """
        return self._with_option(p_value=True, q_value=True, q_method=method)

    # ------------------------------------------------------------------
    # Extras (gtsummary parity)
    # ------------------------------------------------------------------
    def add_significance_stars(
        self,
        *,
        thresholds: tuple[tuple[float, str], ...] = (
            (0.001, "***"),
            (0.01, "**"),
            (0.05, "*"),
        ),
    ) -> SofraTable:
        """Append a ``stars`` column with ``*** / ** / *`` significance markers.

        ``thresholds`` is a tuple of ``(cutoff, marker)`` pairs sorted
        smallest-cutoff first; each p-value is marked with the first
        marker whose cutoff it falls below.
        """
        from ..summary.extras import add_significance_stars

        return add_significance_stars(self, thresholds=thresholds)

    def add_n(self) -> SofraTable:
        """Append a per-row ``N`` column with the non-missing sample size."""
        from ..summary.extras import add_n

        return add_n(self)

    def add_stat_label(self) -> SofraTable:
        """Append a ``Statistic`` column describing each row's summary form."""
        from ..summary.extras import add_stat_label

        return add_stat_label(self)

    def color_scale_if(
        self,
        *,
        column: int,
        palette: tuple[str, str, str] = ("#fff5f0", "#fcae91", "#cb181d"),
        skip_blank: bool = True,
    ) -> SofraTable:
        """Heatmap-style cell colouring for one numeric column (HTML only)."""
        from ..summary.extras import color_scale_if

        return color_scale_if(self, column=column, palette=palette,
                              skip_blank=skip_blank)

    def add_global_p(
        self, *, adjust_for: list[str] | tuple[str, ...] | None = None,
    ) -> SofraTable:
        """Add a joint Type-III p-value column.

        Supported on both :func:`tbl_regression` and
        :func:`tbl_one` / :func:`tbl_summary` tables, via two paths:

        * **tbl_regression** — for each multi-level categorical
          predictor in the model, the joint Wald-F p-value is
          computed via ``model.f_test`` on the contrast matrix that
          zeroes out every level simultaneously. Single-level
          coefficients receive their existing p-value duplicated.
        * **tbl_one / tbl_summary** — for each variable in the table,
          a logistic regression is fit on the source data:
          ``Logit(by == reference_level ~ variable [+ adjust_for])``.
          The joint Wald p-value across the variable's coefficients is
          the new "global p" cell. Adjustment covariates passed via
          ``adjust_for=`` apply to every variable's fit, giving
          covariate-adjusted joint p-values.

        Parameters
        ----------
        adjust_for
            (tbl_one / tbl_summary only) Optional list of covariate
            column names to include in each per-variable regression.
            Continuous numeric covariates enter as-is; non-numeric
            covariates are dummy-coded. Ignored on
            :func:`tbl_regression` tables.

        Raises
        ------
        NotImplementedError
            On composition primitives (``tbl_merge`` / ``tbl_stack``)
            and directly-constructed tables that carry neither a
            fitted ``model`` nor a re-runnable builder spec.
        """
        # tbl_regression path: model is attached in metadata.
        if (self.metadata or {}).get("model") is not None:
            from ..summary.extras import add_global_p

            return add_global_p(self)
        # tbl_one / tbl_summary path: route through the rebuild spec.
        # The rebuild reconstructs the table from spec.options only;
        # columns added by post-build modifiers (``add_difference``,
        # ``add_ci``, ``add_significance_stars``, ...) live in
        # ``self.rows``/``self.headers`` and are NOT preserved by the
        # rebuild. Detect a *known* such column by header text and warn
        # the user so the silent column-drop doesn't mislead them.
        # The correct chaining order is to call ``add_global_p()``
        # *before* any column-adding modifier.
        spec = self._spec
        if spec is not None and spec.builder in ("tbl_one", "tbl_summary"):
            post_build_headers = {"Diff", "[", "[ "}
            header_texts = (
                [c.text for c in self.headers[0].cells] if self.headers else []
            )
            has_diff_col = any(h.startswith("Diff (") for h in header_texts)
            has_sig_col = any(h.lower() == "signif." for h in header_texts)
            del post_build_headers
            if has_diff_col or has_sig_col:
                import warnings as _w
                _w.warn(
                    "add_global_p() reruns the table builder; any "
                    "column added by a prior modifier (e.g. add_difference, "
                    "add_significance_stars) will be dropped. Call "
                    "add_global_p() BEFORE those modifiers to preserve "
                    "their columns.",
                    UserWarning,
                    stacklevel=2,
                )
            return self._with_option(
                global_p=True,
                global_p_adjust_for=tuple(adjust_for or ()),
            )
        from ..summary.extras import add_global_p

        # Falls through to the original error path (which differentiates
        # tbl_cross / composition / unpickled origins).
        return add_global_p(self)

    def add_difference(self, *, digits: int = 2,
                       conf_level: float = 0.95) -> SofraTable:
        """Add a between-group difference column (continuous + dichotomous).

        Requires a 2-group Table 1. Continuous rows get the Welch
        mean-difference + CI; dichotomous rows get the proportion
        difference with Wilson-score-based CI; multi-level categorical
        rows show ``—``.
        """
        from ..summary.extras import add_difference

        return add_difference(self, digits=digits, conf_level=conf_level)

    def add_ci(self, *, conf_level: float = 0.95) -> SofraTable:
        """Append a confidence interval to each summary cell.

        Continuous cells gain ``[lo, hi]`` for the mean; dichotomous
        cells gain ``[lo%, hi%]`` for the proportion (Wilson score).
        """
        from ..summary.extras import add_ci

        return add_ci(self, conf_level=conf_level)

    def with_pvalue_fmt(self, fn: Callable[[float], str]) -> SofraTable:
        """Re-format every p-value cell with the supplied callable."""
        from ..summary.extras import with_pvalue_fmt

        return with_pvalue_fmt(self, fn)

    def with_estimate_fmt(self, fn: Callable[[float], str]) -> SofraTable:
        """Re-format every numeric estimate cell with the supplied callable."""
        from ..summary.extras import with_estimate_fmt

        return with_estimate_fmt(self, fn)

    # ------------------------------------------------------------------
    # Layout hints
    # ------------------------------------------------------------------
    def autofit(self, *, enable: bool = True) -> SofraTable:
        """Hint every renderer to size columns to content.

        Stored as ``metadata['autofit']``. HTML uses content-based sizing
        natively; XLSX auto-sizes column widths to the widest cell; the
        DOCX renderer sets ``table.autofit = True`` when this flag is on.
        """
        new_md = dict(self.metadata) if self.metadata else {}
        new_md["autofit"] = bool(enable)
        return replace(self, metadata=new_md)

    # ------------------------------------------------------------------
    # Rich-cell composition
    # ------------------------------------------------------------------
    def compose(
        self,
        row: int | str,
        column: int | str,
        parts: Any,
    ) -> SofraTable:
        """Replace a cell's content with multiple typographically distinct parts.

        ``parts`` is an iterable of :class:`~pysofra.core.schema.CellPart`
        — each carries its own ``bold`` / ``italic`` / ``superscript`` /
        ``subscript`` / ``color`` / ``link`` flags. Renderers concatenate
        the parts inside the same cell, honouring whichever flags the
        backend supports; the fallback ``text`` is set to the
        concatenated plain text so non-rich backends still print
        something readable.
        """
        from .schema import CellPart as _CellPart

        if isinstance(row, int):
            r_idx = row
        else:
            r_idx = next(
                (i for i, r in enumerate(self.rows) if r.cells[0].text == row),
                -1,
            )
            if r_idx == -1:
                raise KeyError(f"No row labelled {row!r}")
        if not 0 <= r_idx < len(self.rows):
            raise KeyError(f"row {row!r} out of range")

        if isinstance(column, int):
            c_idx = column
        else:
            header_cells = self.headers[-1].cells if self.headers else ()
            c_idx = next(
                (j for j, c in enumerate(header_cells)
                 if c.text.replace("\n", " ") == column or c.text == column),
                -1,
            )
            if c_idx == -1:
                raise KeyError(f"No column labelled {column!r}")
        if not 0 <= c_idx < len(self.rows[r_idx].cells):
            raise KeyError(f"column {column!r} out of range")

        parts_tuple = tuple(parts)
        for p in parts_tuple:
            if not isinstance(p, _CellPart):
                raise TypeError(
                    f"compose() parts must be CellPart instances; "
                    f"got {type(p).__name__}."
                )

        old_row = self.rows[r_idx]
        new_cells = list(old_row.cells)
        fallback = "".join(p.text for p in parts_tuple)
        new_cells[c_idx] = replace(
            new_cells[c_idx],
            text=fallback,
            parts=parts_tuple,
        )
        new_rows = list(self.rows)
        new_rows[r_idx] = replace(old_row, cells=tuple(new_cells))
        return replace(self, rows=tuple(new_rows))

    # ------------------------------------------------------------------
    # Spanning headers (manual API)
    # ------------------------------------------------------------------
    def modify_spanning_header(
        self,
        label: str,
        *,
        start: int,
        end: int,
    ) -> SofraTable:
        """Add (or replace at the same range) a spanning header above columns.

        Columns are 0-indexed and the range is inclusive on both ends.
        Overlapping a previous span removes it.
        """
        from .schema import SpanningHeader
        ncols = (
            len(self.headers[0].cells)
            if self.headers
            else (len(self.rows[0].cells) if self.rows else 1)
        )
        if start < 0 or end >= ncols or start > end:
            raise ValueError(
                f"start={start}, end={end} out of range for {ncols} columns."
            )
        # Drop any existing span that overlaps.
        kept = tuple(
            s for s in self.spanning_headers
            if s.end < start or s.start > end
        )
        new_span = SpanningHeader(label=label, start=start, end=end)
        return replace(self, spanning_headers=kept + (new_span,))

    # ------------------------------------------------------------------
    # Inline text extraction (in-prose)
    # ------------------------------------------------------------------
    def inline_text(self, *, row: int | str, column: int | str) -> str:
        """Pull the text of a single cell for inline use.

        ``row`` and ``column`` accept either a 0-indexed integer or a
        string matched against the first cell of each row / each header
        cell text. Raises ``KeyError`` if no match is found.
        """
        # Resolve row index.
        if isinstance(row, int):
            r_idx = row
        else:
            r_idx = next(
                (i for i, r in enumerate(self.rows) if r.cells[0].text == row),
                -1,
            )
            if r_idx == -1:
                raise KeyError(f"No row labelled {row!r}")
        if not 0 <= r_idx < len(self.rows):
            raise KeyError(f"row {row!r} out of range")

        # Resolve column index.
        if isinstance(column, int):
            c_idx = column
        else:
            header_cells = self.headers[-1].cells if self.headers else ()
            c_idx = next(
                (j for j, c in enumerate(header_cells)
                 if c.text.replace("\n", " ") == column or c.text == column),
                -1,
            )
            if c_idx == -1:
                raise KeyError(f"No column labelled {column!r}")

        cells = self.rows[r_idx].cells
        if not 0 <= c_idx < len(cells):
            raise KeyError(f"column {column!r} out of range")
        return cells[c_idx].text

    # ------------------------------------------------------------------
    # Raster export (PNG of the table itself)
    # ------------------------------------------------------------------
    def to_image(
        self,
        path: str | Path,
        *,
        scale: float = 2.0,
        dpi: int = 300,
    ) -> Path:
        """Render the table to a PNG image.

        Uses matplotlib under the hood; the result is a faithful raster
        of the HTML output. Useful for quick previews, Slack attachments,
        and document figures where a static image is preferable.

        ``scale`` multiplies the pixel density (>= 1 recommended);
        ``dpi`` controls the output resolution (defaults to 300, the
        usual print-quality target).
        """
        from ..render.image import write_image

        return write_image(self, Path(path), scale=scale, dpi=dpi)

    def add_overall(self, label: str = "Overall") -> SofraTable:
        """Add an overall (unstratified) column."""
        return self._with_option(overall=True, overall_label=label)

    def bold_p(self, threshold: float = 0.05) -> SofraTable:
        """Bold rows whose p-value cell carries a value below ``threshold``.

        This is a presentational modifier — it works on any SofraTable
        whose body rows contain a cell of kind ``p_value`` with a numeric
        ``value``.
        """
        threshold = float(threshold)
        return self.bold_if(lambda r: _row_p_below(r, threshold))

    # ------------------------------------------------------------------
    # Conditional formatting
    # ------------------------------------------------------------------
    def bold_if(self, predicate: Callable[[Row], bool]) -> SofraTable:
        """Bold every cell of rows satisfying ``predicate(row) -> bool``.

        Example::

            table.bold_if(lambda r: r.cells[0].text.startswith('age'))
        """
        new_rows: list[Row] = []
        for r in self.rows:
            if predicate(r):
                new_cells = tuple(
                    replace(c, bold=True) if c.text else c for c in r.cells
                )
                new_rows.append(replace(r, cells=new_cells))
            else:
                new_rows.append(r)
        return replace(self, rows=tuple(new_rows))

    def highlight_if(
        self,
        predicate: Callable[[Row], bool],
        *,
        color: str = "#fff3cd",
    ) -> SofraTable:
        """Highlight rows (background colour) satisfying ``predicate``.

        Adds an ``html_style`` metadata entry consumed by the HTML
        renderer; ignored by Markdown / LaTeX. ``color`` accepts any CSS
        colour string.
        """
        new_rows: list[Row] = []
        for r in self.rows:
            if predicate(r):
                md = dict(r.metadata) if r.metadata else {}
                md["highlight"] = color
                new_rows.append(replace(r, metadata=md))
            else:
                new_rows.append(r)
        return replace(self, rows=tuple(new_rows))

    def style_if(
        self,
        predicate: Callable[[Row], bool],
        *,
        bold: bool = False,
        italic: bool = False,
        color: str | None = None,
    ) -> SofraTable:
        """General-purpose conditional row styling.

        Combines :meth:`bold_if`, italic toggling, and an optional row
        background highlight in one call.
        """
        out = self
        if bold:
            out = out.bold_if(predicate)
        if italic:
            new_rows: list[Row] = []
            for r in out.rows:
                if predicate(r):
                    new_cells = tuple(
                        replace(c, italic=True) if c.text else c for c in r.cells
                    )
                    new_rows.append(replace(r, cells=new_cells))
                else:
                    new_rows.append(r)
            out = replace(out, rows=tuple(new_rows))
        if color is not None:
            out = out.highlight_if(predicate, color=color)
        return out

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def to_html(
        self,
        *,
        sticky_header: bool = False,
        max_height: str | None = None,
    ) -> str:
        """Render the table as a standalone HTML fragment.

        ``sticky_header=True`` keeps the column headers in view as the
        body scrolls — pair with ``max_height`` (a CSS length like
        ``"60vh"`` or ``"400px"``) to enable the vertical scroll
        container.
        """
        from ..render.html import HtmlRenderer

        return HtmlRenderer(
            sticky_header=sticky_header,
            max_height=max_height,
        ).render(self)

    def to_markdown(self) -> str:
        """Render the table as GitHub-flavored Markdown."""
        from ..render.markdown import MarkdownRenderer

        return MarkdownRenderer().render(self)

    def to_docx(self, path: str | Path) -> Path:
        """Write the table to a ``.docx`` file. Returns the resolved path."""
        from ..render.docx import DocxRenderer

        return DocxRenderer().write(self, Path(path))

    def to_latex(self, *, booktabs: bool = True,
                 float_position: str = "ht",
                 centering: bool = True) -> str:
        """Render the table as a LaTeX ``table`` float (booktabs by default).

        Requires ``\\usepackage{booktabs}`` in the consumer document
        preamble. Returns the LaTeX source as a string; write it to a
        ``.tex`` file with :func:`pathlib.Path.write_text` if needed.

        Inline plots are embedded only when using :meth:`to_latex_file`
        (which writes a sidecar PDF). For a plain LaTeX string call this
        method and ignore any attached plot.
        """
        from ..render.latex import LatexRenderer

        return LatexRenderer(
            booktabs=booktabs,
            float_position=float_position,
            centering=centering,
        ).render(self)

    def to_latex_file(self, path: str | Path, *, booktabs: bool = True,
                      float_position: str = "ht",
                      centering: bool = True) -> Path:
        """Write a ``.tex`` file plus a sidecar PDF for any inline plot.

        If the table carries an :class:`~pysofra.plot.InlinePlot`, the
        plot is written as ``<stem>_plot.pdf`` next to the ``.tex`` file
        and embedded with ``\\includegraphics``. Requires ``graphicx`` in
        the consuming document preamble.
        """
        from ..render.latex import LatexRenderer

        result = LatexRenderer(
            booktabs=booktabs,
            float_position=float_position,
            centering=centering,
        ).write(self, Path(path))
        return Path(result)

    def to_pptx(self, path: str | Path, *,
                slide_title: str | None = None) -> Path:
        """Write the table to a single-slide ``.pptx`` file.

        Requires the optional ``python-pptx`` dependency
        (``pip install pysofra[pptx]``). If ``slide_title`` is omitted,
        the table's caption is used.
        """
        from ..render.pptx import PptxRenderer

        return PptxRenderer(slide_title=slide_title).write(self, Path(path))

    def to_xlsx(self, path: str | Path, *, sheet_name: str = "Table") -> Path:
        """Write the table to an ``.xlsx`` file via ``xlsxwriter``."""
        from ..render.xlsx import XlsxRenderer

        if XlsxRenderer is None:  # pragma: no cover
            raise ImportError(
                "Excel export requires xlsxwriter. "
                "Install with `pip install xlsxwriter`."
            )
        return XlsxRenderer(sheet_name=sheet_name).write(self, Path(path))

    # ------------------------------------------------------------------
    # Inline plot attachment
    # ------------------------------------------------------------------
    def with_inline_svg(
        self,
        svg: str,
        *,
        position: str = "above",
    ) -> SofraTable:
        """Attach a raw inline-SVG string to this table.

        The HTML renderer embeds the SVG above (default) or below the
        table. Markdown ignores the SVG (no in-line image syntax for
        SVG strings). For a plot that needs to travel through DOCX /
        LaTeX / PPTX as well, use :meth:`with_forest_plot` or
        :meth:`with_km_plot` instead — those serialise a matplotlib
        figure into SVG + PNG + PDF and each renderer picks the
        format it supports.
        """
        if position not in ("above", "below"):
            raise ValueError("position must be 'above' or 'below'")
        return replace(self, inline_svg=svg, inline_svg_position=position)

    def with_forest_plot(
        self,
        *,
        log_x: bool = True,
        null_line: float = 1.0,
        position: str = "above",
        **plot_kwargs: Any,
    ) -> SofraTable:
        """Attach a forest plot rendered from this regression table's coefficients.

        Only valid for tables produced by :func:`tbl_regression`. Reads
        the point estimate + CI cells directly from the table body so
        the plot is guaranteed to match the displayed numbers. The
        attached plot carries SVG / PNG / PDF serialisations so it
        embeds in HTML, DOCX, PPTX, and LaTeX output consistently.
        """
        from ..plot.forest import forest_plot

        plot = forest_plot(self, log_x=log_x, null_line=null_line, **plot_kwargs)
        if position not in ("above", "below"):
            raise ValueError("position must be 'above' or 'below'")
        return replace(
            self,
            inline_svg=plot.svg,
            inline_svg_position=position,
            inline_plot=plot,
        )

    def with_km_plot(
        self,
        *,
        position: str = "above",
        **plot_kwargs: Any,
    ) -> SofraTable:
        """Attach a Kaplan–Meier curve to a :func:`tbl_survival` result.

        Refits the KM curves from the original data using ``lifelines``
        and embeds SVG + PNG + PDF serialisations so the same plot
        renders in HTML, DOCX, PPTX, and LaTeX exports.
        """
        from ..models.survival import attach_km_plot

        return attach_km_plot(self, position=position, **plot_kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Dump the table as a plain dict (useful for snapshot tests)."""
        return {
            "caption": self.caption,
            "footnotes": list(self.footnotes),
            "theme": self.theme_name,
            "headers": [[c.text for c in hr.cells] for hr in self.headers],
            "spanning_headers": [
                {"label": s.label, "start": s.start, "end": s.end}
                for s in self.spanning_headers
            ],
            "rows": [[c.text for c in r.cells] for r in self.rows],
        }

    # ------------------------------------------------------------------
    # Notebook integration
    # ------------------------------------------------------------------
    def _repr_html_(self) -> str:  # noqa: D401 — Jupyter API
        """Rich HTML rendering for Jupyter / Colab / VS Code notebooks."""
        from ..render.html import HtmlRenderer

        return HtmlRenderer(notebook=True).render(self)

    def _repr_markdown_(self) -> str:  # noqa: D401 — Jupyter API
        """Markdown rendering used by Quarto and some markdown-first viewers."""
        return self.to_markdown()

    def _repr_latex_(self) -> str:  # noqa: D401 — Jupyter API
        """LaTeX rendering for environments that prefer it over HTML."""
        return self.to_latex()

    def __repr__(self) -> str:  # pragma: no cover — repr is cosmetic
        ncols = len(self.headers[0].cells) if self.headers else 0
        return f"SofraTable(rows={len(self.rows)}, cols={ncols}, theme={self.theme_name!r})"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _with_option(self, **changes: Any) -> SofraTable:
        if self._spec is None or self._rebuild is None:
            # Three distinct routes produce this state; differentiate them
            # so the error names the actual cause rather than guessing.
            #   (a) ``_spec is None`` — built by composition primitives
            #       (``tbl_merge`` / ``tbl_stack``) or constructed directly
            #       via ``SofraTable(...)``.
            #   (b) ``_spec.builder == 'tbl_cross'`` — a builder that
            #       deliberately doesn't capture a re-runnable spec.
            #   (c) otherwise — unpickled; ``_rebuild`` was stripped on
            #       pickle (see ``__getstate__``).
            if self._spec is None:
                cause = (
                    "either constructed directly via ``SofraTable(...)`` "
                    "or produced by a composition primitive "
                    "(``tbl_merge`` / ``tbl_stack``) — neither path carries "
                    "a re-runnable spec"
                )
            elif self._spec.builder == "tbl_cross":  # pragma: no cover — tbl_cross now carries _rebuild; reachable only on an unpickled tbl_cross (rebuild closure stripped)
                cause = (
                    "an unpickled ``tbl_cross`` table — the recomputation "
                    "closure was stripped on pickle (see "
                    "SofraTable.__getstate__). Re-run ``tbl_cross`` on "
                    "the source DataFrame to restore recomputation"
                )
            else:
                cause = (
                    "unpickled — the recomputation closure ``_rebuild`` is "
                    "stripped on pickle (see SofraTable.__getstate__). "
                    "Re-run the original builder on the source DataFrame to "
                    "restore recomputation"
                )
            raise RuntimeError(
                f"This SofraTable cannot apply statistical modifiers because "
                f"it is {cause}. Presentational modifiers (theme, "
                f"set_caption, with_footnotes, bold_p, with_forest_plot, "
                f"etc.) and renderers still work."
            )
        new_spec = self._spec.updated(**changes)
        rebuilt = self._rebuild(new_spec)
        # Preserve presentational state (theme, caption) across rebuilds.
        # Footnotes are *not* preserved: builders regenerate them based on
        # the current spec (e.g. adding ``.add_p()`` introduces a "Tests:"
        # line). Call ``.add_footnote()`` *after* statistical modifiers to
        # append your own.
        return replace(
            rebuilt,
            theme_name=self.theme_name,
            caption=self.caption,
        )
