"""Quarto-fenced-block export for SofraTable.

Quarto (`https://quarto.org`) is the reproducible-research authoring
framework that subsumed RMarkdown and adds first-class Python/Jupyter
support. A Quarto document (``.qmd``) can interleave prose, code, and
figures, and renders to HTML / PDF / DOCX / EPUB from a single source.

To embed a PySofra table in a Quarto document, the user wants:

1. A *fenced block* that Quarto recognises as raw passthrough content
   for the target format (``:::{=html} … :::`` for HTML output,
   ``:::{=latex} … :::`` for PDF).
2. Optional Quarto cross-reference machinery — a ``#tbl-XXX`` label
   and a caption — so ``@tbl-XXX`` cites work in the prose.

The output of :func:`to_quarto` is a Markdown-string ready to paste
into a ``.qmd`` file, or to write out programmatically and ``{{<
include >}}`` into a chapter.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..core.table import SofraTable


_VALID_FORMATS = ("html", "latex")


def to_quarto(
    table: SofraTable,
    *,
    format: str = "html",
    label: str | None = None,
    caption: str | None = None,
) -> str:
    """Emit a Quarto fenced block for ``table``.

    Parameters
    ----------
    table
        The SofraTable to embed.
    format
        Quarto's pass-through target. ``"html"`` for HTML / EPUB
        targets; ``"latex"`` for PDF targets.
    label
        Optional Quarto cross-reference label. Must begin with
        ``"tbl-"`` if you want ``@label`` cross-references to
        resolve. When ``None`` no label is emitted.
    caption
        Optional caption text. Quarto associates the caption with
        the labelled block when both are present.

    Returns
    -------
    str
        A Markdown-source fragment of the form::

            ::: {#tbl-foo}

            ::: {=html}
            <table>...</table>
            :::

            Baseline characteristics by arm.

            :::
    """
    fmt = format.lower()
    if fmt not in _VALID_FORMATS:
        raise ValueError(
            f"format must be one of {_VALID_FORMATS!r}; got {format!r}"
        )
    if label is not None and not isinstance(label, str):
        raise TypeError(f"label must be str or None; got {type(label)!r}")
    if label is not None and not label.startswith("tbl-"):
        # Quarto's cross-reference grammar requires the `tbl-` prefix
        # for table refs; warn rather than raise to allow ad-hoc labels.
        import warnings
        warnings.warn(
            f"Quarto convention: table labels should start with "
            f"'tbl-' so @{label} resolves as a table cross-reference; "
            f"got {label!r}. Emitting anyway.",
            UserWarning,
            stacklevel=2,
        )

    body = table.to_html() if fmt == "html" else table.to_latex()

    pass_through = f"::: {{={fmt}}}\n{body}\n:::"

    if label is None and caption is None:
        return pass_through

    # Wrap in a label/caption block.
    inner_lines = ["::: {#" + label + "}"] if label else ["::: {.tbl}"]
    inner_lines.append("")
    inner_lines.append(pass_through)
    if caption is not None:
        inner_lines.append("")
        inner_lines.append(caption)
    inner_lines.append("")
    inner_lines.append(":::")
    return "\n".join(inner_lines)
