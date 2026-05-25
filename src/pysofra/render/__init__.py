"""Renderers — backend-agnostic output of :class:`~pysofra.core.SofraTable`."""

from .docx import DocxRenderer
from .html import HtmlRenderer
from .latex import LatexRenderer
from .markdown import MarkdownRenderer

# PPTX and XLSX renderers are optional — gated on their backends.
PptxRenderer: type | None
XlsxRenderer: type | None
try:
    from .pptx import PptxRenderer
except ImportError:  # pragma: no cover
    PptxRenderer = None

try:
    from .xlsx import XlsxRenderer
except ImportError:  # pragma: no cover
    XlsxRenderer = None

__all__ = [
    "DocxRenderer",
    "HtmlRenderer",
    "LatexRenderer",
    "MarkdownRenderer",
    "PptxRenderer",
    "XlsxRenderer",
]
