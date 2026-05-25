"""Core types: :class:`SofraTable` and its schema."""

from .schema import Cell, HeaderCell, HeaderRow, Row, SpanningHeader
from .table import SofraTable, TableSpec

__all__ = [
    "Cell",
    "HeaderCell",
    "HeaderRow",
    "Row",
    "SofraTable",
    "SpanningHeader",
    "TableSpec",
]
