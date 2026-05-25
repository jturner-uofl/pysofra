"""Base renderer interface.

All renderers consume a :class:`~pysofra.core.SofraTable` and produce output
in their target format. Concrete renderers live in sibling modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from ..core.table import SofraTable

T = TypeVar("T")


class Renderer(ABC, Generic[T]):
    """Abstract base for all renderers."""

    @abstractmethod
    def render(self, table: SofraTable) -> T:  # pragma: no cover — interface
        ...
