"""Model-output table builders."""

from .regression import tbl_regression
from .survival import tbl_survival

__all__ = ["tbl_regression", "tbl_survival"]
