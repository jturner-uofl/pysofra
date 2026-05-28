"""PySofra — the missing statistical reporting layer for Python.

PySofra transforms datasets and statistical model outputs into
publication-ready tables across HTML, Markdown, DOCX, LaTeX, PPTX, XLSX,
and PNG. The same underlying :class:`SofraTable` object renders
beautifully in Jupyter and exports identically to disk.

Quick start
-----------

>>> import pandas as pd
>>> import pysofra as ps
>>> df = pd.DataFrame({
...     "age": [55, 62, 47, 68, 51],
...     "sex": ["F", "M", "F", "M", "F"],
...     "arm": ["A", "A", "B", "B", "A"],
... })
>>> tbl = (
...     ps.tbl_one(df, by="arm")
...       .add_p()
...       .add_smd()
...       .theme("clinical")
... )
>>> _ = tbl.to_html()
"""

from __future__ import annotations

from .core.compose import tbl_merge, tbl_stack
from .core.schema import CellPart
from .core.table import SofraTable
from .models.pool import pool
from .models.regression import tbl_regression
from .models.survival import tbl_survival
from .models.uvregression import tbl_uvregression
from .summary.calibrate import design_effect, post_stratify, rake
from .summary.design import SurveyDesign
from .summary.effect_size import (
    auto_effect_size,
    cohen_d,
    cramers_v,
    eta_squared,
    hedges_g,
    omega_squared,
    phi_coefficient,
)
from .summary.tbl_cross import tbl_cross
from .summary.tbl_one import tbl_one
from .summary.tbl_summary import tbl_summary
from .summary.tests import available_tests
from .themes.registry import available_themes, register_theme

__version__ = "0.1.0a11"

__all__ = [
    "CellPart",
    "SofraTable",
    "SurveyDesign",
    "__version__",
    "auto_effect_size",
    "available_tests",
    "available_themes",
    "cohen_d",
    "cramers_v",
    "design_effect",
    "eta_squared",
    "hedges_g",
    "omega_squared",
    "phi_coefficient",
    "pool",
    "post_stratify",
    "rake",
    "register_theme",
    "tbl_cross",
    "tbl_merge",
    "tbl_one",
    "tbl_regression",
    "tbl_stack",
    "tbl_summary",
    "tbl_survival",
    "tbl_uvregression",
]
