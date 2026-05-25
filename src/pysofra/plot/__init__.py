"""Matplotlib-backed plot generation for SofraTables.

Plot helpers return :class:`InlinePlot` objects carrying SVG, PNG, and
PDF serialisations of the same matplotlib figure so every renderer
(HTML, DOCX, PPTX, LaTeX) can embed the plot consistently.

* :func:`forest_plot` — point estimates + CIs from a regression table.
* :func:`km_curve` — Kaplan–Meier survival curves.

Both depend on ``matplotlib``, which is an optional dependency.
"""

from .forest import forest_plot, forest_plot_svg
from .inline import InlinePlot
from .km import km_curve, km_curve_svg

__all__ = [
    "InlinePlot",
    "forest_plot",
    "forest_plot_svg",
    "km_curve",
    "km_curve_svg",
]
