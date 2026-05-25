"""Forest plot rendering for regression SofraTables."""

from __future__ import annotations

import math
from typing import Any

from ..core.table import SofraTable
from .inline import InlinePlot, fig_to_svg, render_inline_plot


def forest_plot(
    table: SofraTable,
    *,
    log_x: bool = True,
    null_line: float = 1.0,
    width_in: float = 6.5,
    height_per_row_in: float = 0.42,
    color: str = "#0b3d91",
) -> InlinePlot:
    """Render a forest plot as an :class:`InlinePlot` (SVG + PNG + PDF).

    Use this when you want the plot embedded across multiple renderers
    (HTML, DOCX, PPTX, LaTeX). For the HTML-only string form, use
    :func:`forest_plot_svg`.
    """
    fig, height = _build_forest_figure(
        table, log_x=log_x, null_line=null_line,
        width_in=width_in, height_per_row_in=height_per_row_in, color=color,
    )
    plot = render_inline_plot(fig, width_in=width_in, height_in=height)
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except ImportError:  # pragma: no cover
        pass
    return plot


def forest_plot_svg(
    table: SofraTable,
    *,
    log_x: bool = True,
    null_line: float = 1.0,
    width_in: float = 6.5,
    height_per_row_in: float = 0.42,
    color: str = "#0b3d91",
) -> str:
    """Render a forest plot for a regression :class:`SofraTable`.

    Reads point estimates and CI bounds from the body rows: it looks for
    a numeric cell of kind ``numeric`` (the point estimate) followed by
    a cell of kind ``ci`` carrying ``(lo, hi)`` tuples — exactly the
    layout produced by :func:`pysofra.tbl_regression`.

    Parameters
    ----------
    table
        A SofraTable produced by ``tbl_regression`` (single- or multi-model).
    log_x
        Plot on a log-scale x-axis. Default ``True`` because the natural
        scale for ORs / HRs / IRRs is multiplicative.
    null_line
        x-coordinate of the null reference (1 for exponentiated, 0 for raw).
    width_in
        Figure width in inches.
    height_per_row_in
        Vertical space per coefficient row.
    color
        Hex string for the point + CI segments.
    """
    fig, _ = _build_forest_figure(
        table, log_x=log_x, null_line=null_line,
        width_in=width_in, height_per_row_in=height_per_row_in, color=color,
    )
    svg = fig_to_svg(fig)
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except ImportError:  # pragma: no cover
        pass
    return svg


def _build_forest_figure(
    table: SofraTable,
    *,
    log_x: bool,
    null_line: float,
    width_in: float,
    height_per_row_in: float,
    color: str,
) -> tuple[Any, float]:
    try:
        from ._backend import use_headless_backend
        use_headless_backend()
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Forest plots require matplotlib. Install with "
            "`pip install matplotlib`."
        ) from e

    points: list[tuple[str, float, float, float]] = []
    for r in table.rows:
        label = r.cells[0].text
        est = next((c for c in r.cells if c.kind == "numeric"
                    and isinstance(c.value, (int, float))), None)
        ci = next((c for c in r.cells if c.kind == "ci"
                   and isinstance(c.value, tuple) and len(c.value) == 2), None)
        if est is None or ci is None:
            continue
        lo, hi = ci.value
        if any(_isnan(x) for x in (est.value, lo, hi)):
            continue
        points.append((label, float(est.value), float(lo), float(hi)))

    if not points:
        raise ValueError("No (estimate, CI) pairs found in table; "
                         "is this a regression table?")

    n = len(points)
    height = max(2.0, height_per_row_in * n + 1.0)
    fig, ax = plt.subplots(figsize=(width_in, height))

    labels = [p[0] for p in points]
    estimates = [p[1] for p in points]
    lows = [p[2] for p in points]
    highs = [p[3] for p in points]

    ys = list(range(n, 0, -1))
    ax.errorbar(
        estimates, ys,
        xerr=[[e - lo for e, lo in zip(estimates, lows, strict=True)],
              [hi - e for e, hi in zip(estimates, highs, strict=True)]],
        fmt="s", color=color, ecolor=color,
        elinewidth=1.5, capsize=4, markersize=7,
    )

    ax.axvline(null_line, color="#888", linewidth=1, linestyle="--", zorder=0)
    if log_x:
        ax.set_xscale("log")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_ylim(0.5, n + 0.5)
    ax.tick_params(axis="y", left=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(table.headers[0].cells[1].text if table.headers else "Estimate")

    return fig, height


def _isnan(x: Any) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return False
