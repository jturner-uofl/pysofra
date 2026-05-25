"""Kaplan–Meier curve rendering for tbl_survival SofraTables.

This function does *not* re-extract the KM fits from the table — those
aren't preserved in the rendered structure. Instead it accepts the same
data the user passed to ``tbl_survival`` and fits curves freshly.
"""

from __future__ import annotations

from typing import Any

from ..core.frames import to_pandas
from .inline import InlinePlot, fig_to_svg, render_inline_plot


def km_curve(
    data: Any,
    *,
    time: str,
    event: str,
    by: str | None = None,
    width_in: float = 6.5,
    height_in: float = 4.0,
    ci: bool = True,
    xlabel: str = "Time",
    ylabel: str = "Survival probability",
    palette: list[str] | None = None,
    risk_times: list[float] | tuple[float, ...] | None = None,
) -> InlinePlot:
    """Render KM curves as an :class:`InlinePlot` (SVG + PNG + PDF).

    ``risk_times`` adds a numbers-at-risk table below the curves at the
    listed time points (no table is added when ``risk_times`` is None).
    """
    fig = _build_km_figure(
        data, time=time, event=event, by=by,
        width_in=width_in, height_in=height_in, ci=ci,
        xlabel=xlabel, ylabel=ylabel, palette=palette,
        risk_times=risk_times,
    )
    plot = render_inline_plot(fig, width_in=width_in, height_in=height_in)
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except ImportError:  # pragma: no cover
        pass
    return plot


def km_curve_svg(
    data: Any,
    *,
    time: str,
    event: str,
    by: str | None = None,
    width_in: float = 6.5,
    height_in: float = 4.0,
    ci: bool = True,
    xlabel: str = "Time",
    ylabel: str = "Survival probability",
    palette: list[str] | None = None,
    risk_times: list[float] | tuple[float, ...] | None = None,
) -> str:
    """Render Kaplan–Meier curves to an inline SVG string."""
    fig = _build_km_figure(
        data, time=time, event=event, by=by,
        width_in=width_in, height_in=height_in, ci=ci,
        xlabel=xlabel, ylabel=ylabel, palette=palette,
        risk_times=risk_times,
    )
    svg = fig_to_svg(fig)
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except ImportError:  # pragma: no cover
        pass
    return svg


def _build_km_figure(
    data: Any,
    *,
    time: str,
    event: str,
    by: str | None,
    width_in: float,
    height_in: float,
    ci: bool,
    xlabel: str,
    ylabel: str,
    palette: list[str] | None,
    risk_times: list[float] | tuple[float, ...] | None = None,
) -> Any:
    try:
        from ._backend import use_headless_backend
        use_headless_backend()
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "KM curves require matplotlib. Install with "
            "`pip install matplotlib`."
        ) from e
    try:
        from lifelines import KaplanMeierFitter
    except ImportError as e:  # pragma: no cover
        raise ImportError("KM curves require lifelines.") from e

    df = to_pandas(data)
    df_groups = (
        [("Overall", df)]
        if by is None
        else list(df.groupby(by, observed=True))
    )

    palette = palette or [
        "#0b3d91", "#c1272d", "#198754", "#ffc107", "#6f42c1", "#fd7e14",
    ]

    # Risk-table augment: shrink the curve axes and add a small axes
    # underneath listing N at risk per group at each requested time.
    n_groups_pred = len(df_groups)

    if risk_times:
        from matplotlib.gridspec import GridSpec

        # Each column needs roughly one digit's worth of breathing room
        # so adjacent two-/three-digit at-risk counts never touch.
        min_w = max(width_in, 1.5 + 0.55 * len(risk_times))
        # Each group row in the risk table needs ~0.28 in plus a small
        # cushion for the heading.
        risk_h_in = 0.28 * n_groups_pred + 0.55
        total_h = height_in + risk_h_in + 0.25
        fig = plt.figure(figsize=(min_w, total_h))
        gs = GridSpec(
            2, 1, figure=fig,
            height_ratios=[height_in, risk_h_in],
            hspace=0.32,
        )
        ax = fig.add_subplot(gs[0])
        ax_risk = fig.add_subplot(gs[1], sharex=ax)
    else:
        fig, ax = plt.subplots(figsize=(width_in, height_in))
        ax_risk = None

    fits: dict[str, Any] = {}
    for i, (label, sub) in enumerate(df_groups):
        sub = sub.dropna(subset=[time, event])
        if sub.empty:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub[time], sub[event], label=str(label))
        fits[str(label)] = kmf
        color = palette[i % len(palette)]
        kmf.plot_survival_function(ax=ax, ci_show=ci, color=color)

    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if by is None:
        legend = ax.get_legend()
        if legend is not None:
            legend.set_visible(False)

    if ax_risk is not None and risk_times:
        # Clamp the x-window to the range the user asked about. Without
        # this, a long-tailed survival curve (e.g. one censored case at
        # t=110 when risk_times stop at 30) inflates xlim and crams every
        # risk-table column into the leftmost slice of the figure.
        rt_min = float(min(risk_times))
        rt_max = float(max(risk_times))
        span = max(rt_max - rt_min, 1e-9)
        # Generous left pad so the first risk-table number sits *inside*
        # the axes (otherwise a centered "165" at t=0 punches through the
        # y-axis into the "Placebo"/"Treatment" label column).
        ax.set_xlim(rt_min - 0.08 * span, rt_max + 0.04 * span)

        # Clear the curve's bottom axis — the risk table owns the x-axis labels.
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)

        # Risk-table axis. We want a *grid of numbers*: rows = groups, cols
        # = time points. Numbers sit at (t_pt, group_index) in data
        # coordinates so they line up vertically with the curve.
        n_groups = len(fits)
        ax_risk.set_xlim(ax.get_xlim())
        ax_risk.set_xticks(list(risk_times))
        # Force the tick labels to render as integers / floats with no
        # trailing zeroes so they don't collide.
        ax_risk.set_xticklabels([
            f"{int(t)}" if float(t).is_integer() else f"{t:g}"
            for t in risk_times
        ], fontsize=9)
        ax_risk.tick_params(axis="x", length=4, pad=2)

        # Group labels on y, ordered top-to-bottom to mirror the curve legend.
        ax_risk.set_yticks(range(n_groups))
        ax_risk.set_yticklabels(list(fits.keys()), fontsize=9)
        ax_risk.set_ylim(-0.5, n_groups - 0.5)
        ax_risk.invert_yaxis()
        ax_risk.tick_params(axis="y", length=0, pad=4)
        for spine in ("top", "right", "left"):
            ax_risk.spines[spine].set_visible(False)

        # Auto-scale font down for very dense risk tables.
        font_size = 9 if len(risk_times) <= 6 else 8

        # Render numbers at each (time, group) cell.
        for i, (_name, kmf) in enumerate(fits.items()):
            for t_pt in risk_times:
                n_at_risk = _n_at_risk(kmf, float(t_pt))
                ax_risk.text(
                    t_pt, i, f"{n_at_risk}",
                    ha="center", va="center", fontsize=font_size,
                )

        ax_risk.set_xlabel(xlabel)
        # "Number at risk" heading sits above the *numbers* portion (centered
        # over the columns), using axes coordinates so it never collides with
        # the curve plot or the values themselves.
        ax_risk.text(
            0.5, 1.20,
            "Number at risk",
            transform=ax_risk.transAxes,
            fontsize=9, fontweight="bold",
            ha="center", va="bottom",
        )
    else:
        ax.set_xlabel(xlabel)

    return fig


def _n_at_risk(kmf: Any, t: float) -> int:
    """Number of individuals at risk *just before* time ``t``.

    See :func:`pysofra.models.survival._n_at_risk` for the convention
    rationale — this is the same implementation, duplicated to avoid
    a cross-module import in the matplotlib hot path.
    """
    try:
        tbl = kmf.event_table
        idx = tbl.index[tbl.index >= t]
        if len(idx) == 0:
            return 0
        first_t = idx.min()
        return int(tbl.loc[first_t, "at_risk"])
    except Exception:  # pragma: no cover
        return 0
