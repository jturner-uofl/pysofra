"""Kaplan–Meier summary tables via :func:`tbl_survival`.

Produces a publication-ready survival summary with:

* N total / N events / N censored, per group
* Median survival with confidence interval
* Survival probability at user-specified time points (with N at risk)
* Log-rank p-value across groups (when ``by=`` is provided)

Requires the optional ``lifelines`` dependency. Install with
``pip install lifelines`` or as part of a survival workflow extras.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..core.format import fmt_number, fmt_p_value
from ..core.frames import to_pandas
from ..core.schema import Cell, HeaderCell, HeaderRow, Row, make_cell
from ..core.table import SofraTable, TableSpec


def tbl_survival(
    data: Any,
    *,
    time: str,
    event: str,
    by: str | None = None,
    times: list[float] | tuple[float, ...] | None = None,
    times_label: str | None = None,
    conf_level: float = 0.95,
    digits: int = 2,
    pct_digits: int = 1,
    labels: dict[str, str] | None = None,
    show_logrank: bool = True,
    weights: str | None = None,
) -> SofraTable:
    """Build a Kaplan–Meier summary table.

    Parameters
    ----------
    data
        Source dataframe (pandas or polars).
    time
        Column carrying follow-up time.
    event
        Column carrying the event indicator (1 = event, 0 = censored).
    by
        Optional stratification column. Without it, a single
        ``"Overall"`` column is produced.
    times
        Optional list of follow-up times at which to report survival
        probability and N at risk. For example ``[12, 24, 36]`` for
        1/2/3-year survival in a months-scaled study.
    times_label
        Unit label appended to each ``times`` header (e.g. ``"months"``
        → ``"S(12 months)"``). Defaults to bare numbers.
    conf_level
        Confidence level for the median survival CI.
    digits
        Decimal places for survival probabilities and median.
    pct_digits
        Decimal places for survival percentages.
    labels
        Optional mapping from group level → display label.
    show_logrank
        Whether to compute and footnote the multi-group log-rank test.
    weights
        Optional column carrying per-row sampling/frequency weights.
        When supplied, the Kaplan–Meier estimator is fit with the
        ``weights=`` kwarg of ``lifelines.KaplanMeierFitter`` (a
        weighted product-limit estimator). N totals / events / censored
        report weighted sums. The log-rank test currently uses
        unweighted ranks regardless — lifelines does not expose a
        weighted log-rank — and a footnote flags this when weights are
        active.
    """
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import multivariate_logrank_test
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "tbl_survival requires lifelines. Install with `pip install lifelines`."
        ) from e

    if not (0.0 < conf_level < 1.0):
        raise ValueError(
            f"conf_level must lie in the open interval (0, 1); "
            f"got {conf_level!r}."
        )
    data = to_pandas(data)
    for col in (time, event):
        if col not in data.columns:
            raise KeyError(f"column {col!r} not in data")

    # Validate time + event content. ``lifelines`` will silently treat
    # negative survival times as zero and any nonzero event value as a
    # death, so input mistakes (e.g. a "censor at last follow-up" column
    # encoded as 0/1/9, or a follow-up time accidentally negated) can
    # produce a misleading survival curve without complaint. Fail loud
    # at the boundary instead.
    time_num = pd.to_numeric(data[time], errors="coerce")
    if (time_num < 0).any():
        n_bad = int((time_num < 0).sum())
        raise ValueError(
            f"column {time!r} contains {n_bad} negative value(s); "
            "survival times must be non-negative."
        )
    event_num = pd.to_numeric(data[event], errors="coerce").dropna()
    bad_events = ~event_num.isin([0, 1])
    if bool(bad_events.any()):
        bad_vals = sorted(event_num[bad_events].unique().tolist())
        raise ValueError(
            f"column {event!r} must contain only 0/1 (or boolean) "
            f"values; got unexpected values: {bad_vals!r}."
        )

    if by is not None and by not in data.columns:
        raise KeyError(f"by column {by!r} not in data")

    labels = dict(labels or {})
    if by is None:
        group_keys: list[Any] = ["Overall"]
        group_masks = {"Overall": pd.Series(True, index=data.index)}
    else:
        by_series = data[by]
        if isinstance(by_series.dtype, pd.CategoricalDtype):
            group_keys = [k for k in by_series.cat.categories if (by_series == k).any()]
        else:
            group_keys = sorted(by_series.dropna().unique(), key=_sort_key)
        group_keys = list(group_keys)
        group_masks = {k: (by_series == k) for k in group_keys}

    # ------------------------------------------------------------------
    # Headers
    # ------------------------------------------------------------------
    header_cells: list[HeaderCell] = [HeaderCell(text="Statistic", align="left")]
    for k in group_keys:
        header_cells.append(HeaderCell(text=str(labels.get(k, k))))
    if show_logrank and by is not None and len(group_keys) > 1:
        header_cells.append(HeaderCell(text="p-value"))

    headers = (HeaderRow(cells=tuple(header_cells)),)

    # ------------------------------------------------------------------
    # KM fits per group
    # ------------------------------------------------------------------
    fits: dict[Any, Any] = {}
    # ``n_*`` are int under unweighted analysis and float (weighted sum)
    # when ``weights=`` is passed. We accept either via ``float`` here
    # because every downstream renderer formats them through ``_fmt_n``
    # which handles both cases.
    n_total: dict[Any, float] = {}
    n_events: dict[Any, float] = {}
    n_censored: dict[Any, float] = {}
    medians: dict[Any, tuple[float | None, float | None, float | None]] = {}

    # Validate weights column (consistent with tbl_one's policy:
    # negative or all-zero weights raise loudly rather than warn-and-
    # drop).
    if weights is not None:
        if weights not in data.columns:
            raise KeyError(f"weights column {weights!r} not in data")
        w_full = pd.to_numeric(data[weights], errors="coerce")
        if (w_full < 0).any():
            raise ValueError(
                f"weights column {weights!r} contains negative value(s); "
                "drop or correct them before calling tbl_survival()."
            )
        if float(w_full.fillna(0.0).sum()) <= 0:
            raise ValueError(
                f"weights column {weights!r} has non-positive total weight."
            )

    for k in group_keys:
        m = group_masks[k]
        cols = [time, event] + ([weights] if weights is not None else [])
        sub = data.loc[m, cols].dropna()
        kmf = KaplanMeierFitter()
        if len(sub) > 0:
            if weights is not None:
                w_arr = sub[weights].to_numpy(dtype=float)
                kmf.fit(sub[time], sub[event],
                        weights=w_arr, alpha=1 - conf_level)
                # Report weighted N counts to match the weighted curve.
                n_total[k] = float(w_arr.sum())
                events_mask = sub[event].to_numpy(dtype=float) > 0
                n_events[k] = float(w_arr[events_mask].sum())
                n_censored[k] = float(n_total[k] - n_events[k])
            else:
                kmf.fit(sub[time], sub[event], alpha=1 - conf_level)
                n_total[k] = int(len(sub))
                n_events[k] = int(sub[event].sum())
                n_censored[k] = int(len(sub) - sub[event].sum())
            fits[k] = kmf
            med = float(kmf.median_survival_time_)
            med_ci = _median_ci(kmf)
            medians[k] = (med, med_ci[0], med_ci[1])
        else:
            fits[k] = None
            n_total[k] = 0
            n_events[k] = 0
            n_censored[k] = 0
            medians[k] = (None, None, None)

    # ------------------------------------------------------------------
    # Log-rank
    # ------------------------------------------------------------------
    logrank_p: float | None = None
    if show_logrank and by is not None and len(group_keys) > 1:
        df = data.dropna(subset=[time, event, by])
        # Suppress only the third-party deprecation warnings emitted by
        # lifelines/pandas during the log-rank call (these are
        # informational and escalate to errors under our strict
        # ``filterwarnings = error`` configuration). Any other
        # exception is a genuine numerical failure and surfaces.
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
            try:
                result = multivariate_logrank_test(df[time], df[by], df[event])
                logrank_p = float(result.p_value)
            except (ValueError, ZeroDivisionError):  # pragma: no cover
                logrank_p = None

    # ------------------------------------------------------------------
    # Body rows
    # ------------------------------------------------------------------
    rows: list[Row] = []
    has_p_col = show_logrank and by is not None and len(group_keys) > 1
    n_groups = len(group_keys)

    def _row_with_blank_p(label_cell: Cell, value_cells: list[Cell]) -> Row:
        cells = [label_cell, *value_cells]
        if has_p_col:
            cells.append(make_cell("", value=None))
        return Row(cells=tuple(cells))

    # N totals — render as integers when unweighted, with one decimal
    # when weighted (so the user can see the sum-of-weights without a
    # 15-digit float spam).
    def _fmt_n(v: float | int) -> str:
        if isinstance(v, int) or float(v).is_integer():
            return f"{int(v):,}"
        return f"{v:,.1f}"

    rows.append(_row_with_blank_p(
        make_cell("N", align="left"),
        [make_cell(_fmt_n(n_total[k]), value=n_total[k],
                   kind="numeric", align="right")
         for k in group_keys],
    ))
    # N events
    rows.append(_row_with_blank_p(
        make_cell("Events", align="left"),
        [make_cell(_fmt_n(n_events[k]), value=n_events[k],
                   kind="numeric", align="right")
         for k in group_keys],
    ))
    # N censored
    rows.append(_row_with_blank_p(
        make_cell("Censored", align="left"),
        [make_cell(_fmt_n(n_censored[k]), value=n_censored[k],
                   kind="numeric", align="right")
         for k in group_keys],
    ))

    # Median survival with CI; the log-rank p attaches to this row.
    median_cells = []
    for k in group_keys:
        med_val, lo, hi = medians[k]
        if med_val is None or np.isnan(med_val):
            text = "—"
        else:
            ci_part = ""
            if lo is not None and hi is not None and not (np.isnan(lo) or np.isnan(hi)):
                ci_part = f" ({fmt_number(lo, digits)}, {fmt_number(hi, digits)})"
            text = f"{fmt_number(med_val, digits)}{ci_part}"
        median_cells.append(make_cell(text, value=med_val, kind="numeric", align="right"))

    median_row_cells = [make_cell(
        f"Median survival ({int(round(conf_level * 100))}% CI)", align="left",
    ), *median_cells]
    if has_p_col:
        median_row_cells.append(make_cell(
            fmt_p_value(logrank_p), value=logrank_p,
            kind="p_value", align="right",
        ))
    rows.append(Row(cells=tuple(median_row_cells)))

    # Survival probability at each fixed time
    if times:
        for t in times:
            row_label = _format_time_label(t, times_label)
            cells: list[Cell] = [make_cell(row_label, align="left")]
            for k in group_keys:
                kmf = fits[k]
                if kmf is None:
                    cells.append(make_cell("—", value=None,
                                           kind="numeric", align="right"))
                    continue
                surv = _survival_at(kmf, t)
                n_at_risk = _n_at_risk(kmf, t)
                if surv is None:
                    cells.append(make_cell("—", value=None,
                                           kind="numeric", align="right"))
                else:
                    pct = surv * 100.0
                    text = f"{pct:.{pct_digits}f}% (n={n_at_risk})"
                    cells.append(make_cell(text, value=surv,
                                           kind="numeric", align="right"))
            if has_p_col:
                cells.append(make_cell("", value=None))
            rows.append(Row(cells=tuple(cells)))

    # ------------------------------------------------------------------
    # Footnotes
    # ------------------------------------------------------------------
    footnotes: list[str] = []
    if times:
        footnotes.append(
            "Survival probability shown with N at risk at each time point."
        )
    footnotes.append(
        f"Median survival reported with {int(round(conf_level * 100))}% confidence interval."
    )
    if weights is not None:
        footnotes.append(
            f"Kaplan–Meier curves are weighted by {weights!r}; N, events, "
            "and censored are reported as weighted sums."
        )
    if has_p_col and logrank_p is not None:
        if weights is not None:
            footnotes.append(
                "p-value: multivariate log-rank test, computed UNWEIGHTED "
                "(lifelines does not expose a weighted log-rank). For a "
                "design-adjusted survival comparison call out to R "
                "survey::svykm directly."
            )
        else:
            footnotes.append("p-value: multivariate log-rank test across groups.")

    del n_groups
    spec = TableSpec(
        builder="tbl_survival",
        options={
            "time": time,
            "event": event,
            "by": by,
            "times": tuple(times) if times else (),
            "conf_level": conf_level,
            "digits": digits,
            "pct_digits": pct_digits,
        },
    )

    table = SofraTable(
        rows=tuple(rows),
        headers=headers,
        footnotes=tuple(footnotes),
        metadata={"builder": "tbl_survival",
                  "logrank_p": logrank_p,
                  "n_groups": len(group_keys),
                  # Closure used by .with_km_plot to fit curves with the
                  # *same* data the table was computed from.
                  "_km_source": {
                      "data": data,
                      "time": time,
                      "event": event,
                      "by": by,
                  }},
        _spec=spec,
    )
    return table


def attach_km_plot(
    table: SofraTable,
    *,
    position: str = "above",
    **plot_kwargs: Any,
) -> SofraTable:
    """Attach a Kaplan–Meier curve to a :func:`tbl_survival` result.

    Reads the original time / event / by columns out of the table
    metadata and refits the KM curves with ``lifelines``. The attached
    plot carries SVG, PNG, and PDF serialisations so it embeds across
    every PySofra render backend.
    """
    from dataclasses import replace as dc_replace

    src = table.metadata.get("_km_source") if table.metadata else None
    if not src:
        raise ValueError(
            "attach_km_plot expects a SofraTable produced by tbl_survival."
        )
    if position not in ("above", "below"):
        raise ValueError("position must be 'above' or 'below'")
    from ..plot.km import km_curve

    plot = km_curve(
        src["data"], time=src["time"], event=src["event"], by=src["by"],
        **plot_kwargs,
    )
    return dc_replace(
        table,
        inline_svg=plot.svg,
        inline_svg_position=position,
        inline_plot=plot,
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _median_ci(kmf: Any) -> tuple[float | None, float | None]:
    """Try to extract a CI for the median survival time from a lifelines KMF.

    The confidence level is fixed by the ``alpha=`` passed at ``kmf.fit``
    time — lifelines bakes the CI into ``kmf.confidence_interval_`` at
    fit time and there is no way to re-derive it post-hoc without
    refitting. Callers must therefore call ``kmf.fit(..., alpha=1−L)``
    upstream to get an *L*-confidence median CI here.
    """
    try:
        from lifelines.utils import median_survival_times

        med_df = median_survival_times(kmf.confidence_interval_)
        # Returns a DataFrame with columns like 'KM_estimate_lower_X.XX'.
        row = med_df.iloc[0]
        if len(row) >= 2:
            return float(row.iloc[0]), float(row.iloc[1])
    except Exception:  # pragma: no cover
        pass
    return None, None


def _survival_at(kmf: Any, t: float) -> float | None:
    """Return ``S(t)`` from a fitted KaplanMeierFitter."""
    try:
        sf = kmf.survival_function_at_times(t)
        val = float(sf.iloc[0])
        if np.isnan(val):
            return None
        return val
    except Exception:  # pragma: no cover
        return None


def _n_at_risk(kmf: Any, t: float) -> int:
    """Return the number of individuals at risk *just before* ``t``.

    Convention: a person is at risk at time ``t`` if they have not yet
    had an event or been censored by ``t``. Equivalently, given
    ``kmf.event_table`` (indexed by event times with an ``at_risk``
    column whose value at row ``t_i`` is the at-risk count just before
    ``t_i``), the at-risk count just before query time ``t`` equals
    the ``at_risk`` value at the first event-table row with
    ``time >= t``. If no such row exists (``t`` is beyond the last
    recorded event), the at-risk pool is empty.
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


def _format_time_label(t: float, unit: str | None) -> str:
    if unit:
        return f"S({t:g} {unit})"
    return f"S(t = {t:g})"


def _sort_key(x: Any) -> tuple[int, Any]:
    if isinstance(x, bool):
        return (0, int(x))
    if isinstance(x, (int, float)):
        return (0, float(x))
    if isinstance(x, str):
        return (1, x)
    return (2, repr(x))
