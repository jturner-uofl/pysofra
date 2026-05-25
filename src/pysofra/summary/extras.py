"""Extra modifiers — joint Type-III tests, between-group differences,
descriptive confidence intervals, formatter overrides.

These are smaller add-ons to the core ``tbl_one`` / ``tbl_summary``
output, modelled on the corresponding ``gtsummary`` functions
(``add_global_p``, ``add_difference``, ``add_ci``, ``estimate_fun=``,
``pvalue_fun=``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..core.format import fmt_number
from ..core.schema import Cell, HeaderCell, HeaderRow, Row, make_cell
from ..core.table import SofraTable


def _weights_col_from_spec(table: SofraTable) -> str | None:
    """Return the name of the frequency-weights column attached to this
    table's tbl_one / tbl_summary spec, or ``None`` if unweighted.

    Centralised here so every modifier in this module honours the same
    weights= the original ``tbl_one(..., weights='wt')`` call did.
    """
    spec = table._spec
    if spec is None:
        return None
    w = spec.options.get("weights") if hasattr(spec, "options") else None
    return str(w) if w else None


def _weighted_mean_var_kish(x: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """Frequency-weighted mean, weighted variance (divisor Σw − 1), and
    Kish effective sample size ``n_eff = (Σw)² / Σw²``.

    The Kish ``n_eff`` is the standard plug-in for the SE of a weighted
    mean (``SE = σ_w / sqrt(n_eff)``); this matches R ``survey::svymean``
    to first order on unstratified weighted designs and is what the
    ``DescrStatsW`` class in statsmodels uses as ``nobs`` for inference.
    """
    sw = float(w.sum())
    sw2 = float((w * w).sum())
    if sw <= 1.0 or sw2 <= 0.0:
        return float("nan"), float("nan"), 0.0
    mean = float((w * x).sum() / sw)
    var = float((w * (x - mean) ** 2).sum() / (sw - 1.0))
    n_eff = (sw * sw) / sw2
    return mean, var, n_eff

# ----------------------------------------------------------------------
# add_global_p — Type-III joint test per categorical variable
# ----------------------------------------------------------------------

def add_significance_stars(
    table: SofraTable,
    *,
    thresholds: tuple[tuple[float, str], ...] = (
        (0.001, "***"),
        (0.01, "**"),
        (0.05, "*"),
    ),
) -> SofraTable:
    """Append a ``stars`` column with ``*** / ** / *`` significance markers.

    ``thresholds`` is a tuple of ``(cutoff, marker)`` pairs sorted from
    smallest to largest cutoff. A p-value is marked with the first
    marker whose cutoff it falls below (matching standard journal
    convention).
    """
    new_headers = _insert_after_pvalue_header(table.headers, "")
    new_rows: list[Row] = []
    for r in table.rows:
        p_cell = next(
            (c for c in r.cells
             if c.kind == "p_value" and isinstance(c.value, (int, float))),
            None,
        )
        marker = ""
        if p_cell is not None and p_cell.value is not None:
            p = float(p_cell.value)
            if not _isnan(p):
                for cutoff, m in thresholds:
                    if p < cutoff:
                        marker = m
                        break
        new_rows.append(_insert_after_pvalue_cell(r, marker, value=None))

    # Drop the placeholder header label — significance stars don't need one.
    cleaned_headers: list[HeaderRow] = []
    for hr in new_headers:
        cleaned_headers.append(hr)
    return replace(table, headers=tuple(cleaned_headers), rows=tuple(new_rows))


def add_n(table: SofraTable) -> SofraTable:
    """Append a per-row ``N`` column with the non-missing sample size.

    Reads the rebuild closure to recover the source data, then counts
    non-missing observations per variable. For categorical rows the
    column shows the variable's overall N (not per-level N).
    """
    if table._spec is None or table._rebuild is None:
        raise ValueError(
            "add_n needs access to the source data — only tables built "
            "directly by tbl_one / tbl_summary qualify."
        )
    data = _data_from_rebuild(table._rebuild)
    if data is None:
        raise ValueError("Could not recover source data from table closure.")

    opts = table._spec.options
    variables = opts["variables"]
    kinds = opts["kinds"]

    n_for: dict[str, int] = {}
    for v in variables:
        n_for[v] = int(data[v].notna().sum())

    new_headers = _insert_after_label_header(table.headers, "N")
    new_rows: list[Row] = []
    for r in table.rows:
        label = r.cells[0].text
        var = _find_variable_for_row(label, variables, kinds, labels=opts.get("labels"))
        text = f"{n_for[var]:,}" if var is not None and var in n_for else ""
        new_rows.append(_insert_after_label_cell(
            r, text, value=n_for.get(var) if var else None,
        ))
    return replace(table, headers=new_headers, rows=tuple(new_rows))


def add_stat_label(table: SofraTable) -> SofraTable:
    """Append a ``Statistic`` column describing each row's summary form.

    Continuous (non-nonnormal) rows display "Mean (SD)"; ``nonnormal``
    rows display "Median (Q1, Q3)"; categorical rows display "n (%)".
    """
    if table._spec is None or table._rebuild is None:
        raise ValueError(
            "add_stat_label needs a tbl_one / tbl_summary source table."
        )
    opts = table._spec.options
    variables = opts["variables"]
    kinds = opts["kinds"]
    nonnormal = set(opts.get("nonnormal", set()))

    label_for: dict[str, str] = {}
    for v in variables:
        if kinds[v] == "continuous":
            label_for[v] = (
                "Median (Q1, Q3)" if v in nonnormal else "Mean (SD)"
            )
        else:
            label_for[v] = "n (%)"

    new_headers = _insert_after_label_header(table.headers, "Statistic")
    new_rows: list[Row] = []
    for r in table.rows:
        label = r.cells[0].text
        var = _find_variable_for_row(label, variables, kinds, labels=opts.get("labels"))
        text = label_for.get(var, "") if var else ""
        new_rows.append(_insert_after_label_cell(r, text, value=None))
    return replace(table, headers=new_headers, rows=tuple(new_rows))


def color_scale_if(
    table: SofraTable,
    *,
    column: int,
    palette: tuple[str, str, str] = ("#fff5f0", "#fcae91", "#cb181d"),
    skip_blank: bool = True,
) -> SofraTable:
    """Heatmap-style cell colouring for a numeric column (HTML only).

    Walks the body rows, finds the cell at ``column``, and assigns a
    background colour interpolated across the three-stop ``palette``
    according to the cell's numeric value. Non-numeric cells are left
    untouched (or skipped when ``skip_blank=True``).

    Renderers other than HTML ignore the colour (DOCX/XLSX could honour
    it via per-cell ``style``; that's left to a future round).
    """
    # Pass 1: collect numeric values.
    vals: list[tuple[int, float]] = []
    for i, r in enumerate(table.rows):
        if column >= len(r.cells):
            continue
        v = r.cells[column].value
        if isinstance(v, (int, float)) and not _isnan(v):
            vals.append((i, float(v)))

    if not vals:
        return table  # nothing to colour
    lo = min(v for _, v in vals)
    hi = max(v for _, v in vals)
    span = hi - lo if hi > lo else 1.0
    mid_color = palette[1]
    lo_color, hi_color = palette[0], palette[2]

    def interp(value: float) -> str:
        t = (value - lo) / span
        # Two-stop: lo→mid for t<0.5, mid→hi for t>=0.5.
        if t < 0.5:
            return _mix_hex(lo_color, mid_color, t / 0.5)
        return _mix_hex(mid_color, hi_color, (t - 0.5) / 0.5)

    new_rows: list[Row] = []
    val_dict = dict(vals)
    for i, r in enumerate(table.rows):
        if i not in val_dict:
            new_rows.append(r)
            continue
        cells = list(r.cells)
        c = cells[column]
        color = interp(val_dict[i])
        style = dict(c.style or {})
        style["html"] = (style.get("html", "") + f"background:{color};").strip(";")
        cells[column] = replace(c, style=style)
        new_rows.append(replace(r, cells=tuple(cells)))
    return replace(table, rows=tuple(new_rows))


def _mix_hex(a: str, b: str, t: float) -> str:
    """Linear-interpolate two ``#rrggbb`` colours at parameter ``t``."""
    t = max(0.0, min(1.0, t))
    a = a.lstrip("#")
    b = b.lstrip("#")
    ar, ag, ab = int(a[0:2], 16), int(a[2:4], 16), int(a[4:6], 16)
    br, bg, bb = int(b[0:2], 16), int(b[2:4], 16), int(b[4:6], 16)
    r = int(round(ar + (br - ar) * t))
    g = int(round(ag + (bg - ag) * t))
    bl = int(round(ab + (bb - ab) * t))
    return f"#{r:02x}{g:02x}{bl:02x}"


def _insert_after_label_header(
    headers: tuple[HeaderRow, ...], label: str,
) -> tuple[HeaderRow, ...]:
    """Insert a header cell right after the first column (the label column)."""
    new_headers: list[HeaderRow] = []
    for hr in headers:
        new_cells = list(hr.cells)
        new_cells.insert(1, HeaderCell(text=label))
        new_headers.append(HeaderRow(cells=tuple(new_cells)))
    return tuple(new_headers)


def _insert_after_label_cell(
    row: Row, text: str, *, value: Any,
) -> Row:
    new_cells = list(row.cells)
    new_cells.insert(1, make_cell(text, value=value, align="right"))
    return replace(row, cells=tuple(new_cells))


def add_global_p(table: SofraTable) -> SofraTable:
    """Add a joint Type-III p-value column to a :func:`tbl_regression` table.

    For each multi-level categorical predictor, the rows share a single
    joint Wald-F p-value computed via ``model.f_test()`` on the
    contrast matrix that zeroes out every level simultaneously.
    Single-level coefficients receive their existing p-value
    duplicated.

    Raises
    ------
    NotImplementedError
        If the table was not built by :func:`tbl_regression` (i.e. no
        fitted ``model`` is attached). Joint Type-III tests on a
        :func:`tbl_one` table require re-fitting per-variable
        regressions on the source data — that path is not yet
        implemented; raising avoids silently emitting a column of
        em-dashes that would mislead a reader of the published table.

    Notes
    -----
    The contrast matrix is built from coefficient stems detected by
    :func:`_coef_stem` (handles statsmodels'
    ``C(race)[T.B]`` / ``arm[T.Treatment]`` markers).
    """
    model = (table.metadata or {}).get("model")
    if model is None or not hasattr(model, "f_test"):
        raise NotImplementedError(
            "add_global_p currently supports tbl_regression tables only. "
            "For a tbl_one / tbl_summary table, joint Type-III tests would "
            "require re-fitting per-variable regressions on the source "
            "data — that path is not implemented yet. Track the issue "
            "before using `add_global_p` on a non-regression table."
        )

    new_headers = _insert_after_pvalue_header(table.headers, "global p")

    # Group coefficient rows by their stem (the part before the level
    # marker that statsmodels uses, e.g. ``C(race)[T.B]``).
    stems: dict[str, list[int]] = {}
    row_label_to_stem: dict[int, str] = {}
    for i, r in enumerate(table.rows):
        label = r.cells[0].text
        stem = _coef_stem(label)
        stems.setdefault(stem, []).append(i)
        row_label_to_stem[i] = stem

    # Compute one joint p-value per stem.
    joint_p: dict[str, float | None] = {}
    params = getattr(model, "params", None)
    param_names = (
        list(params.index)
        if params is not None and hasattr(params, "index")
        else []
    )
    for stem, _idxs in stems.items():
        # The coefficient names contributing to this stem are every param
        # whose stem matches.
        contributing = [p for p in param_names if _coef_stem(p) == stem]
        if not contributing:
            joint_p[stem] = None
            continue
        # Build a constraint string "c1 = 0, c2 = 0, ..."
        constraint = ", ".join(f"{c} = 0" for c in contributing)
        try:
            res = model.f_test(constraint)
            joint_p[stem] = float(res.pvalue)
        except Exception:  # pragma: no cover — exotic models / singular cov
            joint_p[stem] = None

    from ..core.format import fmt_p_value

    new_rows = []
    for i, r in enumerate(table.rows):
        stem = row_label_to_stem[i]
        p = joint_p.get(stem)
        new_rows.append(_insert_after_pvalue_cell(
            r, fmt_p_value(p) if p is not None else "—", value=p,
        ))
    return replace(table, headers=new_headers, rows=tuple(new_rows))


def _coef_stem(name: str) -> str:
    """Strip statsmodels-style level markers from a coefficient name.

    ``C(race)[T.B]`` → ``C(race)``
    ``arm[T.Treatment]`` → ``arm``
    ``age`` → ``age``
    """
    for marker in ("[T.", "[", "_T_"):
        if marker in name:
            return name.split(marker, 1)[0]
    return name


# ----------------------------------------------------------------------
# add_difference — between-group mean / proportion differences
# ----------------------------------------------------------------------

def add_difference(
    table: SofraTable,
    *,
    digits: int = 2,
    conf_level: float = 0.95,
) -> SofraTable:
    """Add an absolute-difference column with CI for a 2-group Table 1."""
    if not (0.0 < conf_level < 1.0):
        raise ValueError(
            f"conf_level must lie in the open interval (0, 1); got {conf_level!r}."
        )
    return _add_difference_impl(table, digits=digits, conf_level=conf_level)


def _add_difference_impl(
    table: SofraTable,
    *,
    digits: int = 2,
    conf_level: float = 0.95,
) -> SofraTable:
    """Add an absolute-difference column with CI for a 2-group Table 1.

    For each continuous row, computes ``mean_2 - mean_1`` and its
    Welch confidence interval. For each dichotomous row, computes
    ``prop_2 - prop_1`` and its **Newcombe hybrid-score CI**
    (Newcombe 1998, *Stat Med* 17:873–890, Method 10). The Newcombe
    interval combines two single-sample Wilson scores and is the
    standard recommendation over the Wald (normal-approximation)
    interval, which collapses at extreme proportions. Multi-level
    categorical rows get a ``—``.

    Requires a SofraTable produced by ``tbl_one`` / ``tbl_summary``
    with exactly two groups (otherwise the differences are ambiguous).
    """
    if table._spec is None or table._spec.builder not in ("tbl_one",):
        raise ValueError(
            "add_difference is only supported on tbl_one / tbl_summary tables."
        )
    spec = table._spec
    opts = spec.options
    by = opts["by"]
    if by is None:
        raise ValueError("add_difference requires a stratification variable (by=).")

    # The rebuild closure is the only handle we have on the original data.
    rebuild = table._rebuild
    if rebuild is None:
        raise ValueError(
            "add_difference needs access to the original data — only tables "
            "built directly by tbl_one / tbl_summary qualify."
        )
    # Extract source data from the rebuild closure cell.
    data = _data_from_rebuild(rebuild)
    if data is None:
        raise ValueError("Could not recover source data from table closure.")

    by_series = data[by]
    levels = sorted(by_series.dropna().unique(), key=str)
    if len(levels) != 2:
        raise ValueError(
            f"add_difference requires exactly 2 groups; got {len(levels)}."
        )
    g1, g2 = levels
    mask1 = by_series == g1
    mask2 = by_series == g2

    kinds = opts["kinds"]
    variables = opts["variables"]

    # Frequency weights (if the original tbl_one was built with weights=).
    weights_col = _weights_col_from_spec(table)
    has_weights = weights_col is not None

    diffs: dict[str, tuple[float | None, float | None, float | None]] = {}
    for var in variables:
        if kinds[var] == "continuous":
            a = pd.to_numeric(data.loc[mask1, var], errors="coerce")
            b = pd.to_numeric(data.loc[mask2, var], errors="coerce")
            if has_weights:
                wa_full = pd.to_numeric(
                    data.loc[mask1, weights_col], errors="coerce",
                )
                wb_full = pd.to_numeric(
                    data.loc[mask2, weights_col], errors="coerce",
                )
                mask_a = a.notna() & wa_full.notna() & (wa_full > 0)
                mask_b = b.notna() & wb_full.notna() & (wb_full > 0)
                a_arr = a[mask_a].to_numpy(dtype=float)
                b_arr = b[mask_b].to_numpy(dtype=float)
                wa_arr = wa_full[mask_a].to_numpy(dtype=float)
                wb_arr = wb_full[mask_b].to_numpy(dtype=float)
                if a_arr.size < 2 or b_arr.size < 2:
                    diffs[var] = (None, None, None)
                    continue
                ma, va, ne_a = _weighted_mean_var_kish(a_arr, wa_arr)
                mb, vb, ne_b = _weighted_mean_var_kish(b_arr, wb_arr)
                if ne_a < 2 or ne_b < 2:
                    diffs[var] = (None, None, None)
                    continue
                diff = float(mb - ma)
                # Welch SE on weighted means uses Kish effective n's:
                sa2_n = va / ne_a
                sb2_n = vb / ne_b
                se = math.sqrt(sa2_n + sb2_n)
                df_w = (sa2_n + sb2_n) ** 2 / (
                    sa2_n ** 2 / max(ne_a - 1, 1)
                    + sb2_n ** 2 / max(ne_b - 1, 1)
                )
            else:
                a_drop = a.dropna()
                b_drop = b.dropna()
                if len(a_drop) < 2 or len(b_drop) < 2:
                    diffs[var] = (None, None, None)
                    continue
                diff = float(b_drop.mean() - a_drop.mean())
                se = math.sqrt(
                    b_drop.var(ddof=1) / len(b_drop)
                    + a_drop.var(ddof=1) / len(a_drop)
                )
                df_w = (
                    (b_drop.var(ddof=1) / len(b_drop)
                     + a_drop.var(ddof=1) / len(a_drop)) ** 2
                    / (
                        (b_drop.var(ddof=1) / len(b_drop)) ** 2
                        / (len(b_drop) - 1)
                        + (a_drop.var(ddof=1) / len(a_drop)) ** 2
                        / (len(a_drop) - 1)
                    )
                )
            tcrit = float(sp_stats.t.ppf(0.5 + conf_level / 2, df=df_w))
            diffs[var] = (diff, diff - tcrit * se, diff + tcrit * se)
        elif kinds[var] == "dichotomous":
            s = data[var]
            if isinstance(s.dtype, pd.CategoricalDtype):
                lvls = list(s.cat.categories)
            else:
                lvls = sorted(s.dropna().unique(), key=str)
            if len(lvls) != 2:
                diffs[var] = (None, None, None)
                continue
            success = lvls[1]
            if has_weights:
                w_col = pd.to_numeric(data[weights_col], errors="coerce").fillna(0.0)
                m1_w = mask1 & (w_col > 0) & s.notna()
                m2_w = mask2 & (w_col > 0) & s.notna()
                # Kish effective n + weighted "x" (number of weighted successes
                # treated as if rounded so Wilson can consume it).
                sw1 = float(w_col[m1_w].sum())
                sw2 = float(w_col[m2_w].sum())
                if sw1 == 0 or sw2 == 0:
                    diffs[var] = (None, None, None)
                    continue
                sw1_2 = float((w_col[m1_w] ** 2).sum())
                sw2_2 = float((w_col[m2_w] ** 2).sum())
                # Kish effective n_eff = (Σw)² / Σw²
                n1 = max(int(round((sw1 ** 2) / sw1_2)), 1)
                n2 = max(int(round((sw2 ** 2) / sw2_2)), 1)
                # Weighted success proportion
                p1 = float(
                    (w_col[m1_w] * (data.loc[m1_w, var] == success).astype(float)).sum()
                    / sw1
                )
                p2 = float(
                    (w_col[m2_w] * (data.loc[m2_w, var] == success).astype(float)).sum()
                    / sw2
                )
                # Wilson takes integer (x, n); convert via x = p * n_eff
                x1 = int(round(p1 * n1))
                x2 = int(round(p2 * n2))
            else:
                n1 = int(mask1.sum())
                n2 = int(mask2.sum())
                x1 = int((data.loc[mask1, var] == success).sum())
                x2 = int((data.loc[mask2, var] == success).sum())
                if n1 == 0 or n2 == 0:  # pragma: no cover
                    diffs[var] = (None, None, None)
                    continue
                p1, p2 = x1 / n1, x2 / n2
            diff = p2 - p1
            zcrit = float(sp_stats.norm.ppf(0.5 + conf_level / 2))
            # Newcombe's (1998) Method 10 — the Wilson-based hybrid CI
            # for the difference of two independent proportions. It is
            # the standard recommendation over the Wald
            # (normal-approximation) interval, which collapses at the
            # extremes p≈0 or p≈1. Reference: Newcombe (1998), Stat Med
            # 17:873–890. Here diff = p2 - p1; the lower bound is
            # attained at p1=U1, p2=L2 (and vice versa for the upper).
            lo1, hi1 = _wilson_ci(x1, n1, z=zcrit)
            lo2, hi2 = _wilson_ci(x2, n2, z=zcrit)
            lo = diff - math.sqrt((hi1 - p1) ** 2 + (p2 - lo2) ** 2)
            hi = diff + math.sqrt((p1 - lo1) ** 2 + (hi2 - p2) ** 2)
            diffs[var] = (diff, lo, hi)
        else:
            diffs[var] = (None, None, None)

    # Insert a new column right before any p-value column.
    new_headers = _insert_after_groups_header(
        table.headers,
        f"Diff ({int(round(conf_level * 100))}% CI)",
    )

    # Walk rows and patch.
    new_rows: list[Row] = []
    for r in table.rows:
        label = r.cells[0].text
        var = _find_variable_for_row(label, variables, kinds, labels=opts.get("labels"))
        text: str
        value: Any
        if var is not None and var in diffs:
            d_opt, lo_opt, hi_opt = diffs[var]
            if (
                d_opt is None or lo_opt is None or hi_opt is None
                or any(_isnan(x) for x in (d_opt, lo_opt, hi_opt))
            ):
                text = "—"
                value = None
            else:
                d, lo, hi = d_opt, lo_opt, hi_opt
                text = (
                    f"{fmt_number(d, digits)} "
                    f"({fmt_number(lo, digits)}, {fmt_number(hi, digits)})"
                )
                value = (d, lo, hi)
        else:
            text = ""
            value = None
        new_rows.append(_insert_after_groups_cell(r, text, value=value,
                                                  kind="ci"))
    return replace(table, headers=new_headers, rows=tuple(new_rows))


# ----------------------------------------------------------------------
# add_ci — confidence intervals for each summary cell
# ----------------------------------------------------------------------

def add_ci(
    table: SofraTable,
    *,
    conf_level: float = 0.95,
) -> SofraTable:
    """Append a parenthesised confidence interval to each summary cell.

    For continuous rows the existing ``mean (SD)`` cell becomes
    ``mean (SD) [lo, hi]`` where ``[lo, hi]`` is the Welch CI for the
    mean.  For dichotomous rows the ``n (%)`` cell gains a Wilson-score
    CI for the proportion.  Multi-level categorical rows are unchanged.
    """
    if not (0.0 < conf_level < 1.0):
        raise ValueError(
            f"conf_level must lie in the open interval (0, 1); got {conf_level!r}."
        )
    if table._spec is None or table._rebuild is None:
        raise ValueError(
            "add_ci needs access to the source data — only tables built "
            "directly by tbl_one / tbl_summary qualify."
        )
    data = _data_from_rebuild(table._rebuild)
    if data is None:
        raise ValueError("Could not recover source data from table closure.")

    opts = table._spec.options
    by = opts["by"]
    kinds = opts["kinds"]
    variables = opts["variables"]

    group_keys, group_masks = _resolve_groups(data, by)
    if opts.get("overall"):
        group_keys = [opts.get("overall_label", "Overall"), *group_keys]
        group_masks = {opts.get("overall_label", "Overall"):
                       pd.Series(True, index=data.index), **group_masks}

    new_rows: list[Row] = []
    z = float(sp_stats.norm.ppf(0.5 + conf_level / 2))

    # Honour the original tbl_one's weights= so the CI reflects the
    # weighted analysis the rest of the table is built on.
    weights_col = _weights_col_from_spec(table)
    has_weights = weights_col is not None

    for r in table.rows:
        label = r.cells[0].text
        var = _find_variable_for_row(label, variables, kinds, labels=opts.get("labels"))
        if var is None:
            new_rows.append(r)
            continue
        kind = kinds[var]
        # Patch group cells (columns 1..1+len(group_keys)).
        new_cells = list(r.cells)
        for offset, k in enumerate(group_keys):
            col = 1 + offset
            if col >= len(new_cells):
                break
            old = new_cells[col]
            mask = group_masks[k]
            if kind == "continuous":
                v_series = pd.to_numeric(data.loc[mask, var], errors="coerce")
                if has_weights:
                    w_series = pd.to_numeric(
                        data.loc[mask, weights_col], errors="coerce",
                    )
                    keep = v_series.notna() & w_series.notna() & (w_series > 0)
                    v_arr = v_series[keep].to_numpy(dtype=float)
                    w_arr = w_series[keep].to_numpy(dtype=float)
                    if v_arr.size < 2:
                        continue
                    m, var_, n_eff = _weighted_mean_var_kish(v_arr, w_arr)
                    if n_eff < 2 or not (np.isfinite(m) and np.isfinite(var_)):
                        continue
                    se = math.sqrt(var_ / n_eff)
                    tcrit = float(
                        sp_stats.t.ppf(0.5 + conf_level / 2, df=max(n_eff - 1, 1)),
                    )
                else:
                    v = v_series.dropna()
                    if len(v) < 2:
                        continue
                    m = float(v.mean())
                    se = float(v.std(ddof=1)) / math.sqrt(len(v))
                    tcrit = float(
                        sp_stats.t.ppf(0.5 + conf_level / 2, df=len(v) - 1),
                    )
                lo, hi = m - tcrit * se, m + tcrit * se
                ci = f" [{fmt_number(lo, 2)}, {fmt_number(hi, 2)}]"
                new_cells[col] = replace(old, text=old.text + ci)
            elif kind == "dichotomous" and "=" in label:
                # Dichotomous rows have "label = success_level"
                s = data[var]
                lvls = (list(s.cat.categories)
                        if isinstance(s.dtype, pd.CategoricalDtype)
                        else sorted(s.dropna().unique(), key=str))
                if len(lvls) != 2:
                    continue
                success = lvls[1]
                if has_weights:
                    w_full = pd.to_numeric(
                        data[weights_col], errors="coerce",
                    ).fillna(0.0)
                    m2 = mask & (w_full > 0) & s.notna()
                    sw = float(w_full[m2].sum())
                    sw2 = float((w_full[m2] ** 2).sum())
                    if sw == 0 or sw2 == 0:
                        continue
                    p = float(
                        (w_full[m2] *
                         (data.loc[m2, var] == success).astype(float)).sum() / sw
                    )
                    # Kish effective n → integer for Wilson
                    n_eff = max(int(round((sw * sw) / sw2)), 1)
                    x = int(round(p * n_eff))
                    n = n_eff
                else:
                    n = int(data.loc[mask, var].notna().sum())
                    x = int((data.loc[mask, var] == success).sum())
                    if n == 0:
                        continue
                lo, hi = _wilson_ci(x, n, z=z)
                ci = f" [{fmt_number(100*lo, 1)}%, {fmt_number(100*hi, 1)}%]"
                new_cells[col] = replace(old, text=old.text + ci)
        new_rows.append(replace(r, cells=tuple(new_cells)))

    fn = (
        f"Bracketed intervals: {int(round(conf_level*100))}% confidence "
        "interval (Welch for means, Wilson-score for proportions)."
    )
    return replace(
        table,
        rows=tuple(new_rows),
        footnotes=tuple([*table.footnotes, fn]),
    )


def _wilson_ci(x: int, n: int, *, z: float) -> tuple[float, float]:
    """Wilson score CI for a proportion.

    References
    ----------
    Wilson, E. B. (1927). Probable inference, the law of succession,
      and statistical inference. *J. Am. Stat. Assoc.*, 22(158), 209–212.
    """
    if n == 0:
        return float("nan"), float("nan")
    p = x / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


# ----------------------------------------------------------------------
# Formatter override modifiers
# ----------------------------------------------------------------------

def with_pvalue_fmt(
    table: SofraTable,
    fn: Callable[[float], str],
) -> SofraTable:
    """Re-format every p-value cell with ``fn(value) -> str``."""
    return _apply_formatter(table, kind="p_value", fn=fn)


def with_estimate_fmt(
    table: SofraTable,
    fn: Callable[[float], str],
) -> SofraTable:
    """Re-format every numeric estimate cell with ``fn(value) -> str``."""
    return _apply_formatter(table, kind="numeric", fn=fn)


def _apply_formatter(
    table: SofraTable,
    *,
    kind: str,
    fn: Callable[[float], str],
) -> SofraTable:
    new_rows: list[Row] = []
    for r in table.rows:
        new_cells = []
        for c in r.cells:
            if c.kind == kind and isinstance(c.value, (int, float)) \
                    and not _isnan(c.value):
                new_cells.append(replace(c, text=fn(float(c.value))))
            else:
                new_cells.append(c)
        new_rows.append(replace(r, cells=tuple(new_cells)))
    return replace(table, rows=tuple(new_rows))


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _isnan(x: Any) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return False


def _data_from_rebuild(rebuild: Callable[..., Any]) -> pd.DataFrame | None:
    """Recover the source DataFrame captured by a builder's rebuild closure."""
    closure = getattr(rebuild, "__closure__", None)
    if not closure:
        return None
    for cell in closure:
        contents = cell.cell_contents
        if isinstance(contents, pd.DataFrame):
            return contents
    return None


def _find_variable_for_row(
    label: str,
    variables: tuple[str, ...],
    kinds: dict[str, str],
    *,
    labels: dict[str, str] | None = None,
) -> str | None:
    """Match a body-row's displayed text back to its source variable.

    Handles three cases:

    * Raw variable name (``"age"``)
    * Dichotomous renaming (``"sex = Male"``)
    * Display-relabelled rows via the ``labels={...}`` argument to
      ``tbl_one`` (``"Patient sex = Male"`` for ``labels={"sex":
      "Patient sex"}``)
    """
    labels = labels or {}
    for v in variables:
        if label == v:
            return v
        if label.startswith(f"{v} ="):
            return v
    # Display-relabelled rows: scan the labels mapping.
    for src, disp in labels.items():
        if not disp:
            continue
        if label == disp:
            return src
        if label.startswith(f"{disp} ="):
            return src
    return None


def _resolve_groups(data: pd.DataFrame, by: str | None) -> tuple[list[Any], dict[Any, pd.Series]]:
    if by is None:
        return ["Overall"], {"Overall": pd.Series(True, index=data.index)}
    s = data[by]
    levels = (list(s.cat.categories)
              if isinstance(s.dtype, pd.CategoricalDtype)
              else sorted(s.dropna().unique(), key=str))
    return list(levels), {k: (s == k) for k in levels}


def _insert_after_pvalue_header(headers: tuple[HeaderRow, ...], label: str) -> tuple[HeaderRow, ...]:
    """Insert a header cell named ``label`` right after the first p-value column."""
    new_headers: list[HeaderRow] = []
    for hr in headers:
        new_cells = list(hr.cells)
        for j, c in enumerate(new_cells):
            if c.text.lower().startswith("p-value") or c.text.lower() == "p":
                new_cells.insert(j + 1, HeaderCell(text=label))
                break
        else:
            new_cells.append(HeaderCell(text=label))
        new_headers.append(HeaderRow(cells=tuple(new_cells)))
    return tuple(new_headers)


def _insert_after_pvalue_cell(row: Row, text: str, *, value: Any) -> Row:
    new_cells = list(row.cells)
    for j, c in enumerate(new_cells):
        if c.kind == "p_value":
            new_cells.insert(j + 1, make_cell(text, value=value, align="right"))
            break
    else:
        new_cells.append(make_cell(text, value=value, align="right"))
    return replace(row, cells=tuple(new_cells))


def _insert_after_groups_header(
    headers: tuple[HeaderRow, ...], label: str,
) -> tuple[HeaderRow, ...]:
    """Insert a header cell named ``label`` right before any p-value column."""
    new_headers: list[HeaderRow] = []
    for hr in headers:
        new_cells = list(hr.cells)
        insert_at = len(new_cells)
        for j, c in enumerate(new_cells):
            if c.text.lower().startswith(("p-value", "p", "smd")):
                insert_at = j
                break
        new_cells.insert(insert_at, HeaderCell(text=label))
        new_headers.append(HeaderRow(cells=tuple(new_cells)))
    return tuple(new_headers)


def _insert_after_groups_cell(
    row: Row,
    text: str,
    *,
    value: Any,
    kind: Any = "text",
) -> Row:
    new_cells: list[Cell] = list(row.cells)
    insert_at = len(new_cells)
    for j, c in enumerate(new_cells):
        if c.kind in ("p_value", "q_value") or (
            c.kind == "numeric" and j == len(new_cells) - 1
            and isinstance(c.value, (int, float))
            and not _isnan(c.value or 0)
            and c.text and c.text.replace(".", "").replace("-", "").isdigit()
        ):
            insert_at = j
            break
    new_cells.insert(insert_at, make_cell(text, value=value, kind=kind, align="right"))
    return replace(row, cells=tuple(new_cells))
