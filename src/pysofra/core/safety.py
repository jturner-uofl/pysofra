"""Publication-safety auto-checker for SofraTable.

A small set of pattern-based audits that flag rows whose statistics
have, in published clinical / epidemiological literature, been
associated with errata or retractions. The checks are deliberately
conservative: they only flag patterns whose presence is almost
always a coding error or a methodological oversight, not a
substantive finding.

The intended workflow:

>>> t = ps.tbl_one(df, by='arm').add_p()
>>> for w in t.check_safety():
...     print(w)
>>> # Or attach the warnings as footnotes on the published table:
>>> t_safe = t.with_safety_warnings()

This is *not* a substitute for statistical review or peer review;
it catches obvious mechanical errors before they reach the page.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .table import SofraTable


# ----------------------------------------------------------------------
# Thresholds.  Each is chosen to (a) never trigger on a genuinely
# published clinical finding the author would actually want to keep,
# and (b) reliably trigger on the corresponding coding mistake.
# ----------------------------------------------------------------------

# A subgroup with >= this many rows whose proportion rounds to exactly
# 100% or exactly 0% almost always reflects a coding error (e.g. the
# wrong outcome column was used, or the reference level was reversed).
_PROP_DEGENERATE_MIN_N = 30

# A p-value rendered as < 0.001 with effective n below this threshold
# is often an under-powered "interesting" finding that won't replicate;
# flag for the reader.
_SPARSE_PVALUE_MIN_N = 30

# Standardised mean difference > 1.0 between two arms of an RCT
# is essentially impossible after randomisation, and outside RCTs
# is large enough to suggest a coding mismatch (units, scaling).
_SMD_EXTREME = 1.0

# An exponentiated coefficient (OR / HR / IRR / TR) outside [0.1, 10]
# in a clinical-trial context is almost always either separation or
# a units/coding error (e.g. years vs months, mmHg vs kPa).
_EXP_COEF_LO = 0.1
_EXP_COEF_HI = 10.0

# A variable missing in more than this fraction of the analytic
# sample seriously threatens generalisability; flag the column for
# a missing-data discussion / sensitivity analysis.
_MISSING_THRESHOLD = 0.50


@dataclass(frozen=True)
class SafetyWarning:
    """One flagged row from :func:`check_safety`."""

    code: str          # short machine-readable identifier
    row_label: str     # the cell-0 text of the offending row
    message: str       # human-readable description of the concern

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"[{self.code}] {self.row_label}: {self.message}"


# ----------------------------------------------------------------------
# Patterns
# ----------------------------------------------------------------------

# "n (xx.x%)"  or  "n (xx%)"  — capture the percentage
_PCT_RE = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*%\s*\)")
# "n,nnn (xx.x%)" — capture the n
_N_PCT_RE = re.compile(r"([\d,]+)\s*\(\s*(-?\d+(?:\.\d+)?)\s*%\s*\)")


def _parse_n_pct(text: str) -> tuple[int, float] | None:
    """From a cell like '152 (38.0%)' return (152, 38.0); else None."""
    m = _N_PCT_RE.search(text)
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", "")), float(m.group(2))
    except ValueError:
        return None


def _row_kind(row, headers) -> str:
    """Heuristic — classify a row by first-column label + cell content."""
    if not row.cells:
        return "unknown"
    label = row.cells[0].text.strip()
    if not label:
        return "unknown"
    # Missing-row formatting from tbl_one: indented "Missing"
    if label.lower().startswith("missing"):
        return "missing"
    # Empty group cells with content only in p / smd → header for
    # a multi-level categorical.
    return "row"


# ----------------------------------------------------------------------
# Individual checks
# ----------------------------------------------------------------------

def _check_extreme_proportions(table: SofraTable,
                                ) -> list[SafetyWarning]:
    out: list[SafetyWarning] = []
    for row in table.rows:
        if not row.cells:
            continue
        for cell in row.cells[1:]:
            parsed = _parse_n_pct(cell.text)
            if parsed is None:
                continue
            n, pct = parsed
            if n >= _PROP_DEGENERATE_MIN_N and pct in (0.0, 100.0):
                out.append(SafetyWarning(
                    code="extreme_proportion",
                    row_label=row.cells[0].text,
                    message=(
                        f"a cell reports {pct:.0f}% on n={n} "
                        f"(≥ {_PROP_DEGENERATE_MIN_N}); often a coding "
                        f"error (wrong outcome column / reference level "
                        f"reversed). Verify the level definition."
                    ),
                ))
                break  # one warning per row is enough
    return out


def _check_sd_exceeds_mean(table: SofraTable,
                            ) -> list[SafetyWarning]:
    """Continuous rows showing 'Mean (SD)' with SD > |Mean|.

    Often indicates outliers, units mismatch (e.g. SD in raw data
    after a log transform), or a variable that should be summarised
    as median (Q1, Q3) instead.
    """
    out: list[SafetyWarning] = []
    cont_re = re.compile(
        r"^\s*(-?\d+(?:\.\d+)?)\s*\(\s*(\d+(?:\.\d+)?)\s*\)\s*$"
    )
    for row in table.rows:
        if not row.cells:
            continue
        for cell in row.cells[1:]:
            m = cont_re.match(cell.text)
            if not m:
                continue
            mean = float(m.group(1))
            sd = float(m.group(2))
            if abs(mean) > 1e-9 and sd > abs(mean):
                out.append(SafetyWarning(
                    code="sd_exceeds_mean",
                    row_label=row.cells[0].text,
                    message=(
                        f"a cell reports Mean (SD) = {mean:g} ({sd:g}) "
                        f"— SD exceeds |Mean|, suggesting outliers or a "
                        f"skewed distribution. Consider Median (Q1, Q3) "
                        f"via the `nonnormal=` argument."
                    ),
                ))
                break
    return out


def _check_sparse_pvalues(table: SofraTable,
                          ) -> list[SafetyWarning]:
    """p < 0.001 in a row whose group-cell n's are all below threshold."""
    out: list[SafetyWarning] = []
    for row in table.rows:
        p_val = None
        cell_ns: list[int] = []
        for cell in row.cells:
            if cell.kind == "p_value" and isinstance(cell.value, (int, float)):
                p_val = float(cell.value)
            else:
                parsed = _parse_n_pct(cell.text)
                if parsed is not None:
                    cell_ns.append(parsed[0])
        if p_val is None or p_val >= 0.001:
            continue
        if cell_ns and all(n < _SPARSE_PVALUE_MIN_N for n in cell_ns):
            out.append(SafetyWarning(
                code="sparse_pvalue",
                row_label=row.cells[0].text,
                message=(
                    f"p < 0.001 reported with every cell n below "
                    f"{_SPARSE_PVALUE_MIN_N} (cells: {cell_ns}). "
                    f"Use Fisher's exact via `tests={{'var': 'fisher'}}` "
                    f"and discuss in the limitations."
                ),
            ))
    return out


def _check_extreme_smd(table: SofraTable,
                        ) -> list[SafetyWarning]:
    """SMD column with |SMD| > _SMD_EXTREME."""
    out: list[SafetyWarning] = []
    # Locate the SMD column from headers
    smd_col_idx = None
    if table.headers:
        for j, h in enumerate(table.headers[0].cells):
            if h.text.strip().upper() == "SMD":
                smd_col_idx = j
                break
    if smd_col_idx is None:
        return out
    for row in table.rows:
        if smd_col_idx >= len(row.cells):
            continue
        v = row.cells[smd_col_idx].value
        if not isinstance(v, (int, float)):
            continue
        if abs(v) > _SMD_EXTREME:
            out.append(SafetyWarning(
                code="extreme_smd",
                row_label=row.cells[0].text,
                message=(
                    f"|SMD| = {abs(v):.2f} > {_SMD_EXTREME} — extreme "
                    f"between-arm imbalance. In an RCT this is "
                    f"essentially impossible after randomisation; in "
                    f"observational work it often indicates a "
                    f"coding / units mismatch. Inspect the variable."
                ),
            ))
    return out


def _check_extreme_exp_coef(table: SofraTable,
                             ) -> list[SafetyWarning]:
    """A regression-table cell with OR/HR/TR/IRR outside [0.1, 10]."""
    out: list[SafetyWarning] = []
    if not table.headers:
        return out
    headers = [h.text.strip().upper() for h in table.headers[0].cells]
    exp_idx = None
    for j, h in enumerate(headers):
        if h in ("OR", "HR", "IRR", "RR", "TR"):
            exp_idx = j
            break
    if exp_idx is None:
        return out
    for row in table.rows:
        if exp_idx >= len(row.cells):
            continue
        v = row.cells[exp_idx].value
        if not isinstance(v, (int, float)):
            continue
        if v <= 0 or v < _EXP_COEF_LO or v > _EXP_COEF_HI:
            out.append(SafetyWarning(
                code="extreme_exp_coef",
                row_label=row.cells[0].text,
                message=(
                    f"exponentiated coefficient = {v:g}, outside the "
                    f"[{_EXP_COEF_LO}, {_EXP_COEF_HI}] range that is "
                    f"physiologically plausible in most clinical "
                    f"contexts. Almost always a units / scaling error "
                    f"or unrecognised separation; verify before "
                    f"publishing."
                ),
            ))
    return out


def _check_dominant_missing(table: SofraTable,
                             ) -> list[SafetyWarning]:
    """A `Missing` row whose count exceeds 50 % of the column total."""
    out: list[SafetyWarning] = []
    # Column N totals come from the header text e.g. "A · N = 200"
    col_totals: list[int | None] = [None]
    if table.headers:
        for cell in table.headers[0].cells[1:]:
            m = re.search(r"N\s*=\s*([\d,]+)", cell.text)
            col_totals.append(int(m.group(1).replace(",", "")) if m else None)
    for row in table.rows:
        if not row.cells:
            continue
        label = row.cells[0].text.strip().lower()
        if not (label == "missing" or label.startswith("missing")):
            continue
        for j, cell in enumerate(row.cells[1:], start=1):
            if j >= len(col_totals) or col_totals[j] is None:
                continue
            parsed = _parse_n_pct(cell.text)
            if parsed is None:
                continue
            n_miss, _ = parsed
            denom = col_totals[j]
            if denom and n_miss / denom > _MISSING_THRESHOLD:
                # Find the variable label by walking up to the
                # nearest non-Missing label
                var_label = "(unknown variable)"
                for upstream in table.rows:
                    if upstream is row:
                        break
                    lab = upstream.cells[0].text.strip()
                    if lab and not lab.lower().startswith("missing"):
                        var_label = lab
                out.append(SafetyWarning(
                    code="dominant_missing",
                    row_label=var_label,
                    message=(
                        f"variable is missing in {n_miss}/{denom} "
                        f"({100 * n_miss / denom:.1f}%) of one column "
                        f"— exceeds the {int(100 * _MISSING_THRESHOLD)}% "
                        f"generalisability threshold. Discuss in the "
                        f"limitations or sensitivity-analyse."
                    ),
                ))
                break  # one per variable
    return out


# ----------------------------------------------------------------------
# Public surface
# ----------------------------------------------------------------------

def check_safety(table: SofraTable) -> list[SafetyWarning]:
    """Run every safety check; return the flat list of warnings."""
    out: list[SafetyWarning] = []
    out.extend(_check_extreme_proportions(table))
    out.extend(_check_sd_exceeds_mean(table))
    out.extend(_check_sparse_pvalues(table))
    out.extend(_check_extreme_smd(table))
    out.extend(_check_extreme_exp_coef(table))
    out.extend(_check_dominant_missing(table))
    return out


def with_safety_warnings(table: SofraTable) -> SofraTable:
    """Return a copy of ``table`` whose footnotes carry the safety flags.

    No-op (returns the original) when no warnings fire — keeps the
    rendered output clean for safe tables.
    """
    from dataclasses import replace
    warns = check_safety(table)
    if not warns:
        return table
    new_footnotes = list(table.footnotes)
    new_footnotes.append(
        f"SAFETY ({len(warns)} flag{'s' if len(warns) != 1 else ''}):"
    )
    for w in warns:
        new_footnotes.append(f"  • [{w.code}] {w.row_label}: {w.message}")
    return replace(table, footnotes=tuple(new_footnotes))
