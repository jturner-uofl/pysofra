"""General descriptive summary tables — equivalent to ``gtsummary::tbl_summary``.

``tbl_summary`` is broadly the same engine as :func:`tbl_one` but with more
flexible knobs: you can pass custom statistic templates, override missing
handling per variable, and produce summaries without a stratification
variable as the natural default.

For the MVP, ``tbl_summary`` delegates to the same :func:`_build` engine
under the hood — there is genuinely one statistical computation; only the
defaults differ between the two front doors.
"""

from __future__ import annotations

import pandas as pd

from ..core.table import SofraTable
from .tbl_one import tbl_one
from .typing import VarKind


def tbl_summary(
    data: pd.DataFrame,
    *,
    by: str | None = None,
    variables: list[str] | None = None,
    labels: dict[str, str] | None = None,
    types: dict[str, VarKind] | None = None,
    nonnormal: list[str] | None = None,
    digits: int = 2,
    pct_digits: int = 1,
    missing: str = "ifany",
) -> SofraTable:
    """Build a general descriptive summary table.

    See :func:`pysofra.tbl_one` for parameter documentation. The two
    functions share an engine; the names exist separately because the
    *intent* differs (Table 1 baseline vs. arbitrary descriptive summary)
    and we may diverge their defaults further in future releases.
    """
    return tbl_one(
        data,
        by=by,
        variables=variables,
        labels=labels,
        types=types,
        nonnormal=nonnormal,
        digits=digits,
        pct_digits=pct_digits,
        missing=missing,
    )
