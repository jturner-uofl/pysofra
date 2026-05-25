"""Automatic variable typing for summary tables.

We classify each variable into one of four kinds:

* ``continuous``  — numeric and treated as continuous
* ``categorical`` — discrete factor variable (strings, booleans, low-cardinality ints)
* ``dichotomous`` — categorical with exactly two non-missing levels
* ``ordinal``     — pandas ``Categorical`` with ``ordered=True``

The classifier errs on the side of categorical when ambiguous (e.g. integer
columns with very few unique values) because mistakenly summarising a
factor as continuous produces nonsense output in publication tables.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
    is_timedelta64_dtype,
)


def _is_categorical(series: pd.Series) -> bool:
    return isinstance(series.dtype, pd.CategoricalDtype)

VarKind = Literal["continuous", "categorical", "dichotomous", "ordinal"]

# Integer columns with at most this many distinct values *and* whose values
# all lie in a small low-integer range get classified as categorical.
# Real-world continuous variables (age, BMI, systolic BP, etc.) virtually
# always exceed this — the heuristic is intentionally conservative so we
# don't accidentally summarise a continuous variable as n (%) per level.
_MAX_CAT_CARDINALITY_INT = 5
_MAX_CAT_INT_VALUE = 20


def infer_kind(series: pd.Series) -> VarKind:
    """Infer the variable kind of a pandas Series."""
    s = series.dropna()
    if s.empty:
        # No information — default to categorical so we render n (%) of NaNs.
        return "categorical"

    if _is_categorical(series):
        if getattr(series.cat, "ordered", False):
            return "ordinal"
        return "dichotomous" if s.nunique() == 2 else "categorical"

    if is_bool_dtype(series):
        return "dichotomous"

    if is_string_dtype(series) or is_object_dtype(series):
        return "dichotomous" if s.nunique() == 2 else "categorical"

    if is_numeric_dtype(series):
        uniques = s.unique()
        # Exactly {0, 1} (or {0.0, 1.0}) is a strong dichotomous signal.
        # ``int(np.inf)`` raises ``OverflowError``; columns containing
        # ``inf`` or ``-inf`` are by definition not 0/1, so we fall
        # through to the continuous branch instead of crashing.
        try:
            uvals = set(int(x) for x in uniques)
            if uvals.issubset({0, 1}) and len(uvals) == 2:
                return "dichotomous"
        except (TypeError, ValueError, OverflowError):
            pass

        try:
            # ``s.dtype`` is ``np.dtype | ExtensionDtype`` under
            # pandas-stubs; only the np.dtype branch is meaningful here
            # (extension dtypes are handled earlier), so silence the
            # narrowed union for this site.
            is_int = bool(np.issubdtype(s.dtype, np.integer))  # type: ignore[arg-type]
        except TypeError:  # pragma: no cover — defensive: numpy is_numeric_dtype guarantees a known dtype
            is_int = False
        if (
            is_int
            and len(uniques) <= _MAX_CAT_CARDINALITY_INT
            and float(np.min(uniques)) >= -_MAX_CAT_INT_VALUE
            and float(np.max(uniques)) <= _MAX_CAT_INT_VALUE
        ):
            return "dichotomous" if len(uniques) == 2 else "categorical"
        return "continuous"

    # Datetime / timedelta — PySofra doesn't summarise temporal columns
    # natively (there is no "median date (Q1, Q3)" idiom that maps cleanly
    # to a publication table). Falling through to the categorical branch
    # would put every unique timestamp on its own row, which is almost
    # always not what the user wants. Emit a UserWarning so they notice
    # and switch to a derived numeric column (e.g.
    # ``(df.date - ref).dt.days``), then still return categorical so the
    # call doesn't crash.
    if is_datetime64_any_dtype(series) or is_timedelta64_dtype(series):
        import warnings
        warnings.warn(
            f"Variable {series.name!r} has dtype {series.dtype!s}; PySofra "
            "does not summarise temporal columns. Convert it to a numeric "
            "duration (e.g. (df.date - reference).dt.days) and pass that "
            "instead.",
            UserWarning,
            stacklevel=2,
        )
    # Any other unrecognised dtype — treat as categorical for safety.
    return "categorical"


def apply_overrides(
    inferred: dict[str, VarKind],
    overrides: dict[str, VarKind] | None,
) -> dict[str, VarKind]:
    if not overrides:
        return inferred
    merged = dict(inferred)
    for k, v in overrides.items():
        if v not in ("continuous", "categorical", "dichotomous", "ordinal"):
            raise ValueError(
                f"Invalid variable kind {v!r} for {k!r}. "
                "Must be one of continuous, categorical, dichotomous, ordinal."
            )
        merged[k] = v
    return merged
