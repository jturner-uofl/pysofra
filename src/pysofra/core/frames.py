"""DataFrame adaptation.

PySofra's public API accepts any object with the pandas DataFrame shape —
``pandas.DataFrame``, ``polars.DataFrame``, or ``polars.LazyFrame``. The
adapter in this module normalises the input to pandas internally so the
statistical engines have one type to reason about.

We keep the dependency on polars *optional* — importing this module does
not require polars to be installed.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def to_pandas(data: Any) -> pd.DataFrame:
    """Convert a DataFrame-like input to a pandas DataFrame.

    Accepted inputs:

    * ``pandas.DataFrame`` — returned as-is (no copy).
    * ``polars.DataFrame`` — converted via ``.to_pandas()``.
    * ``polars.LazyFrame`` — collected first, then converted.
    * Any object exposing ``.to_pandas()`` — invoked and validated.

    Raises ``TypeError`` for unrecognised inputs.
    """
    if isinstance(data, pd.DataFrame):
        return data

    # Duck-typed polars detection — don't import polars unless we see it.
    cls = type(data)
    qualname = f"{cls.__module__}.{cls.__name__}"
    if qualname.startswith("polars."):
        # LazyFrame needs an explicit ``.collect()`` first.
        if cls.__name__ == "LazyFrame":
            data = data.collect()
        try:
            pandas_df = data.to_pandas()
        except (ImportError, ModuleNotFoundError):  # pragma: no cover
            # ``polars.DataFrame.to_pandas`` routes through pyarrow by
            # default. The ``pysofra[polars]`` and ``pysofra[all]``
            # extras now declare ``pyarrow``, so this fallback is
            # exercised only by users who hand-pin polars without it.
            # Falls back to a column-wise conversion that needs only
            # the standard library + pandas.
            pandas_df = pd.DataFrame(
                {col: data[col].to_list() for col in data.columns}
            )
        if not isinstance(pandas_df, pd.DataFrame):  # pragma: no cover
            raise TypeError(
                "polars to_pandas() did not return a pandas DataFrame; "
                f"got {type(pandas_df).__name__}."
            )
        return pandas_df

    # Generic fallback: any object that knows how to give us pandas.
    if hasattr(data, "to_pandas"):
        result = data.to_pandas()
        if isinstance(result, pd.DataFrame):
            return result

    raise TypeError(
        f"Unsupported DataFrame type {qualname!r}. "
        "PySofra accepts pandas.DataFrame and polars.DataFrame / LazyFrame."
    )
