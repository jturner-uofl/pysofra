"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def small_trial() -> pd.DataFrame:
    """Tiny synthetic two-arm trial dataset."""
    rng = np.random.default_rng(20260520)
    n = 60
    arm = np.array(["A"] * 30 + ["B"] * 30)
    df = pd.DataFrame({
        "arm": arm,
        "age": rng.normal(60, 10, n).round(1),
        "bmi": rng.normal(27, 4, n).round(1),
        "sex": rng.choice(["F", "M"], n),
        "smoker": rng.choice([0, 1], n, p=[0.7, 0.3]),
        "race": rng.choice(["White", "Black", "Asian", "Other"], n),
        "event": rng.integers(0, 2, n),
    })
    # Introduce a small amount of missingness in age
    df.loc[df.sample(3, random_state=1).index, "age"] = np.nan
    return df


@pytest.fixture
def trial_with_three_arms() -> pd.DataFrame:
    rng = np.random.default_rng(99)
    n = 90
    arm = np.array(["A"] * 30 + ["B"] * 30 + ["C"] * 30)
    return pd.DataFrame({
        "arm": arm,
        "age": rng.normal(60, 10, n).round(1),
        "sex": rng.choice(["F", "M"], n),
    })
