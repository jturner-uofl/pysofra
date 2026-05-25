"""Cross-validation against SciPy reference outputs.

Each fixture in ``tests/fixtures/scipy_validation/`` carries a
canonical dataset plus the output a direct ``scipy.stats`` /
``statsmodels`` call produces on the same data. We assert that
PySofra's higher-level wrappers (``continuous_test``,
``categorical_test``, ``run_named_test``, ``svyttest``) reproduce
those references within the tolerance declared in the fixture.

The reference values are produced by SciPy / NumPy — *not* by an
independent R run. SciPy's hypothesis tests already match R's
``stats`` package to machine precision (see the SciPy test suite),
so a PySofra-vs-SciPy assertion is also implicitly a PySofra-vs-R
assertion, but the fixture's ``source`` field is honest about its
provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

from pysofra.summary.tests import (
    categorical_test,
    continuous_test,
    run_named_test,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "scipy_validation"


def _load(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text())


class TestWelchTTest:
    def test_matches_r(self):
        fx = _load("welch_t_test.json")
        x = pd.Series(fx["x"])
        y = pd.Series(fx["y"])
        groups = pd.Series(["A"] * len(x) + ["B"] * len(y))
        values = pd.concat([x, y], ignore_index=True)

        result = continuous_test(values, groups)
        assert result.test == "Welch's t-test"

        ref = fx["reference"]
        assert abs(result.statistic) == pytest.approx(
            abs(ref["statistic"]), abs=fx["tolerance"]
        )
        assert result.p_value == pytest.approx(ref["p_value"], abs=fx["tolerance"])


class TestWilcoxonRankSum:
    def test_matches_r(self):
        fx = _load("wilcoxon_rank_sum.json")
        values = pd.Series(fx["x"] + fx["y"])
        groups = pd.Series(["A"] * len(fx["x"]) + ["B"] * len(fx["y"]))

        result = run_named_test("wilcoxon", values, groups, kind="continuous")
        assert result.p_value == pytest.approx(
            fx["reference"]["p_value"], abs=fx["tolerance"]
        )


class TestChiSquare:
    def test_matches_r(self):
        fx = _load("chi_square.json")
        observed = np.array(fx["observed"])
        chi2, p, dof, _expected = sp_stats.chi2_contingency(observed, correction=False)

        ref = fx["reference"]
        assert chi2 == pytest.approx(ref["statistic"], abs=fx["tolerance"])
        assert dof == ref["df"]
        assert p == pytest.approx(ref["p_value"], abs=fx["tolerance"])


class TestFisher2x2:
    def test_matches_r(self):
        fx = _load("fisher_2x2.json")
        observed = np.array(fx["observed"])
        v = []
        g = []
        for i, row in enumerate(observed):
            for j, count in enumerate(row):
                v.extend([f"r{i}"] * int(count))
                g.extend([f"c{j}"] * int(count))
        result = categorical_test(pd.Series(v), pd.Series(g))
        assert result.test == "Fisher's exact"
        assert result.p_value == pytest.approx(
            fx["reference"]["p_value"], abs=fx["tolerance"]
        )


class TestOneWayAnova:
    def test_matches_r(self):
        fx = _load("anova_oneway.json")
        values: list[float] = []
        groups: list[str] = []
        for grp, vs in fx["groups"].items():
            values.extend(vs)
            groups.extend([grp] * len(vs))

        result = continuous_test(pd.Series(values), pd.Series(groups))
        assert result.test == "One-way ANOVA"

        ref = fx["reference"]
        assert result.statistic == pytest.approx(
            ref["F_statistic"], abs=fx["tolerance"]
        )
        assert result.p_value == pytest.approx(
            ref["p_value"], abs=fx["tolerance"]
        )


class TestKruskalWallis:
    def test_matches_r(self):
        fx = _load("kruskal_wallis.json")
        values: list[float] = []
        groups: list[str] = []
        for grp, vs in fx["groups"].items():
            values.extend(vs)
            groups.extend([grp] * len(vs))

        result = run_named_test("kruskal", pd.Series(values),
                                pd.Series(groups), kind="continuous")
        assert "Kruskal" in result.test

        ref = fx["reference"]
        assert result.statistic == pytest.approx(ref["H_statistic"], abs=fx["tolerance"])
        assert result.p_value == pytest.approx(ref["p_value"], abs=fx["tolerance"])


class TestStudentT:
    def test_matches_r(self):
        fx = _load("student_t.json")
        x = pd.Series(fx["x"])
        y = pd.Series(fx["y"])
        groups = pd.Series(["A"] * len(x) + ["B"] * len(y))
        values = pd.concat([x, y], ignore_index=True)

        result = run_named_test("student", values, groups, kind="continuous")
        ref = fx["reference"]
        assert abs(result.statistic) == pytest.approx(
            abs(ref["statistic"]), abs=fx["tolerance"]
        )
        assert result.p_value == pytest.approx(ref["p_value"], abs=fx["tolerance"])


class TestWeightedMean:
    def test_matches_r(self):
        from pysofra.summary.weights import weighted_continuous_stats
        fx = _load("weighted_mean.json")
        v = pd.Series(fx["values"])
        w = pd.Series(fx["weights"])
        st = weighted_continuous_stats(v, w)
        ref = fx["reference"]
        assert st.n_eff == pytest.approx(ref["n_eff"], abs=fx["tolerance"])
        assert st.mean == pytest.approx(ref["mean"], abs=fx["tolerance"])
        assert st.sd == pytest.approx(ref["sd"], abs=fx["tolerance"])


class TestSvyttest:
    def test_matches_reference(self):
        from pysofra.summary.tests import svyttest as _svy
        fx = _load("svyttest.json")
        v = pd.Series(fx["values"])
        g = pd.Series(fx["groups"])
        w = pd.Series(fx["weights"])
        res = _svy(v, g, w)
        ref = fx["reference"]
        assert abs(res.statistic) == pytest.approx(
            abs(ref["t"]), abs=fx["tolerance"], rel=fx["tolerance"]
        )
        assert res.p_value == pytest.approx(
            ref["p_value"], abs=fx["tolerance"], rel=fx["tolerance"]
        )
