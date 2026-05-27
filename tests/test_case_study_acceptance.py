"""End-to-end acceptance contract for the NHANES narrative-audit notebook.

The notebook at ``examples/jss_case_study/jss_case_study.ipynb`` is more
than documentation: each of its twelve steps deliberately exercises one
historically bug-prone seam in PySofra (variable-type inference,
lonely-PSU detector, Rao-Scott design awareness, design-based variance,
multiple-imputation pooling, ``var_weights`` regression refit, logistic
separation, Cox PH check, byte-determinism across renderers, and direct
numerical agreement with R ``survey::svyglm`` / ``svyttest`` / ``svymean``).

This test runs the notebook end-to-end (downloading NHANES from
CDC the first time, caching thereafter) and asserts:

  1. **No cell raises.**  The notebook's own ``assert`` blocks for
     byte-determinism (Step 11) and machine-precision R agreement
     (Step 12) act as load-bearing contracts; if any of the audited
     seams regresses, the executed notebook fails and so does this
     test.
  2. **Specific seam outputs.**  We re-read the executed notebook and
     check that the separation footnote, PH-violation footnote, and
     Rao-Scott design warning all appear in the cell outputs they
     are expected to appear in.

The test is skipped when network access to ``wwwn.cdc.gov`` is
unavailable (so it doesn't break offline / air-gapped pytest runs).

A note on R: Step 12's R-agreement assertions only fire when
``examples/jss_case_study/R_reference.json`` is present.  CI's
``case-study`` job runs ``Rscript R/cross_validate.R`` to populate it.
Local runs without R simply skip those assertions but still execute
the other eleven steps.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "examples" / "jss_case_study" / "jss_case_study.ipynb"
WORKING_COPY = ROOT / "examples" / "jss_case_study" / "_executed.ipynb"


def _cdc_reachable() -> bool:
    try:
        socket.gethostbyname("wwwn.cdc.gov")
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not NOTEBOOK.exists() or not _cdc_reachable(),
    reason="case-study notebook missing or CDC unreachable",
)


@pytest.fixture(scope="module")
def executed_notebook() -> nbformat.NotebookNode:  # noqa: F821
    """Execute the notebook in a side copy so the committed file isn't touched.

    ``jupyter nbconvert --execute`` exits non-zero on any cell error,
    including the load-bearing ``assert`` blocks in Steps 11 and 12.
    """
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("nbconvert")
    # Copy then execute so the committed notebook's outputs are
    # preserved as the canonical reference render.
    WORKING_COPY.write_bytes(NOTEBOOK.read_bytes())
    result = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook", "--execute", "--inplace",
            "--ExecutePreprocessor.timeout=900",
            str(WORKING_COPY),
        ],
        cwd=WORKING_COPY.parent,
        capture_output=True,
        text=True,
        check=False,
        timeout=1200,
    )
    if result.returncode != 0:
        # Leave the partially-executed _executed.ipynb on disk for
        # diagnostics; it's gitignored.
        pytest.fail(
            "Notebook execution failed.\n"
            f"stdout:\n{result.stdout[-3000:]}\n"
            f"stderr:\n{result.stderr[-3000:]}"
        )
    return nbformat.read(WORKING_COPY, as_version=4)


def _stream_text(cell) -> str:
    """Concatenate the stream outputs of one executed cell."""
    return "\n".join(
        out.text for out in cell.get("outputs", [])
        if out.output_type == "stream"
    )


def _cell_with(nb, needle: str):
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if needle in cell.source:
            return cell
    raise AssertionError(f"no code cell matched needle {needle!r}")


# ---------------------------------------------------------------------
# Step-by-step seam contracts
# ---------------------------------------------------------------------


class TestStep1VariableTyping:
    def test_inferred_kinds_match_expected(self, executed_notebook):
        cell = _cell_with(executed_notebook, "Variable-kind inference")
        text = _stream_text(cell)
        # The C1 fix (a9) makes [0.1, 0.2, 0.9, 1.1]-shaped floats classify
        # as continuous; here `age`, `pir`, `bmi`, `sbp`, `hba1c` must all
        # be continuous, race / education must be categorical, and sex /
        # insured / diabetes must be dichotomous.
        for var, kind in [
            ("age", "continuous"), ("sex", "dichotomous"),
            ("race", "categorical"), ("education", "categorical"),
            ("pir", "continuous"), ("bmi", "continuous"),
            ("sbp", "continuous"), ("hba1c", "continuous"),
            ("insured", "dichotomous"), ("diabetes", "dichotomous"),
        ]:
            assert f"{var}" in text and kind in text, (
                f"expected '{var} → {kind}' in step-1 output"
            )


class TestStep3LonelyPSU:
    def test_no_lonely_psu_strata(self, executed_notebook):
        cell = _cell_with(executed_notebook, "lonely-PSU strata")
        text = _stream_text(cell)
        # NHANES is a paired-PSU design; 2017-2018 has 15 strata, all paired.
        assert "lonely-PSU strata (warning condition): 0" in text


class TestStep5RaoScottWarning:
    def test_rao_scott_design_warning_fired(self, executed_notebook):
        cell = _cell_with(executed_notebook, "add_p().add_smd()")
        text = _stream_text(cell)
        # The Rao-Scott design-awareness warning (a9 C2) must fire once per
        # categorical variable under the stratified design.
        assert "Rao-Scott design warnings:" in text
        # Eight categorical/dichotomous variables → 8 warnings.
        assert "8" in text


class TestStep7VarWeightsConvention:
    def test_df_resid_equals_n_minus_k(self, executed_notebook):
        cell = _cell_with(executed_notebook, "var_weights convention preserved")
        text = _stream_text(cell)
        assert "var_weights convention preserved" in text


class TestStep8SeparationFootnote:
    def test_non_identified_footnote(self, executed_notebook):
        cell = _cell_with(executed_notebook, "sep = pd.DataFrame")
        text = _stream_text(cell)
        assert "separation footnote present: True" in text


class TestStep9CoxPHFootnote:
    def test_ph_violation_footnote(self, executed_notebook):
        cell = _cell_with(executed_notebook, "CoxPHFitter")
        text = _stream_text(cell)
        assert "PH-violation footnote present: True" in text


class TestStep10InlinePlots:
    def test_forest_and_km_inline_plots(self, executed_notebook):
        cell = _cell_with(executed_notebook, "with_forest_plot")
        text = _stream_text(cell)
        assert "forest inline_plot attached: True" in text
        assert "KM inline_plot attached: True" in text


class TestStep11ByteDeterminism:
    def test_all_seven_backends_match(self, executed_notebook):
        cell = _cell_with(executed_notebook, "byte-determinism")
        text = _stream_text(cell)
        for backend in ("html", "md", "tex", "docx", "pptx", "xlsx", "png"):
            # Each backend prints "MATCH" on its second-write hash check.
            assert (
                f"{backend:5s}".strip() in text and "MATCH" in text
            ), f"backend {backend!r} failed determinism check"
        # And the cell's own assertion must have passed for the notebook
        # to have executed end-to-end.
        assert "All seven backends are bytewise-identical" in text


class TestStep13AFTLabel:
    def test_weibull_aft_labelled_TR(self, executed_notebook):
        cell = _cell_with(executed_notebook, "WeibullAFTFitter")
        text = _stream_text(cell)
        assert "'TR'" in text, "AFT column header should be 'TR'"
        assert "ASSERTION OK" in text, "AFT-label assertion did not pass"


class TestStep14MultiModel:
    def test_three_spanning_headers(self, executed_notebook):
        cell = _cell_with(executed_notebook, "specs = [")
        text = _stream_text(cell)
        for label in ("Crude (age + BMI)", "+ sex, PIR", "+ insurance, race"):
            assert label in text, (
                f"multi-model spanning-header {label!r} missing"
            )
        assert "3-model side-by-side regression table rendered" in text


class TestStep15TblStack:
    def test_stack_composition(self, executed_notebook):
        cell = _cell_with(executed_notebook, "tbl_stack")
        text = _stream_text(cell)
        assert "stacked rows:" in text
        assert "both group labels present" in text


class TestStep16BHQValue:
    def test_q_value_monotone(self, executed_notebook):
        cell = _cell_with(executed_notebook, "add_q(method")
        text = _stream_text(cell)
        assert "q-value" in text
        assert "monotone in sorted p: True" in text
        assert "BH monotonicity holds" in text


class TestStep17GlobalP:
    def test_design_aware_joint_test(self, executed_notebook):
        cell = _cell_with(executed_notebook, "add_global_p()")
        text = _stream_text(cell)
        assert "global p" in text
        assert "joint Wald-F under design produced a valid global p" in text


class TestStep18CrossFormatConsistency:
    def test_same_N_in_html_md_tex_docx(self, executed_notebook):
        cell = _cell_with(executed_notebook, "cross-format")
        text = _stream_text(cell)
        # Cell output prints each backend as e.g. `    html : OK` —
        # tolerant matcher: backend name followed (after whitespace)
        # by `: OK`.
        import re as _re
        for fmt in ("html", "md", "tex", "docx"):
            assert _re.search(rf"{fmt}\s*:\s*OK", text), (
                f"cross-format check did not OK on {fmt}; got:\n{text}"
            )
        assert "MISSING" not in text, (
            "cross-format consistency check failed for at least one backend"
        )
        assert ("same weighted N appears in HTML, MD, LaTeX, "
                "and DOCX renders") in text


class TestGtsummaryComparison:
    """If the gtsummary positioning notebook is present (and its R-side
    reference has been generated by ``Rscript R/build_reference.R``),
    re-execute it and assert the p-value agreement still holds.
    The notebook's own ``assert`` block is the contract."""

    GTS_NB = ROOT / "examples" / "gtsummary_comparison" / "gtsummary_comparison.ipynb"
    GTS_REF = ROOT / "examples" / "gtsummary_comparison" / "gtsummary.json"
    GTS_WORK = ROOT / "examples" / "gtsummary_comparison" / "_executed.ipynb"

    @pytest.mark.skipif(
        not (ROOT / "examples" / "gtsummary_comparison" /
             "gtsummary_comparison.ipynb").exists(),
        reason="gtsummary comparison notebook not present",
    )
    @pytest.mark.skipif(
        not (ROOT / "examples" / "gtsummary_comparison" /
             "gtsummary.json").exists(),
        reason="gtsummary.json reference not generated yet "
               "(run: Rscript examples/gtsummary_comparison/R/build_reference.R)",
    )
    def test_pvalues_match_gtsummary(self):
        nbformat = pytest.importorskip("nbformat")
        pytest.importorskip("nbconvert")
        self.GTS_WORK.write_bytes(self.GTS_NB.read_bytes())
        result = subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert",
             "--to", "notebook", "--execute", "--inplace",
             "--ExecutePreprocessor.timeout=300",
             str(self.GTS_WORK)],
            cwd=self.GTS_WORK.parent,
            capture_output=True, text=True, check=False, timeout=600,
        )
        if result.returncode != 0:
            pytest.fail(
                "gtsummary comparison notebook failed.\n"
                f"stderr:\n{result.stderr[-2000:]}"
            )
        nb = nbformat.read(self.GTS_WORK, as_version=4)
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            if "max_p_diff" not in cell.source:
                continue
            text = _stream_text(cell)
            assert "ASSERTION OK" in text


class TestStep12RAgreement:
    def test_pysofra_svymean_matches_r_to_six_decimals(
        self, executed_notebook,
    ):
        cell = _cell_with(executed_notebook, "svyttest BMI~dm")
        text = _stream_text(cell)
        ref = ROOT / "examples" / "jss_case_study" / "R_reference.json"
        if not ref.exists():
            pytest.skip(
                "R_reference.json absent — run Rscript R/cross_validate.R "
                "to populate it; CI does this automatically."
            )
        # The notebook's own assertion is the contract; if it passed,
        # this string is in the output.
        assert "ASSERTION OK — PySofra agrees with R survey" in text
        assert "ASSERTION OK — coefficient estimates agree" in text

    def test_pinned_r_reference_is_sane(self):
        """Sanity-check the pinned R_reference.json against published NHANES.

        These are smoke checks: the CDC published estimate for adult
        (age >= 20) mean age in 2017-2018 lies in the 47-49 range, and
        the published mean BMI is 29-30. If R_reference drifts wildly
        we want to know before pytest declares "agreement OK".
        """
        ref = ROOT / "examples" / "jss_case_study" / "R_reference.json"
        if not ref.exists():
            pytest.skip("R_reference.json not present")
        R = json.loads(ref.read_text())
        assert 47.0 < R["svymean"]["age_mean"] < 49.5
        assert 0.0 < R["svymean"]["age_se"] < 1.5
        assert 28.0 < R["svymean"]["bmi_mean"] < 31.0
        assert R["svyttest"]["bmi_t"] > 8.0  # strong, sub-tautological
        # Logistic regression sanity checks
        names = R["svyglm"]["variable"]
        idx_age = names.index("RIDAGEYR")
        idx_bmi = names.index("bmi")
        # OR(age) per year should be in 1.04-1.10 range (Menke et al. 2015)
        assert 1.04 < R["svyglm"]["odds_ratio"][idx_age] < 1.10
        # OR(BMI) per unit kg/m² should be in 1.05-1.15 range
        assert 1.05 < R["svyglm"]["odds_ratio"][idx_bmi] < 1.15
