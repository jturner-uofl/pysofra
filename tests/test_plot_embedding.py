"""Tests for cross-backend plot image embedding (DOCX, PPTX, LaTeX)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pysofra as ps


@pytest.fixture
def regression_with_plot():
    sm = pytest.importorskip("statsmodels.api")
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "age": rng.normal(60, 10, n),
        "bmi": rng.normal(27, 4, n),
        "sex_M": rng.integers(0, 2, n),
    })
    df["event"] = (df["age"] / 10 + rng.normal(0, 1, n) > 6).astype(int)
    X = sm.add_constant(df[["age", "bmi", "sex_M"]])
    fit = sm.Logit(df["event"], X).fit(disp=False)
    return ps.tbl_regression(fit, exponentiate=True).with_forest_plot()


class TestInlinePlot:
    def test_carries_svg_png_pdf(self, regression_with_plot):
        plot = regression_with_plot.inline_plot
        assert plot is not None
        assert plot.svg.startswith("<svg")
        assert plot.png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
        assert plot.pdf_bytes.startswith(b"%PDF")


class TestDocxEmbedding:
    def test_docx_includes_image(self, regression_with_plot, tmp_path):
        out = tmp_path / "with_forest.docx"
        regression_with_plot.to_docx(out)
        # .docx is a zip; the image lands at word/media/<file>.png
        import zipfile
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
            media = [n for n in names if n.startswith("word/media/")]
        assert media, "no media files found in docx; plot was not embedded"


class TestPptxEmbedding:
    def test_pptx_includes_image(self, regression_with_plot, tmp_path):
        pytest.importorskip("pptx")
        out = tmp_path / "with_forest.pptx"
        regression_with_plot.to_pptx(out)
        import zipfile
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
            media = [n for n in names if n.startswith("ppt/media/")]
        assert media, "no media files found in pptx; plot was not embedded"


class TestLatexSidecar:
    def test_to_latex_file_writes_sidecar_pdf(self, regression_with_plot, tmp_path):
        out = tmp_path / "with_forest.tex"
        regression_with_plot.to_latex_file(out)
        assert out.exists()
        pdf = tmp_path / "with_forest_plot.pdf"
        assert pdf.exists()
        # tex source should reference the sidecar
        tex_src = out.read_text()
        assert "with_forest_plot.pdf" in tex_src
        assert r"\includegraphics" in tex_src

    def test_to_latex_string_no_image_command_when_no_basename(
        self, regression_with_plot,
    ):
        # to_latex() returns a string; without to_latex_file, no sidecar is
        # written and \includegraphics should not appear.
        out = regression_with_plot.to_latex()
        assert r"\includegraphics" not in out
