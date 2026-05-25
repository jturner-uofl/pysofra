#!/usr/bin/env python3
"""Generate the README hero images.

Renders three signature tables from the showcase narrative and saves
them as PNGs under ``assets/readme/``. The README embeds those PNGs so
the GitHub front page actually *shows* what PySofra produces instead
of just describing it.

Two pieces of plumbing worth noting:

* ``SofraTable.to_image()`` renders only the table grid — it does NOT
  composite the inline forest plot or KM curve carried by
  ``with_forest_plot()`` / ``with_km_plot()``. To get the full visual
  punch in the README, we render the table separately, pull
  ``tbl.inline_plot.png_bytes`` out, and stitch them vertically with
  Pillow so the hero image matches what the user actually sees in
  Jupyter / the showcase HTML.
* The 500-row synthetic dataset is the same one the showcase uses,
  so the rendered numbers match the showcase outputs exactly.

Run:

    .venv/bin/python scripts/generate_readme_assets.py

Re-run after any change that affects the rendered look of Table 1,
the adjusted-OR regression, or the Kaplan-Meier survival output.
"""
from __future__ import annotations

import io
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from PIL import Image

import pysofra as ps

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "assets" / "readme"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Fabricated 500-row two-arm trial — same simulator as the showcase.
# ---------------------------------------------------------------------------
def make_df() -> pd.DataFrame:
    rng = np.random.default_rng(2026_05_20)
    n = 500
    df = pd.DataFrame({
        "arm":    rng.choice(["Placebo", "Drug X"], n, p=[0.5, 0.5]),
        "age":    rng.normal(62, 11, n).round(1),
        "sex":    rng.choice(["Female", "Male"], n, p=[0.55, 0.45]),
        "race":   rng.choice(["White", "Black", "Asian", "Other"], n,
                              p=[0.60, 0.20, 0.15, 0.05]),
        "bmi":    rng.normal(31, 5, n).round(1),
        "smoker": rng.choice([0, 1], n, p=[0.70, 0.30]),
        "hba1c":  rng.lognormal(np.log(7.4), 0.15, n).round(2),
        "ldl_baseline": rng.normal(135, 28, n).round(1),
    })
    linpred = (
        -2.5
        + 0.04 * (df["age"] - 60)
        + 0.7 * (df["hba1c"] - 7.0)
        + 0.5 * df["smoker"]
        - 1.0 * (df["arm"] == "Drug X")
    )
    df["event_12mo"] = (rng.uniform(0, 1, n)
                        < 1 / (1 + np.exp(-linpred))).astype(int)
    hazard = np.exp(linpred) * 0.04
    df["time_mo"] = np.minimum(rng.exponential(1 / hazard), 12.0).round(2)
    df.loc[rng.choice(df.index, 15, replace=False), "bmi"] = np.nan
    return df


def stitch_plot_above_table(table_png_path: Path, plot_png_bytes: bytes,
                             out_path: Path, *, gap_px: int = 24,
                             bg: tuple = (255, 255, 255, 255)) -> None:
    """Vertically composite a plot on top of a table image.

    Mirrors what ``with_forest_plot(position='above')`` does in the
    HTML renderer — table renderers are independent so we have to
    stitch by hand for the static PNG case.
    """
    table_img = Image.open(table_png_path).convert("RGBA")
    plot_img = Image.open(io.BytesIO(plot_png_bytes)).convert("RGBA")

    # Match plot width to table width (preserving aspect ratio).
    target_w = table_img.width
    if plot_img.width != target_w:
        new_h = int(plot_img.height * (target_w / plot_img.width))
        plot_img = plot_img.resize((target_w, new_h), Image.LANCZOS)

    total_h = plot_img.height + gap_px + table_img.height
    canvas = Image.new("RGBA", (target_w, total_h), bg)
    canvas.paste(plot_img, (0, 0), plot_img)
    canvas.paste(table_img, (0, plot_img.height + gap_px), table_img)
    # Flatten to RGB before saving so file is smaller.
    Image.alpha_composite(
        Image.new("RGBA", canvas.size, bg), canvas,
    ).convert("RGB").save(out_path, "PNG", optimize=True)


def main() -> int:
    df = make_df()
    df_fit = df.dropna(subset=["bmi"]).copy()

    tmp = Path(tempfile.mkdtemp())

    # ────────────────────────────────────────────────────────────────
    # 1. Baseline characteristics — the canonical Table 1.
    # ────────────────────────────────────────────────────────────────
    print("rendering Table 1...")
    (
        ps.tbl_one(
            df, by="arm",
            variables=["age", "sex", "race", "bmi", "smoker",
                        "hba1c", "ldl_baseline"],
            nonnormal=["hba1c"],
            labels={
                "age": "Age (years)",
                "bmi": "BMI (kg/m²)",
                "smoker": "Current smoker",
                "hba1c": "HbA1c (%)",
                "ldl_baseline": "LDL-C, baseline (mg/dL)",
            },
        )
        .add_p().add_smd().add_overall()
        .theme("jama")
        .set_caption("Baseline characteristics, by treatment arm")
        .to_image(OUT / "table_one.png")
    )

    # ────────────────────────────────────────────────────────────────
    # 2. Adjusted-OR regression with inline forest plot.
    # ────────────────────────────────────────────────────────────────
    print("rendering regression (table + forest plot)...")
    X = sm.add_constant(
        pd.concat([
            df_fit[["age", "bmi", "hba1c"]],
            pd.get_dummies(df_fit["sex"], drop_first=True, dtype=float),
            pd.get_dummies(df_fit["arm"], drop_first=False,
                            dtype=float)[["Drug X"]],
        ], axis=1)
    )
    fit = sm.Logit(df_fit["event_12mo"], X.astype(float)).fit(disp=False)
    tbl_reg = (
        ps.tbl_regression(
            fit, exponentiate=True,
            labels={"age": "Age (years)", "bmi": "BMI (kg/m²)",
                    "hba1c": "HbA1c (%)", "Male": "Sex (Male)",
                    "Drug X": "Drug X (vs Placebo)"},
        )
        .bold_p(threshold=0.05)
        .with_forest_plot(null_line=1.0)
        .theme("clinical")
        .set_caption("Adjusted odds ratios for the composite primary event")
    )
    tbl_reg.to_image(tmp / "reg_table.png")
    stitch_plot_above_table(
        table_png_path=tmp / "reg_table.png",
        plot_png_bytes=tbl_reg.inline_plot.png_bytes,
        out_path=OUT / "regression_forest.png",
    )

    # ────────────────────────────────────────────────────────────────
    # 3. KM survival with curve.
    # ────────────────────────────────────────────────────────────────
    print("rendering survival (table + KM curve)...")
    tbl_surv = (
        ps.tbl_survival(
            df, time="time_mo", event="event_12mo", by="arm",
            times=[3, 6, 9, 12],
            labels={"arm": "Treatment"},
        )
        .with_km_plot()
        .theme("clinical")
        .set_caption("Event-free survival, by treatment arm")
    )
    tbl_surv.to_image(tmp / "surv_table.png")
    stitch_plot_above_table(
        table_png_path=tmp / "surv_table.png",
        plot_png_bytes=tbl_surv.inline_plot.png_bytes,
        out_path=OUT / "survival_km.png",
    )

    print()
    for p in sorted(OUT.glob("*.png")):
        kb = p.stat().st_size // 1024
        im = Image.open(p)
        print(f"  - {p.relative_to(REPO)}  ({im.size[0]}x{im.size[1]} px, {kb} KB)")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raise SystemExit(main())
