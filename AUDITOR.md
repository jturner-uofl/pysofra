# External Auditor's Guide

This document is the canonical recipe for an **external auditor** —
a reviewer, sceptic, or downstream user — to independently verify
PySofra's published claims end-to-end. The audit involves no special
tooling, no Docker, no maintainer cooperation, and no trust in the
maintainer's own CI: you reproduce the same case-study notebook that
GitHub Actions runs on every commit, with the exact same pinned
dependency versions, against the same publicly downloadable NHANES
data, and the audit terminates with a single canonical success line.

If the audit passes, you have independently verified:

- **53 numbered contracts** spanning numerical agreement with R
  `survey`, lifelines, scipy, statsmodels; structural correctness of
  every public builder, modifier, and renderer; reproducibility via
  byte-deterministic output and content-hash snapshot locks;
  cross-backend semantic-content preservation across HTML / LaTeX /
  Typst / Markdown; honest documentation of every known limitation.
- **1032 unit / property / integration tests**, including the
  API-stability snapshot that freezes the entire public surface.
- That PyPI and GitHub are in sync: the notebook's first executable
  cell hard-asserts `pysofra.__version__ == "0.1.0a17"` against the
  release you installed from PyPI.

## TL;DR — the audit, in one command block

```bash
# 1. Get the audit artefacts (notebook + pinned environment).
git clone https://github.com/jturner-uofl/pysofra.git
cd pysofra
git checkout v0.1.0a17    # the tag matching the PyPI release

# 2. Create an isolated environment.
python -m venv .venv-audit
source .venv-audit/bin/activate

# 3. Install the exact PyPI release + the exact CI-resolved dep set.
pip install --upgrade pip
pip install pysofra==0.1.0a17
pip install -r requirements-audit.txt
pip install jupyter nbconvert ipykernel   # not in requirements

# 4. Run the audit notebook end-to-end.
jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 \
  examples/jss_case_study/jss_case_study.ipynb
```

If the final command exits 0 and the notebook's last cell prints

```
AUDIT COMPLETE — 51/51 contracts passed | pysofra 0.1.0a17 | <UTC>
```

the audit succeeded. If any contract failed, `nbconvert` exits non-zero
and the failing cell contains an explicit `AssertionError` with a
diagnostic message naming the contract and the gap observed.

## Recommended workflows

There are two reasonable audit depths. Pick whichever matches the time
you have:

### Read-only (5 minutes)

You do **not** need to install anything. The repository ships the
**pre-executed** notebook rendered as HTML so you can read every step,
every print statement, and every assertion outcome without running
code:

- [`examples/jss_case_study/jss_case_study.html`](examples/jss_case_study/jss_case_study.html)

This is what CI produced on the latest commit to `main`. Look for
`ASSERTION OK` after each step and the final `AUDIT COMPLETE` banner.

### End-to-end reproduction (20-30 minutes)

Run the TL;DR block above. This is the canonical audit. The notebook
downloads NHANES 2017-2018 directly from the CDC (≈40 MB across seven
files), fits real models, and cross-checks every numerical claim
against the reference library named in that step's contract.

If you also want to verify the **R `survey` package agreement** at
Step 12 / 38 / 39 / 40, install R 4.4+ with packages `survey`,
`gtsummary`, and `broom`, then run:

```bash
Rscript examples/jss_case_study/r_validation.R > /tmp/r_out.txt
# the notebook re-reads /tmp/r_out.txt and re-asserts equality.
```

The exact same script runs in CI's `case-study` job — see
`.github/workflows/tests.yml`.

## What's checked (contract index)

The 51 numbered contracts split across two categories. Each one is
named in the Summary table at the end of the notebook with its
tolerance, observed gap, and pass status.

| Group | Steps | What it proves |
| ----- | ----- | -------------- |
| Section 0 — pre-flight | 0 | NHANES data loaded with correct row counts. |
| Section I — Table 1 + reweighting | 1-7 | Continuous / categorical dispatch, design-aware p-values, MI pooling, regression refits agree with statsmodels. |
| Section II — Mathematical foundations | 8-12, 19-26 | Wilson / Newcombe CIs, lifelines KM equality, scipy Welch t df, R `survey::svymean` / `svyttest` / `svyglm` agreement, Rubin (1987) pooling. |
| Section III — Robustness | 13-18, 27-32 | AFT TR-vs-HR labelling, weighted KM = lifelines, polars = pandas markdown, permutation invariance, method-chain integrity, byte-deterministic renderers. |
| Section IV — Reproducibility tools | 33-37 | Snapshot lock, publication-safety checker, Quarto / Typst / CLI surface. |
| Section VI — R `survey` parity | 38-40 | First-order Rao–Scott chi-square documented gap; design-based sandwich SE / CI / p matches R `svyglm` to numerical precision. |
| Section VII — Negative controls | 41-43 | Wrong weight column / wrong strata / `freq_weights` inflate as expected — the design adjustments are *responsive*, not cosmetic. |
| Section VIII — Sensitivity | 44-46 | MI pooling stable in m, complete-case vs MI side-by-side, three outcome definitions side-by-side. |
| Section IX — Coverage + asymmetry | 47-48 | Monte Carlo coverage of design-based CIs in [92 %, 97 %]; exponentiated CIs preserve `(exp(β_lo), exp(β_hi))`. |
| **Section X — Maturity contracts** | **49-53** | **Public-API surface manifest + copy-on-write modifiers + docstring coverage + zero-pysofra-DeprecationWarning; one spec → 4 backends with every numeric token preserved; `Cell.value` typed provenance + `bold_p` queries the float; declarative-vs-pandas error-surface comparison; the three documented limitations each surface a renderer-level footnote.** |

## Verifying the version sync (PyPI ↔ GitHub)

The very first executable cell of the notebook asserts:

```python
EXPECTED_PYSOFRA_VERSION = "0.1.0a17"
assert ps.__version__ == EXPECTED_PYSOFRA_VERSION, (...)
```

This is the canonical sync check. The notebook will refuse to run if
the `pysofra` your `pip install` resolved is not the exact version
this tagged commit was authored against. If you see a version-drift
error, install the exact release:

```bash
pip install pysofra==0.1.0a17
```

## Independently inspectable evidence

The audit does not ask you to trust the executed notebook alone.
Several orthogonal artefacts cross-check each other:

| Artefact | Location | What it cross-checks |
| -------- | -------- | -------------------- |
| Test suite | `tests/` (1032 tests) | Same statistical claims as the notebook, but in unit-test form (`pytest -q`). |
| API-stability snapshot | `tests/test_api_stability.py` (17 tests) | Public surface is the exact 28 names + 45 methods documented. |
| Cross-backend consistency | `tests/test_cross_backend_consistency.py` (3 tests) | One spec, 4 backends, identical numeric payload. |
| R-survey parity | `examples/jss_case_study/r_validation.R` | Verifiable in R independently of pysofra. |
| CI workflow | `.github/workflows/tests.yml` | The exact recipe the maintainer's CI uses — readable; no hidden steps. |
| Documented scope | `docs/concepts/limitations.md` | Names every approximation and its workaround. |
| Stability policy | `docs/concepts/stability.md` | Names what the public contract guarantees and the post-1.0 deprecation ladder. |

## Reporting a discrepancy

If any contract fails — or if you find a claim in the notebook that
your environment cannot reproduce — please open an issue tagged
`audit-failure` at https://github.com/jturner-uofl/pysofra/issues
with: the step number, the assertion text, your Python version, your
output of `pip freeze`, and the full traceback. These are treated as
release-blocking.

## Citation

To cite the audited release in academic work:

> Turner, J. PySofra: statistical reporting and table preparation
> framework for Python (Version 0.1.0a17) [Software]. 2026.
> https://github.com/jturner-uofl/pysofra
