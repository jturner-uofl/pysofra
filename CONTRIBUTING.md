# Contributing to PySofra

Thanks for your interest in improving PySofra. This document covers
how to file issues, propose changes, set up a development environment,
and pass the quality gates before opening a pull request.

## Code of conduct

Participation in this project — issues, pull requests, discussions — is
governed by the [Code of Conduct](CODE_OF_CONDUCT.md). By
participating, you agree to uphold those standards.

## Filing an issue

Please open an issue on GitHub before you start non-trivial work, so
we can scope it together. Helpful issues include:

* **Bug reports** — please attach a minimal reproducible example, the
  output you observed, the output you expected, and the versions of
  PySofra, Python, and the optional extras you're using
  (`pysofra --version`, `python -V`, `pip list | grep -E
  'lifelines|matplotlib|polars|statsmodels'`).
* **Feature requests** — describe the use case (preferably with a
  reference to the equivalent R workflow) and an outline of the
  proposed API.
* **Documentation gaps** — point to the page or symbol that confused
  you and what you wanted to know.

## Local setup

PySofra uses [`uv`](https://github.com/astral-sh/uv) for fast
environment management, but any modern Python ≥ 3.11 toolchain will
work.

```bash
git clone https://github.com/jturner-uofl/pysofra
cd pysofra
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,all]"
```

The `[dev,all]` extras pull every optional dependency (lifelines,
matplotlib, polars, scikit-learn, xlsxwriter, python-pptx) so that
the full test suite runs locally.

## Running the quality gates

Every pull request must pass:

```bash
pytest --cov=pysofra --cov-fail-under=100   # tests + strict coverage
mypy src/pysofra                            # mypy strict, zero issues
ruff check src tests                        # lint clean
mkdocs build --strict                       # docs build with no warnings
```

CI runs the same four gates on Python 3.11 and 3.12, Ubuntu + macOS.

For visual changes to the tutorial:

```bash
python scripts/render_tutorial.py
```

re-executes the notebook and re-renders the polished HTML.

## Style guide

* **Formatting**: ruff defaults (PEP 8, `E`, `F`, `I`, `UP`, `B`, `SIM`
  rule families). `line-length = 100`.
* **Types**: every public function and method has type hints; mypy
  strict is the bar.
* **Docstrings**: every public symbol has a NumPy-style docstring.
  Include a `Notes` section when the statistical convention is
  non-obvious (e.g. Welch's df, Wilson-score CI, Newcombe interval).
* **Tests**: every code path has a test. Bug-fix PRs include a test
  that fails without the fix.
* **No emojis in source files** (docstrings, comments, code). Emojis
  in `README.md` and tutorial markdown are fine.

## Proposing changes

1. Fork the repo and create a topic branch off `main`:
   `git checkout -b fix/short-description`.
2. Make focused commits with clear messages
   ([Conventional Commits](https://www.conventionalcommits.org/) is
   nice but not required).
3. Run the quality gates locally.
4. Open a pull request describing **what** changed, **why**, and
   referencing any related issue.
5. Add a `CHANGELOG.md` entry under `## [Unreleased]`.

## Adding a new feature

If your feature has a counterpart in R's `tableone` / `gtsummary` /
`flextable`, please link to the relevant R documentation in your PR
description. PySofra deliberately mirrors those workflows where the
mapping is natural; the cross-reference helps maintainers verify
behavioural parity.

For new statistical estimators, the PR must include:

1. A docstring with the exact formula and at least one literature
   reference.
2. A unit test that validates the output against an independent
   reference (scipy, statsmodels, lifelines, R, or a hand-computed
   textbook value).
3. A property-based test (Hypothesis) if the estimator has a clear
   invariant (e.g. CI bounds ordered, p-value ∈ [0, 1]).

## Reporting security issues

Security-sensitive issues (e.g. anything involving untrusted-input
parsing, dependency confusion, or malicious file embedding) should be
reported by email to **jason.turner@louisville.edu** with the subject
prefix `[PySofra security]`, **not** as a public GitHub issue.

## Becoming a maintainer

We welcome long-term contributors. After a few merged PRs and active
participation in issue triage, please email or open a discussion to
ask about commit access.

## License

By contributing to PySofra you agree that your contributions will be
licensed under the project's
[GNU GPL v3.0-or-later](LICENSE).
