# Changelog

All notable changes to PySofra will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0a4] — 2026-05-25

### Added
- Input validation for duplicate names in `variables=` (now raises
  `ValueError` instead of silently accepting duplicates).
- Confidence-level range check in `.add_ci()` and related modifiers
  (must lie in `(0, 1)`).

### Changed
- Renamed several test files for clarity. No public API changes.

## [0.1.0a3] — 2026-05-24

### Changed
- Documentation polish across README, changelog, and inline docstrings.
  No public API or behavioural changes.

## [0.1.0a2] — 2026-05-23

### Fixed
- Theme styling now survives notebook viewers that strip `<style>` blocks
  (e.g. GitHub's notebook viewer). Critical theme properties (font, border,
  padding) are emitted as inline `style` attributes on each table element, so
  `jama` vs `nejm` vs `clinical` vs `minimal` stay visibly distinct everywhere.
- README image and link URLs are now absolute so they render on PyPI.

## [0.1.0a1] — 2026-05-20

### Added

- Initial alpha release.
- Core `SofraTable` object with immutable method chaining.
- `tbl_one()` — baseline characteristic tables (Table 1) with continuous /
  categorical summaries, stratification, missing data summaries, overall
  column, p-values, and standardized mean differences (SMDs).
- `tbl_summary()` — general descriptive summary tables with grouping and
  configurable statistics.
- `tbl_regression()` — regression tables for `statsmodels` linear / logistic
  / Poisson models, with confidence intervals, exponentiation, and p-values.
- `tbl_merge()` / `tbl_stack()` — table composition.
- HTML renderer with rich notebook `_repr_html_` output (dark-mode aware,
  responsive, sticky headers).
- Markdown renderer.
- DOCX renderer via `python-docx` (publication-quality Word tables with
  captions, footnotes, merged spanning headers).
- Themes: `clinical`, `compact`, `jama`, `nejm`, `minimal`.
- Automatic statistical test selection with override hooks.
- Snapshot tests for HTML output.
