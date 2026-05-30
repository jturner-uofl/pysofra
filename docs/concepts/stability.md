# API stability & deprecation policy

PySofra commits to a precise, machine-checked stability contract for its
public surface. This page documents what is guaranteed, what is
provisional, and how a future breaking change would be staged.

## What is "public"

The public surface is everything re-exported from the top-level
`pysofra` namespace (28 names — see `pysofra.__all__`) plus the
documented public methods and attributes of `pysofra.SofraTable`. The
exact set is **frozen in code** by
[`tests/test_api_stability.py`](https://github.com/jturner-uofl/pysofra/blob/main/tests/test_api_stability.py),
which fails CI if any documented name disappears, gains an undocumented
sibling, or has its call signature drift.

Anything reachable through a name starting with `_` (e.g.
`pysofra.core.table._rebuild`, `SofraTable._spec`) is **internal**. It
may change between any two releases without notice. Please do not rely
on it; if you need a capability that only an internal exposes, please
open an issue so we can promote a public counterpart.

## What the contract guarantees

The API-stability test suite locks in **five** structural and behavioural
contracts:

| Contract                       | Pinned by                                          |
| ------------------------------ | -------------------------------------------------- |
| Top-level name set             | `EXPECTED_PUBLIC_NAMES` (28 names)                 |
| Builder call signatures        | `test_tbl_one_signature_stable`, etc.              |
| `SofraTable` method set        | `EXPECTED_SOFRATABLE_METHODS` (~45 methods)        |
| `SofraTable` attribute set     | `EXPECTED_SOFRATABLE_ATTRIBUTES` (9 attributes)    |
| `SurveyDesign` dataclass shape | `test_survey_design_dataclass_fields` (6 fields)   |

On top of the structural lock, four **behavioural** contracts are
asserted on every test run:

1. **Builders return `SofraTable`** — `tbl_one`, `tbl_summary`,
   `tbl_cross` all yield a `SofraTable` instance (not a `pandas.Styler`,
   not a `DataFrame`, not a string). Downstream code can always chain
   `.add_p().to_html()` without type-sniffing.
2. **Modifiers are copy-on-write** — every chainable modifier
   (`add_p`, `add_overall`, `add_smd`, `add_n`, `add_stat_label`,
   `add_significance_stars`, `bold_p`, `autofit`, …) returns a *new*
   `SofraTable`. It is never `self` and never `None`. Pipelines never
   silently mutate an earlier result.
3. **Public surface is docstring-complete** — every name in
   `pysofra.__all__` and every public method of `SofraTable` carries a
   non-empty docstring. `help(...)` and the rendered docs site never
   produce apologetic blanks.
4. **No pysofra-originated deprecation warnings on a representative
   build** — a `tbl_one → add_p → add_overall → add_smd → to_html /
   to_markdown / to_latex` pipeline emits zero `DeprecationWarning` or
   `PendingDeprecationWarning` whose source frame is inside the
   `pysofra` package. Nothing the user calls today is on a quiet
   removal timer.

## Versioning

PySofra adheres to [Semantic Versioning 2.0](https://semver.org/). The
current release line is the **0.1.0a*** alpha series. While the
foundation (immutable spec, multi-backend renderers, statistical
abstractions) is locked down by the contract above, the package is
explicitly **pre-stable**: minor numeric defaults, prose wording, and
non-public internals may change between alphas without further notice.

When 1.0.0 ships, the alpha contract becomes the long-term contract and
the deprecation ladder below activates in full.

## Deprecation ladder (post-1.0)

A breaking change to any name on the public surface will progress
through three releases:

1. **Soft deprecation (X.Y.0)** — the name continues to work but emits
   a `DeprecationWarning` on first call, pointing at the replacement.
   Documentation is updated.
2. **Hard deprecation (X.(Y+1).0)** — the name continues to work but
   the warning is upgraded to `FutureWarning` and is no longer
   suppressible by the default warning filter. The CHANGELOG entry
   pins the removal release.
3. **Removal (X+1.0.0)** — the name is removed; `EXPECTED_PUBLIC_NAMES`
   is updated in the same commit; the major version bumps. There is
   never silent removal.

Equivalently: a user pinned to `pysofra >= X.0, < X+1.0` will never see
their code stop working without first seeing a warning that names the
replacement.

## Reporting an accidental break

If you see PySofra change a documented surface without going through
this ladder, please open an issue tagged `api-break`. We treat those as
release-blocking.
