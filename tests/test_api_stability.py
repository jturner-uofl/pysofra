"""Public API-stability snapshot.

Freezes the public surface of ``pysofra`` so any unintended rename,
removal, or signature change surfaces as a failed test instead of a
silent breakage downstream.

If you intentionally change the API, update the constants in this file
**and** bump the project version per semver.
"""

from __future__ import annotations

import inspect

import pysofra as ps
from pysofra.core.table import SofraTable

# ----------------------------------------------------------------------
# Public top-level names
# ----------------------------------------------------------------------
EXPECTED_PUBLIC_NAMES = frozenset({
    # core types
    "CellPart",
    "SofraTable",
    "SurveyDesign",
    # builders
    "tbl_one",
    "tbl_summary",
    "tbl_cross",
    "tbl_regression",
    "tbl_uvregression",
    "tbl_survival",
    # composition
    "tbl_merge",
    "tbl_stack",
    # effect sizes
    "cohen_d",
    "hedges_g",
    "eta_squared",
    "omega_squared",
    "cramers_v",
    "phi_coefficient",
    "auto_effect_size",
    # calibration / survey
    "rake",
    "post_stratify",
    "design_effect",
    # MI pooling
    "pool",
    # themes
    "available_themes",
    "register_theme",
    # introspection
    "available_tests",
})


def test_public_api_surface():
    """Every expected public name must exist on the package."""
    for name in EXPECTED_PUBLIC_NAMES:
        assert hasattr(ps, name), f"missing public name: {name}"

    # The package should not have *removed* any of the expected names
    # in this release — i.e. EXPECTED_PUBLIC_NAMES is a strict subset of
    # what's currently exported.
    actual_names = {n for n in dir(ps) if not n.startswith("_")}
    missing = EXPECTED_PUBLIC_NAMES - actual_names
    assert not missing, f"public names disappeared: {missing}"


# ----------------------------------------------------------------------
# Signature snapshots for the most user-facing builders
# ----------------------------------------------------------------------
def _params(fn):
    """Return a tuple of (name, kind, default-marker) for each parameter."""
    out = []
    for p in inspect.signature(fn).parameters.values():
        default = p.default if p.default is not inspect.Parameter.empty else "<no_default>"
        out.append((p.name, str(p.kind), repr(default)))
    return tuple(out)


def test_tbl_one_signature_stable():
    """The argument list for ``tbl_one`` is part of the user contract."""
    expected_names = (
        "data", "by", "variables", "labels", "types", "nonnormal",
        "tests", "weights", "design", "digits", "pct_digits",
        "missing", "include_missing",
    )
    actual = tuple(p[0] for p in _params(ps.tbl_one))
    assert actual == expected_names, (
        f"tbl_one signature drifted:\n  expected {expected_names}\n  actual   {actual}"
    )


def test_tbl_survival_signature_stable():
    expected = (
        "data", "time", "event", "by", "times", "times_label",
        "conf_level", "digits", "pct_digits", "labels", "show_logrank",
        "weights",
    )
    actual = tuple(p[0] for p in _params(ps.tbl_survival))
    assert actual == expected


def test_tbl_regression_signature_stable():
    expected = (
        "model", "exponentiate", "conf_level", "digits", "labels",
        "intercept", "estimate_label", "model_labels", "design", "data",
    )
    actual = tuple(p[0] for p in _params(ps.tbl_regression))
    assert actual == expected


def test_tbl_uvregression_signature_stable():
    expected = (
        "data", "outcome", "predictors", "method", "method_kwargs",
        "adjust_for", "exponentiate", "conf_level", "digits", "labels",
    )
    actual = tuple(p[0] for p in _params(ps.tbl_uvregression))
    assert actual == expected


def test_survey_design_dataclass_fields():
    """SurveyDesign field order is part of the API."""
    expected_fields = (
        "weights", "strata", "cluster", "fpc",
        "replicate_weights", "replicate_type",
    )
    actual = tuple(f.name for f in
                   ps.SurveyDesign.__dataclass_fields__.values())
    assert actual == expected_fields


# ----------------------------------------------------------------------
# SofraTable user-facing methods + dataclass attributes.
#
# Both sets are part of the documented public API: methods because
# users call them in their reporting code, attributes because they
# appear in the documented schema (README and notebook examples
# enumerate ``rows``, ``headers``, ``footnotes``, etc.).
# ----------------------------------------------------------------------
EXPECTED_SOFRATABLE_METHODS = frozenset({
    # presentational
    "set_caption", "with_footnotes", "with_inline_svg",
    "with_forest_plot", "with_km_plot",
    "theme",
    "with_pvalue_fmt", "with_estimate_fmt",
    "add_footnote", "modify_spanning_header", "compose",
    # statistical modifiers
    "add_overall", "add_p", "add_q", "add_smd", "add_n",
    "add_stat_label", "add_significance_stars",
    "add_global_p", "add_difference", "add_ci",
    # conditional formatting
    "bold_p", "bold_if", "highlight_if", "style_if", "color_scale_if",
    # layout / fit
    "autofit",
    # introspection
    "inline_text",
    # rendering
    "to_html", "to_markdown", "to_latex", "to_latex_file",
    "to_docx", "to_pptx", "to_xlsx", "to_image",
    "to_quarto", "to_typst", "to_typst_file",
    "to_dict",
    # reproducibility — snapshot lock (0.1.0a10)
    "snapshot_hash", "lock_snapshot", "assert_snapshot",
    # publication safety (0.1.0a10)
    "check_safety", "with_safety_warnings",
})

EXPECTED_SOFRATABLE_ATTRIBUTES = frozenset({
    "rows", "headers", "spanning_headers", "footnotes",
    "caption", "theme_name",
    "inline_plot", "inline_svg", "inline_svg_position",
})


def test_sofratable_method_surface():
    """Every modifier / renderer must remain a public method of SofraTable."""
    methods = {m for m in dir(SofraTable) if not m.startswith("_")}
    missing = EXPECTED_SOFRATABLE_METHODS - methods
    assert not missing, f"SofraTable lost methods: {missing}"


def test_sofratable_attribute_surface():
    """Every documented dataclass attribute must remain on SofraTable."""
    attrs = {m for m in dir(SofraTable) if not m.startswith("_")}
    missing = EXPECTED_SOFRATABLE_ATTRIBUTES - attrs
    assert not missing, f"SofraTable lost attributes: {missing}"


def test_sofratable_no_undocumented_public_surface():
    """
    Reject silently-added public names. Anything new must be added to
    one of the two expected-sets above (so downstream users see the API
    change in the diff).
    """
    public = {m for m in dir(SofraTable) if not m.startswith("_")}
    documented = EXPECTED_SOFRATABLE_METHODS | EXPECTED_SOFRATABLE_ATTRIBUTES
    undocumented = sorted(public - documented)
    assert not undocumented, (
        "SofraTable grew undocumented public names; add to either "
        "EXPECTED_SOFRATABLE_METHODS or EXPECTED_SOFRATABLE_ATTRIBUTES: "
        f"{undocumented}"
    )


def test_sofratable_to_image_signature_stable():
    """The PNG renderer's public kwargs are part of the documented public API."""
    actual = tuple(p[0] for p in _params(SofraTable.to_image))
    assert actual == ("self", "path", "scale", "dpi")


# ----------------------------------------------------------------------
# Behavioural contracts on top of the structural snapshot.
#
# These guarantees are part of the *user-visible* maturity contract:
#
# * builders construct SofraTable instances (not bare dicts / Styler /
#   strings), so downstream renderers and modifier chains always have a
#   uniform target;
# * every modifier method is copy-on-write — it returns a NEW
#   SofraTable, never ``self`` and never ``None`` — so chained
#   pipelines cannot silently mutate an earlier result;
# * every public symbol carries a docstring (so ``help(...)`` and the
#   docs site never have to apologise for an empty entry);
# * a representative end-to-end build emits no DeprecationWarning or
#   PendingDeprecationWarning whose source frame is inside the pysofra
#   package — i.e. nothing the user calls today is on a quiet
#   removal timer.
# ----------------------------------------------------------------------
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _toy_frame(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    return pd.DataFrame({
        "arm": rng.choice(["A", "B"], n),
        "age": rng.normal(45.0, 12.0, n),
        "sex": rng.choice(["M", "F"], n),
    })


def test_tbl_one_returns_sofratable():
    """``tbl_one`` returns a SofraTable, not a Styler / DataFrame / str."""
    t = ps.tbl_one(_toy_frame(), by="arm")
    assert isinstance(t, SofraTable)


def test_tbl_summary_returns_sofratable():
    t = ps.tbl_summary(_toy_frame())
    assert isinstance(t, SofraTable)


def test_tbl_cross_returns_sofratable():
    t = ps.tbl_cross(_toy_frame(), row="sex", column="arm")
    assert isinstance(t, SofraTable)


# Modifiers we exercise. Each takes a built table and (no args) should
# return a *new* SofraTable. We pick the ones that work without extra
# inputs; the rest are signature-locked above.
_COPY_ON_WRITE_MODIFIERS: tuple[str, ...] = (
    "add_p",
    "add_overall",
    "add_smd",
    "add_n",
    "add_stat_label",
    "add_significance_stars",
    "bold_p",
    "autofit",
)


def test_modifiers_are_copy_on_write():
    """Every chainable modifier must return a NEW SofraTable.

    The immutability contract is the spine of PySofra's pipeline model —
    if any modifier ever returned ``self`` or ``None`` it would silently
    break user code that relies on ``t2 = t.add_p()`` leaving ``t``
    untouched.
    """
    base = ps.tbl_one(_toy_frame(), by="arm")
    for name in _COPY_ON_WRITE_MODIFIERS:
        method = getattr(base, name)
        out = method()
        assert out is not None, f"{name}() returned None"
        assert isinstance(out, SofraTable), (
            f"{name}() returned {type(out).__name__}, not SofraTable"
        )
        assert out is not base, (
            f"{name}() returned ``self`` — modifiers must be "
            f"copy-on-write so chained pipelines don't mutate earlier "
            f"results"
        )


def test_public_top_level_symbols_have_docstrings():
    """Every public top-level name documents itself.

    Empty docstrings on the public API are a maturity smell — they show
    up in ``help(ps.tbl_one)`` and in the rendered docs site as
    apologetic blanks. Pin them as non-empty here.
    """
    blanks = [n for n in EXPECTED_PUBLIC_NAMES
              if not (getattr(ps, n).__doc__ or "").strip()]
    assert not blanks, f"public symbols missing docstrings: {blanks}"


def test_sofratable_public_methods_have_docstrings():
    """Every public SofraTable method documents itself."""
    blanks = [m for m in EXPECTED_SOFRATABLE_METHODS
              if not (getattr(SofraTable, m).__doc__ or "").strip()]
    assert not blanks, (
        f"SofraTable public methods missing docstrings: {blanks}"
    )


def test_representative_build_emits_no_pysofra_deprecation():
    """A representative end-to-end build emits no DeprecationWarning
    (or PendingDeprecationWarning) whose source is inside the pysofra
    package.

    We filter on filename so that any deprecation from upstream
    libraries (pandas, numpy, statsmodels) — which we cannot control —
    does not fail this contract. The check is: *nothing the user calls
    in pysofra today is on a quiet removal timer.*
    """
    df = _toy_frame()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        t = (ps.tbl_one(df, by="arm")
             .add_p()
             .add_overall()
             .add_smd())
        # Exercise every text-renderer once.
        _ = t.to_html()
        _ = t.to_markdown()
        _ = t.to_latex()
    pysofra_deps = [
        w for w in ws
        if issubclass(w.category, (DeprecationWarning,
                                   PendingDeprecationWarning))
        and "pysofra" in (w.filename or "")
    ]
    assert not pysofra_deps, (
        "pysofra-originated deprecation warning(s) on representative "
        "build:\n  "
        + "\n  ".join(f"{w.category.__name__} {w.filename}: "
                       f"{w.message}" for w in pysofra_deps)
    )
