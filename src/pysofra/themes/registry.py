"""Theme registry.

A theme is a :class:`Theme` instance carrying enough information for every
renderer to produce a consistent visual style. Renderers consume the theme
through three keyed dicts (``css``, ``docx``, ``pptx``); they do not parse
arbitrary CSS strings, so theme definitions stay small and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Theme:
    """A named visual theme.

    ``css`` is a mapping of semantic keys to CSS declarations; the HTML
    renderer assembles a scoped stylesheet from it. ``docx`` and ``pptx``
    carry the corresponding hints for the Word / PowerPoint renderers
    (font name, size, header shading, border weights, etc.).
    """

    name: str
    css: dict[str, dict[str, str]] = field(default_factory=dict)
    docx: dict[str, Any] = field(default_factory=dict)
    pptx: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Built-in themes
# ----------------------------------------------------------------------

_BASE_FONT = (
    '"Helvetica Neue", Helvetica, Arial, "Segoe UI", '
    '"Liberation Sans", sans-serif'
)

# Faded variant for separator borders and footnotes. Previously used CSS
# ``color-mix(in srgb, currentColor 25%, transparent)`` — readable in every
# *interactive* notebook frontend (Chrome ≥ 111, Safari ≥ 16.2,
# Firefox ≥ 113) but the GitHub.com .ipynb renderer's HTML sanitiser
# strips the parenthesised arguments mid-attribute, leaking raw CSS text
# into the rendered cell content. Since GitHub is the primary
# "browse the notebook without running it" surface for JSS reviewers,
# we use a fixed neutral grey (rgba 50 %) instead. Tradeoff: borders no
# longer follow the surrounding light/dark text colour, but the grey is
# legible on both. The numerical result is unchanged.
_FADED_25 = "rgba(127, 127, 127, 0.30)"
_FADED_70 = "rgba(127, 127, 127, 0.75)"

_DEFAULT = Theme(
    name="default",
    css={
        "table": {
            "border-collapse": "collapse",
            "font-family": _BASE_FONT,
            "font-size": "14px",
            "line-height": "1.45",
            # Inherit the surrounding text colour so we always have contrast
            # against the actual page background — no media-query rules
            # that compete with the host application's own theme.
            "color": "inherit",
            "background": "transparent",
            "margin": "0.75em 0",
        },
        "caption": {
            "caption-side": "top",
            "text-align": "left",
            "font-weight": "700",
            "padding": "0.4em 0.2em",
            "font-size": "15px",
            "color": "inherit",
        },
        "th": {
            "padding": "0.55em 0.85em",
            "text-align": "center",
            "border-top": "2px solid currentColor",
            "border-bottom": "1.25px solid currentColor",
            "font-weight": "700",
            "vertical-align": "bottom",
            "color": "inherit",
            "background": "transparent",
        },
        "td": {
            "padding": "0.4em 0.85em",
            "border-bottom": f"1px solid {_FADED_25}",
            "vertical-align": "top",
            "color": "inherit",
        },
        "tr:last-child td": {
            "border-bottom": "2px solid currentColor",
        },
        "tr.group-header td": {
            "font-weight": "700",
            "padding-top": "0.7em",
        },
        "tfoot td": {
            "font-size": "12px",
            "color": _FADED_70,
            "border-bottom": "none",
            "padding-top": "0.55em",
        },
        ".pysofra-num": {"text-align": "right", "font-variant-numeric": "tabular-nums"},
        ".pysofra-bold": {"font-weight": "700"},
        ".pysofra-indent": {"padding-left": "1.75em"},
        ".pysofra-spanning": {
            "border-bottom": "1px solid currentColor",
            "text-align": "center",
            "font-weight": "700",
            "padding": "0.35em 0.5em",
        },
    },
    docx={
        "font_name": "Calibri",
        "font_size": 10,
        "header_bold": True,
        "header_bottom_border": True,
        "outer_border": True,
        "row_zebra": False,
    },
    pptx={"font_name": "Calibri", "font_size": 14},
)


def _override(parent: Theme, name: str, css_overrides: dict[str, dict[str, str]],
              docx_overrides: dict[str, Any] | None = None,
              pptx_overrides: dict[str, Any] | None = None) -> Theme:
    new_css: dict[str, dict[str, str]] = {k: dict(v) for k, v in parent.css.items()}
    for k, v in css_overrides.items():
        new_css.setdefault(k, {}).update(v)
    new_docx = dict(parent.docx)
    if docx_overrides:
        new_docx.update(docx_overrides)
    new_pptx = dict(parent.pptx)
    if pptx_overrides:
        new_pptx.update(pptx_overrides)
    return Theme(name=name, css=new_css, docx=new_docx, pptx=new_pptx)


_CLINICAL = _override(
    _DEFAULT,
    "clinical",
    {
        "table": {"font-size": "14px"},
        "caption": {"font-size": "15px"},
        "th": {
            "border-top": "2.5px solid currentColor",
            "border-bottom": "1.5px solid currentColor",
        },
        "td": {"padding": "0.45em 0.9em"},
    },
    docx_overrides={"font_name": "Calibri", "font_size": 10, "header_bottom_border": True},
)

_COMPACT = _override(
    _DEFAULT,
    "compact",
    {
        "table": {"font-size": "13px"},
        "th": {"padding": "0.35em 0.6em"},
        "td": {"padding": "0.25em 0.6em"},
    },
    docx_overrides={"font_size": 9},
)

_JAMA = _override(
    _DEFAULT,
    "jama",
    {
        "table": {"font-family": '"Times New Roman", Times, serif', "font-size": "13.5px"},
        "caption": {
            "font-family": '"Times New Roman", Times, serif',
            "font-weight": "700",
            "font-size": "15px",
        },
        "th": {
            "border-top": "2.5px solid currentColor",
            "border-bottom": "1.5px solid currentColor",
            "background": "transparent",
        },
        # JAMA-style: no internal row separators; strong bottom rule only.
        "td": {"border-bottom": "none"},
        "tr:last-child td": {"border-bottom": "2px solid currentColor"},
        "tfoot td": {"font-family": '"Times New Roman", Times, serif'},
    },
    docx_overrides={"font_name": "Times New Roman", "font_size": 10, "outer_border": True},
)

_NEJM = _override(
    _DEFAULT,
    "nejm",
    {
        "table": {"font-family": '"Georgia", "Times New Roman", serif', "font-size": "13.5px"},
        "th": {
            "border-top": "2.5px solid currentColor",
            "border-bottom": "1.25px solid currentColor",
            "background": "transparent",
        },
        "td": {"border-bottom": "none", "padding": "0.35em 0.85em"},
        "tr:last-child td": {"border-bottom": "2px solid currentColor"},
    },
    docx_overrides={"font_name": "Georgia", "font_size": 10, "outer_border": True},
)

_MINIMAL = _override(
    _DEFAULT,
    "minimal",
    {
        "th": {
            "border-top": "none",
            "border-bottom": "1.25px solid currentColor",
            "background": "transparent",
        },
        "td": {"border-bottom": "none"},
        "tr:last-child td": {"border-bottom": "1.25px solid currentColor"},
    },
    docx_overrides={"header_bottom_border": True, "outer_border": False},
)


_THEMES: dict[str, Theme] = {
    "default": _DEFAULT,
    "clinical": _CLINICAL,
    "compact": _COMPACT,
    "jama": _JAMA,
    "nejm": _NEJM,
    "minimal": _MINIMAL,
}


def resolve_theme(name: str) -> Theme:
    """Resolve a theme name to a :class:`Theme`. Raises ``ValueError`` if unknown."""
    try:
        return _THEMES[name]
    except KeyError as exc:
        available = ", ".join(sorted(_THEMES))
        raise ValueError(f"Unknown theme {name!r}. Available themes: {available}") from exc


_BUILTIN_THEME_NAMES = frozenset(
    {"default", "clinical", "compact", "jama", "nejm", "minimal"}
)


def register_theme(theme: Theme, *, overwrite: bool = False) -> None:
    """Register a user-defined theme.

    By default this refuses to overwrite a built-in theme; pass
    ``overwrite=True`` to force it. Overwriting an existing user theme
    is allowed without the flag — the guard exists only to keep
    ``ps.tbl_one(...).theme('clinical')`` from silently rendering with a
    user replacement that doesn't match what the documentation says.
    """
    if theme.name in _BUILTIN_THEME_NAMES and not overwrite:
        raise ValueError(
            f"Theme {theme.name!r} is a built-in. "
            "Pass overwrite=True to replace it, or pick a different name."
        )
    _THEMES[theme.name] = theme


def available_themes() -> list[str]:
    """Return a sorted list of every registered theme name.

    Includes both the six built-in themes (``default``, ``clinical``,
    ``jama``, ``nejm``, ``compact``, ``minimal``) and any user themes
    added via :func:`register_theme`. Apply a theme with
    :meth:`~pysofra.SofraTable.theme`.

    Examples
    --------
    >>> import pysofra as ps
    >>> ps.available_themes()
    ['clinical', 'compact', 'default', 'jama', 'minimal', 'nejm']
    """
    return sorted(_THEMES)
