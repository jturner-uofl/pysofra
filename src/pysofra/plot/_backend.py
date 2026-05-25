"""Headless-safe matplotlib backend setup.

PySofra never opens a window — every figure is serialised to bytes
(PNG, SVG, PDF). We therefore force matplotlib's ``Agg`` backend
before pyplot first creates a figure. Without this, the default
backend on macOS is ``MacOSX`` which calls into the GUI subsystem;
in sandboxed environments (HOME=/nonexistent, no display, container
without X) this aborts with a Cocoa error during figure creation.

This helper is idempotent: ``matplotlib.use("Agg", force=True)`` is
a cheap dictionary update once the backend is loaded. Calling it
from every plot entry point is the simplest way to guarantee Agg
is in effect regardless of import order.
"""

from __future__ import annotations


def use_headless_backend() -> None:
    """Force matplotlib's Agg backend in this process.

    Called at the top of every render/plot entry point in PySofra,
    immediately before ``import matplotlib.pyplot as plt``. Safe to
    call repeatedly. No-op if matplotlib is not installed (the caller
    handles the optional-dependency import error in its own
    try/except).
    """
    try:
        import matplotlib
    except ImportError:  # pragma: no cover — caller raises a friendlier error
        return
    matplotlib.use("Agg", force=True)
