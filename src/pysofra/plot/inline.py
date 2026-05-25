"""Cross-backend inline plot representation.

When a plot is attached to a :class:`SofraTable`, we keep multiple
serialised forms so each renderer can pick the one it needs:

* ``svg`` — for the HTML renderer.
* ``png_bytes`` — for DOCX and PPTX.
* ``pdf_bytes`` — for LaTeX (written as a sidecar file at export time).

All three are rendered once from the same matplotlib figure, ensuring
the visual representation is consistent across formats.

**Determinism.** matplotlib by default embeds the current wall-clock
timestamp and a process-random hash salt into every SVG/PNG/PDF it
writes. That makes binary renders unstable across processes and
breaks PySofra's "byte-identical reproducibility" guarantee for
plot-embedded tables. The helpers in this module strip those
non-deterministic fields so the same figure always serialises to the
same bytes.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from typing import Any

# A fixed (project-stable) hash salt overrides matplotlib's random
# default for inline SVG/PDF element IDs.
_HASH_SALT = "pysofra-inline-plot"

# Constant timestamp baked into PDF metadata so two renders of the
# same figure are byte-identical. The literal predates the project
# and has no operational meaning.
_PDF_FIXED_DATE = b"D:20260101000000Z"


def _configure_deterministic_matplotlib() -> None:
    """Pin matplotlib's hash salt and metadata so output is reproducible.

    Idempotent — safe to call repeatedly.
    """
    try:
        import matplotlib as mpl
    except ImportError:  # pragma: no cover — matplotlib is an optional extra
        return
    mpl.rcParams["svg.hashsalt"] = _HASH_SALT


# Patterns used to strip the timestamp from SVG / PNG / PDF outputs.
_SVG_DATE_RE = re.compile(
    rb"<dc:date>[^<]*</dc:date>",
    re.IGNORECASE,
)
_PDF_DATE_RE = re.compile(
    rb"/(?:CreationDate|ModDate)\s*\(D:\d+(?:Z|[+\-]\d{2}'\d{2}')?\)",
)
_PDF_ID_RE = re.compile(
    rb"/ID\s*\[\s*<[0-9A-Fa-f]+>\s*<[0-9A-Fa-f]+>\s*\]",
)


def _strip_svg_nondeterminism(svg_bytes: bytes) -> bytes:
    """Remove timestamp metadata from an SVG byte stream."""
    return _SVG_DATE_RE.sub(b"<dc:date>2026-01-01T00:00:00</dc:date>", svg_bytes)


def _strip_pdf_nondeterminism(pdf_bytes: bytes) -> bytes:
    """Pin /CreationDate, /ModDate, and /ID in a PDF byte stream."""
    out = _PDF_DATE_RE.sub(b"/CreationDate (" + _PDF_FIXED_DATE + b")", pdf_bytes)
    # We always emit a deterministic /ID derived from the file hash so the
    # two-id PDF reference is stable across runs.
    digest = hashlib.sha256(out).hexdigest()[:32].upper().encode()
    out = _PDF_ID_RE.sub(b"/ID [<" + digest + b"><" + digest + b">]", out)
    return out


def _strip_png_nondeterminism(png_bytes: bytes) -> bytes:
    """Strip the optional ``tIME`` and ``tEXt`` chunks from a PNG.

    PNG's ``tIME`` chunk stores the modification time; ``tEXt``/``iTXt``
    chunks added by matplotlib's `Software` key embed the matplotlib
    version + Python build banner. Removing them gives byte-stable PNG
    output across runs and across matplotlib patch releases.
    """
    # PNG layout: 8-byte signature, then a sequence of chunks
    #   [length (4) | type (4) | data (length) | crc (4)]
    if not png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return png_bytes  # pragma: no cover — not a PNG
    out = bytearray(png_bytes[:8])
    pos = 8
    drop_types = {b"tIME", b"tEXt", b"iTXt", b"zTXt"}
    while pos < len(png_bytes):
        if pos + 8 > len(png_bytes):  # pragma: no cover — malformed
            break
        length = int.from_bytes(png_bytes[pos : pos + 4], "big")
        ctype = png_bytes[pos + 4 : pos + 8]
        chunk_end = pos + 8 + length + 4  # data + crc
        if ctype not in drop_types:
            out.extend(png_bytes[pos:chunk_end])
        pos = chunk_end
    return bytes(out)


def fig_to_svg(fig: Any) -> str:
    """Serialise a matplotlib Figure to a deterministic inline SVG string.

    Trims the ``<?xml ...?>`` / ``<!DOCTYPE ...>`` headers so the result
    can be embedded directly into HTML, replaces the explicit
    width / height attributes with a responsive ``max-width:100%``
    style so the SVG scales inside its container, and strips the
    ``<dc:date>`` timestamp so two consecutive renders are byte-equal.
    """
    _configure_deterministic_matplotlib()
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    raw = _strip_svg_nondeterminism(buf.getvalue())
    svg = raw.decode("utf-8")
    idx = svg.find("<svg")
    if idx > 0:
        svg = svg[idx:]
    svg = svg.replace(
        "<svg ", '<svg style="max-width:100%;height:auto;" ', 1,
    )
    return svg


@dataclass(frozen=True)
class InlinePlot:
    """A plot serialised for every backend PySofra supports."""

    svg: str
    png_bytes: bytes
    pdf_bytes: bytes
    width_in: float
    height_in: float


def render_inline_plot(fig: Any, *, width_in: float, height_in: float,
                       dpi: int = 200) -> InlinePlot:
    """Serialise a matplotlib figure to SVG + PNG + PDF in one pass.

    All three byte streams are post-processed to remove the
    timestamps / process-randomised IDs matplotlib would otherwise
    embed, so the output is byte-identical across runs of the same
    PySofra version.
    """
    _configure_deterministic_matplotlib()
    svg = fig_to_svg(fig)

    # PNG bytes — for DOCX / PPTX.
    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png", bbox_inches="tight", dpi=dpi,
                metadata={"Software": None})
    png_bytes = _strip_png_nondeterminism(png_buf.getvalue())

    # PDF bytes — for LaTeX sidecar.
    pdf_buf = io.BytesIO()
    fig.savefig(pdf_buf, format="pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    pdf_bytes = _strip_pdf_nondeterminism(pdf_buf.getvalue())

    return InlinePlot(
        svg=svg,
        png_bytes=png_bytes,
        pdf_bytes=pdf_bytes,
        width_in=width_in,
        height_in=height_in,
    )
