"""Make ZIP-based document outputs (.docx, .pptx) byte-deterministic
across processes.

Both ``python-docx`` and ``python-pptx`` save their documents as OOXML
ZIP archives. The XML *contents* are deterministic given a deterministic
input, but the underlying ``zipfile.ZipFile.writestr`` stamps each entry
with ``time.localtime()`` by default, so two saves at different wall
times produce different bytes even though every file inside is
identical.

PySofra's published claim is that every renderer produces
byte-deterministic output. To honour that for the OOXML formats, after
``python-docx`` / ``python-pptx`` finishes writing the archive we
rewrite it in-place with every entry's ``date_time`` pinned to a fixed
epoch and the compression level fixed.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

# Fixed wall-clock used for every ZIP entry's ``date_time`` field. The
# ZIP format only stores DOS time (2-second granularity from 1980); we
# use a deterministic constant well clear of that lower bound.
_FIXED_DATE_TIME: tuple[int, int, int, int, int, int] = (2000, 1, 1, 0, 0, 0)


def make_zip_deterministic(path: Path) -> None:
    """Rewrite ``path`` (an OOXML zip) so its bytes are reproducible.

    Reads every entry, pins ``date_time`` to ``_FIXED_DATE_TIME``, and
    writes the archive back with the same compression mode and preserved
    entry order. Idempotent — applying twice produces the same bytes.

    Parameters
    ----------
    path
        Path to a ZIP-format file (``.docx``, ``.pptx``, ``.xlsx``).
    """
    p = Path(path)
    raw = p.read_bytes()
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw), mode="r") as src, \
            zipfile.ZipFile(buf, mode="w") as dst:
        for info in src.infolist():
            new_info = zipfile.ZipInfo(
                filename=info.filename,
                date_time=_FIXED_DATE_TIME,
            )
            new_info.compress_type = info.compress_type
            new_info.external_attr = info.external_attr
            new_info.create_system = info.create_system
            new_info.internal_attr = info.internal_attr
            dst.writestr(new_info, src.read(info.filename))
    p.write_bytes(buf.getvalue())
