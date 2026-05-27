"""Snapshot-lock for SofraTable: pin a published table to a content hash.

Once a Table 1 has been published in a paper, the authors typically
want CI to fail if anyone changes the upstream code/data in a way
that would alter the published numbers. Two friction points have
historically made this hard:

1. ``.to_html()`` carries a randomised CSS class on every render
   (``pysofra-<rand>``), so byte-comparing HTML produces false
   positives across runs of the same code on the same data.
2. ZIP-based binary backends (DOCX/PPTX/XLSX) had timestamp
   non-determinism until 0.1.0a2.

PySofra solves #2 (cross-process byte-determinism, see Step 11 of
the narrative-audit notebook). This module solves #1 by exposing
``SofraTable.snapshot_hash()``, ``.lock_snapshot(path)``, and
``.assert_snapshot(path)``, all of which hash the table's *logical
content* (rendered Markdown, which is content-only, plus the
table's footnotes and spanning headers) — not its presentational
randomness.

The intended workflow:

>>> # In the author's analysis script, run once:
>>> t = ps.tbl_one(df, by='arm').add_p()
>>> t.lock_snapshot("paper/table1.lock")
>>> # Commit paper/table1.lock to the repo.

>>> # In CI, on every PR:
>>> t = ps.tbl_one(df, by='arm').add_p()
>>> t.assert_snapshot("paper/table1.lock")
>>> # Any change to the table's logical content fails the build.

Hash policy
-----------
The hash includes:
  * Rendered Markdown (which captures all header text + cell text
    + spanning headers + alignment + missing-row formatting).
  * Footnotes (which encode test names, design info, level
    declarations — substantive content).

The hash excludes:
  * Theme (presentation only — the same numbers in jama vs minimal
    must produce the same hash so a style refresh isn't a content
    change).
  * Caption (often auto-generated, sometimes a build artefact).
  * Metadata dict (mostly internal state — builder name, attached
    fitted model — none of which is publishable content).
  * Inline plots (presentation — same numbers may yield a slightly
    different PNG due to matplotlib refresh).

If a future PySofra release changes its Markdown renderer in a
purely-cosmetic way (e.g. tightening whitespace), the snapshot
versioning escape hatch is the ``schema_version`` field embedded in
the lock file. A lock file from PySofra 0.1.x is compatible with
PySofra 0.1.x; bumping the schema_version makes older lock files
explicitly recognisable as incompatible.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .table import SofraTable


# Bumping this invalidates existing .lock files; do so only on
# intentional, breaking Markdown-renderer changes.
_SCHEMA_VERSION = 1


def snapshot_content(table: SofraTable) -> str:
    """Return the canonical-content string the hash is computed over.

    Exposed for debugging — call this to see exactly what changed
    when an ``assert_snapshot`` mismatch fires.
    """
    parts: list[str] = []
    parts.append("##MARKDOWN##")
    parts.append(table.to_markdown())
    parts.append("##FOOTNOTES##")
    for fn in table.footnotes or ():
        parts.append(fn)
    parts.append("##SPANNING##")
    for sh in table.spanning_headers or ():
        # SpanningHeader carries label + cspan; both substantive
        parts.append(f"{sh.label}|{getattr(sh, 'cspan', '?')}")
    return "\n".join(parts)


def snapshot_hash(table: SofraTable) -> str:
    """Return the SHA-256 hex digest of the table's logical content."""
    content = snapshot_content(table)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def lock_snapshot(table: SofraTable, path: str | Path) -> dict[str, Any]:
    """Write the table's snapshot to ``path``.

    The lock file is a small JSON blob carrying the schema version,
    the SHA-256 hash, and the raw content (for diff-friendly review).
    Returns the dict that was written, for the caller's convenience.
    """
    content = snapshot_content(table)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "sha256":         digest,
        "content":        content,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def assert_snapshot(table: SofraTable, path: str | Path) -> None:
    """Compare the table to the snapshot at ``path``; raise on mismatch.

    Raises
    ------
    FileNotFoundError
        if the lock file does not exist (call ``lock_snapshot`` first).
    ValueError
        if the lock file schema is from a newer/incompatible PySofra.
    AssertionError
        if the table's current snapshot hash does not equal the
        pinned hash. The error message includes a unified diff
        between the pinned content and the current content so the
        author can see exactly what changed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"snapshot lock not found at {p}; call "
            f"`table.lock_snapshot({p!r})` first."
        )
    payload = json.loads(p.read_text())
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(
            f"snapshot at {p} uses schema_version="
            f"{payload.get('schema_version')!r}, but this PySofra "
            f"speaks v{_SCHEMA_VERSION}. Regenerate the lock with "
            f"`table.lock_snapshot({p!r})` after verifying the "
            f"numbers haven't drifted unintentionally."
        )
    expected = payload["sha256"]
    actual = snapshot_hash(table)
    if expected == actual:
        return
    import difflib
    diff = "\n".join(difflib.unified_diff(
        payload["content"].splitlines(),
        snapshot_content(table).splitlines(),
        fromfile="pinned",
        tofile="current",
        lineterm="",
    ))
    raise AssertionError(
        f"Snapshot mismatch for table at {p}.\n"
        f"  pinned sha256:  {expected}\n"
        f"  current sha256: {actual}\n"
        f"\nDiff (pinned → current):\n{diff[:4000]}"
    )
