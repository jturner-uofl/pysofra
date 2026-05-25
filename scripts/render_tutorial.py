#!/usr/bin/env python3
"""End-to-end tutorial-render pipeline.

Run this whenever the tutorial notebook source changes:

    .venv/bin/python scripts/render_tutorial.py

The script:

1. Re-executes ``examples/pysofra_tutorial.ipynb`` in place.
2. Renders it to ``examples/pysofra_tutorial.html``.
3. Injects a polished CSS overlay (modern typography, blockquote
   callouts, constrained body width, light + dark mode) so the
   standalone HTML looks publication-grade in any browser.

The CSS overlay lives in this file (see ``OVERLAY_CSS``) so any future
visual tweaks happen here and survive re-renders.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NB = REPO / "examples" / "pysofra_tutorial.ipynb"
HTML = REPO / "examples" / "pysofra_tutorial.html"
PYTHON = REPO / ".venv" / "bin" / "python"
JUPYTER = REPO / ".venv" / "bin" / "jupyter"


OVERLAY_CSS = """
<style>/* pysofra-tutorial-polish */
/* Pin the page to light colour scheme regardless of the OS / browser
 * preference. Documentation artifacts should look the same to every
 * reader; relying on prefers-color-scheme produced dark-on-dark text
 * in users with system-wide dark mode. */
:root {
  color-scheme: light;
  --jp-ui-font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue",
                       Arial, "Noto Sans", "Liberation Sans", sans-serif;
  --jp-content-font-family: var(--jp-ui-font-family);
  --jp-code-font-family: "JetBrains Mono", "SFMono-Regular", Consolas,
                         "Liberation Mono", Menlo, monospace;
  --jp-ui-font-size1: 14px;
  --jp-content-font-size1: 15px;
  --jp-ui-font-color0: #0f172a;
  --jp-ui-font-color1: #1a202c;
  --jp-ui-font-color2: #2d3748;
  --jp-ui-font-color3: #475569;
  --jp-content-font-color0: #0f172a;
  --jp-content-font-color1: #1a202c;
  --jp-cell-editor-background: #f7f8fa;
  --jp-cell-editor-border-color: #e2e8f0;
  --jp-layout-color0: #ffffff;
  --jp-layout-color1: #ffffff;
  --jp-border-color0: #e2e8f0;
  --jp-border-color1: #e2e8f0;
  --jp-border-color2: #cbd5e0;
  --jp-mirror-editor-keyword-color: #5a67d8;
  --jp-mirror-editor-string-color: #047857;
  --jp-mirror-editor-number-color: #b91c1c;
  --jp-mirror-editor-comment-color: #64748b;
  --jp-mirror-editor-variable-color: #1a202c;
  --jp-mirror-editor-operator-color: #334155;
  --jp-mirror-editor-punctuation-color: #334155;
}

/* Force-light body so a dark-mode browser doesn't auto-darken us. */
html, body {
  background: #fafbfc !important;
  color: #1a202c !important;
  color-scheme: light;
  font-family: var(--jp-ui-font-family);
  font-size: var(--jp-ui-font-size1);
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* Constrain content width for readability. */
#notebook-container,
.jp-Notebook,
.jp-notebook,
main {
  max-width: 920px !important;
  margin: 2rem auto !important;
  padding: 0 1.5rem !important;
  background: transparent;
  box-shadow: none;
}

/* Markdown headers — selectors include the bare tag form so we win
 * the cascade against nbconvert's default rules, which are class-less. */
h1, .jp-MarkdownOutput h1, .jp-RenderedMarkdown h1 {
  font-size: 2.2rem;
  font-weight: 800;
  color: #0b1a3d !important;
  margin: 2rem 0 1rem;
  padding-bottom: 0.4rem;
  border-bottom: 3px solid #0b3d91;
  letter-spacing: -0.02em;
}
h2, .jp-MarkdownOutput h2, .jp-RenderedMarkdown h2 {
  font-size: 1.5rem;
  font-weight: 700;
  color: #0f172a !important;
  margin: 2.5rem 0 1rem;
  padding-bottom: 0.3rem;
  border-bottom: 1px solid #cbd5e0;
  letter-spacing: -0.01em;
}
h3, .jp-MarkdownOutput h3, .jp-RenderedMarkdown h3 {
  font-size: 1.2rem;
  font-weight: 700;
  color: #1e293b !important;
  margin: 1.5rem 0 0.75rem;
}
h4, .jp-MarkdownOutput h4, .jp-RenderedMarkdown h4 {
  font-size: 1.05rem;
  font-weight: 700;
  color: #334155 !important;
  margin: 1rem 0 0.5rem;
}

/* Markdown body */
p, .jp-MarkdownOutput p, .jp-RenderedMarkdown p {
  color: #1a202c !important;
  margin: 0.65em 0;
}

/* Inline code */
p code, li code, td code, th code,
.jp-MarkdownOutput code,
.jp-RenderedMarkdown code {
  background: #eef1f5 !important;
  color: #0b3d91 !important;
  padding: 0.1em 0.4em;
  border-radius: 4px;
  font-family: var(--jp-code-font-family);
  font-size: 0.92em;
}

/* Blockquotes — used for tips & punch-lines */
blockquote,
.jp-MarkdownOutput blockquote,
.jp-RenderedMarkdown blockquote {
  border-left: 4px solid #0b3d91 !important;
  background: #eff4fb !important;
  padding: 0.7em 1em;
  margin: 1em 0;
  color: #0b2241 !important;
  font-style: italic;
  border-radius: 0 4px 4px 0;
}
blockquote p,
.jp-MarkdownOutput blockquote p,
.jp-RenderedMarkdown blockquote p {
  margin: 0.2em 0;
  color: #0b2241 !important;
}

/* Lists */
ul, ol,
.jp-MarkdownOutput ul, .jp-MarkdownOutput ol,
.jp-RenderedMarkdown ul, .jp-RenderedMarkdown ol {
  padding-left: 1.4em;
}
li,
.jp-MarkdownOutput li,
.jp-RenderedMarkdown li {
  margin: 0.3em 0;
  color: #1a202c !important;
}

/* Markdown tables (e.g. the TOC in the intro) */
.jp-MarkdownOutput table,
.jp-RenderedMarkdown table {
  border-collapse: collapse;
  margin: 1em 0;
  font-size: 0.95em;
  color: #1a202c;
}
.jp-MarkdownOutput th,
.jp-RenderedMarkdown th {
  background: #f1f5f9 !important;
  color: #0f172a !important;
  border-bottom: 2px solid #cbd5e0;
  padding: 0.5em 0.9em;
  text-align: left;
  font-weight: 600;
}
.jp-MarkdownOutput td,
.jp-RenderedMarkdown td {
  color: #1a202c !important;
  border-bottom: 1px solid #e2e8f0;
  padding: 0.45em 0.9em;
}

/* Horizontal rules — used as section breaks */
.jp-MarkdownOutput hr,
.jp-RenderedMarkdown hr {
  border: none;
  border-top: 2px dashed #cbd5e0;
  margin: 2.5rem 0;
}

/* Code cells */
.jp-CodeCell .jp-Cell-inputWrapper,
.jp-CodeCell .jp-InputArea-editor,
.jp-InputArea-editor,
.highlight {
  background: #f7f8fa !important;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 0.4em 0.8em;
  margin: 0.5em 0;
}

.highlight pre {
  margin: 0 !important;
  padding: 0 !important;
  background: transparent !important;
  font-family: var(--jp-code-font-family);
  font-size: 13.5px;
  line-height: 1.55;
}

/* Hide cell input/output prompts for a publication look. */
.jp-InputPrompt,
.jp-OutputPrompt {
  display: none;
}

/* Output area */
.jp-Cell-outputWrapper,
.jp-OutputArea-output {
  background: transparent !important;
  margin: 0.5em 0;
}

/* Plain-text output (print statements) */
.jp-OutputArea-output pre {
  background: #f7f8fa;
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  padding: 0.6em 0.9em;
  font-family: var(--jp-code-font-family);
  font-size: 13px;
  line-height: 1.45;
  color: #1a202c;
  white-space: pre-wrap;
}

/* PySofra tables already self-style; just give them a bit of margin. */
table.pysofra {
  margin: 1em 0 !important;
}

/* Image outputs (KM plots, forest plots, etc.) */
.jp-OutputArea img,
.jp-RenderedImage img {
  display: block;
  margin: 1em auto;
  max-width: 100%;
  border-radius: 4px;
}

/* Cell spacing */
.jp-Cell {
  margin: 0.75em 0;
}

/* Strong override: under no circumstances should any markdown text in
 * the rendered notebook be rendered in a low-contrast colour. This
 * defends against future nbconvert template changes that re-introduce
 * low-contrast --jp-* variable usage. */
.jp-Cell .jp-RenderedMarkdown,
.jp-Cell .jp-MarkdownOutput {
  color: #1a202c !important;
}
</style>
"""


def run(cmd: list[str]) -> None:
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    if not NB.exists():
        sys.stderr.write(f"notebook not found: {NB}\n")
        return 1

    # 1. Re-execute the notebook in place.
    run([
        str(JUPYTER), "nbconvert",
        "--to", "notebook",
        "--execute", str(NB),
        "--output", NB.name,
    ])

    # 2. Render to HTML with embedded images.
    run([
        str(JUPYTER), "nbconvert",
        "--to", "html",
        "--embed-images",
        str(NB),
    ])

    # 3. Inject the polish CSS overlay.
    html = HTML.read_text()
    # Strip any prior overlay before re-injecting so the script is idempotent.
    html = re.sub(
        r"<style>/\* pysofra-tutorial-polish \*/.*?</style>",
        "",
        html,
        flags=re.DOTALL,
    )
    html = html.replace("</head>", f"{OVERLAY_CSS}</head>", 1)
    HTML.write_text(html)
    print(f"Wrote polished {HTML} ({len(html):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
