#!/usr/bin/env python3
"""Render the showcase notebook to a polished, self-contained HTML page.

Follows the post-process recipe:

1. Run nbconvert with the classic template to get plain-but-correct HTML.
2. Surgical regex pass to replace ``<title>``, strip dead CDN scripts,
   inject a CSS bundle into ``<head>``, drop a hero ``<div>`` after
   ``<body>``, wrap the notebook container in a TOC + main grid, and
   inject a scrollspy script before ``</body>``.

No custom Jinja templates, no Sphinx, no Quarto — just regex + a CSS
bundle + a small vanilla-JS scrollspy.

Run:

    .venv/bin/python scripts/render_showcase.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NB = REPO / "examples" / "pysofra_showcase.ipynb"
HTML = REPO / "examples" / "pysofra_showcase.html"


# ---------------------------------------------------------------------------
# DESIGN TOKENS
# ---------------------------------------------------------------------------
# One accent color used everywhere; warm off-white background; dark code
# theme even on the light page. CSS variables so any chromatic tweak
# propagates without touching individual rules.
POLISH_CSS = """
:root {
  --pcd-bg: #fafaf9;
  --pcd-fg: #1a1a1a;
  --pcd-muted: #555;
  --pcd-accent: #0b6e7c;
  --pcd-accent-dark: #074d57;
  --pcd-accent-soft: #e6f1f3;
  --pcd-card: #ffffff;
  --pcd-border: #e8e8e6;
  --pcd-code-bg: #1e1e2e;
  --pcd-code-fg: #cdd6f4;
  --pcd-table-band: #f5f5f4;
  --pcd-shadow: 0 1px 2px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.06);
  --pcd-radius: 10px;
  --pcd-font-body: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI',
                   system-ui, 'Helvetica Neue', Arial, sans-serif;
  --pcd-font-display: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI',
                      system-ui, sans-serif;
  --pcd-font-mono: 'JetBrains Mono', 'SF Mono', SFMono-Regular, Menlo,
                   Consolas, monospace;
}

* { box-sizing: border-box; }
html, body { background: var(--pcd-bg); color: var(--pcd-fg); }
body {
  font-family: var(--pcd-font-body);
  font-size: 16px;
  line-height: 1.65;
  -webkit-font-smoothing: antialiased;
  margin: 0;
  padding: 0;
}

/* Hide nbconvert prompts and the anchor-link pilcrows */
.prompt, .input_prompt, .output_prompt { display: none !important; }
.anchor-link { display: none !important; }

/* Hero banner */
.pcd-hero {
  background: linear-gradient(135deg, #062a32 0%, #0b6e7c 55%, #129d96 100%);
  color: white;
  padding: 88px 48px 96px;
  margin-bottom: 0;
  position: relative;
  overflow: hidden;
}
.pcd-hero::after {
  content: '';
  position: absolute;
  inset: 0;
  background:
    radial-gradient(ellipse at 80% 20%, rgba(255,255,255,0.10), transparent 60%),
    radial-gradient(ellipse at 10% 90%, rgba(255,255,255,0.06), transparent 50%);
  pointer-events: none;
}
.pcd-hero-inner {
  max-width: 1180px;
  margin: 0 auto;
  position: relative;
  z-index: 1;
}
.pcd-hero h1 {
  font-family: var(--pcd-font-display);
  font-size: 64px;
  font-weight: 800;
  letter-spacing: -0.035em;
  margin: 0 0 16px;
  line-height: 1.0;
  color: white;
  background: none;
}
.pcd-hero .tagline {
  font-size: 22px;
  font-weight: 400;
  opacity: 0.92;
  margin: 0 0 32px;
  max-width: 760px;
  line-height: 1.5;
}
.pcd-hero .chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}
.pcd-hero .chip {
  display: inline-block;
  padding: 6px 14px;
  background: rgba(255,255,255,0.15);
  border: 1px solid rgba(255,255,255,0.25);
  border-radius: 999px;
  font-size: 13px;
  font-weight: 500;
  letter-spacing: 0.02em;
  color: white;
  text-decoration: none;
  transition: background 0.15s;
}
.pcd-hero .chip:hover { background: rgba(255,255,255,0.25); }
.pcd-hero .meta {
  margin-top: 32px;
  font-size: 14px;
  opacity: 0.78;
  font-family: var(--pcd-font-mono);
}

/* Layout: sticky TOC + main */
.pcd-page {
  display: grid;
  grid-template-columns: 240px minmax(0, 1fr);
  gap: 48px;
  max-width: 1280px;
  margin: 0 auto;
  padding: 48px 32px 96px;
}
@media (max-width: 900px) {
  .pcd-page { grid-template-columns: 1fr; padding: 32px 20px; }
  .pcd-toc { display: none; }
}
.pcd-toc {
  position: sticky;
  top: 24px;
  align-self: start;
  font-size: 13px;
  max-height: calc(100vh - 48px);
  overflow-y: auto;
}
.pcd-toc-title {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--pcd-muted);
  margin: 0 0 14px;
  padding-left: 12px;
}
.pcd-toc ul { list-style: none; padding: 0; margin: 0; }
.pcd-toc li.h2 { margin-bottom: 4px; }
.pcd-toc li.h3 { margin-left: 16px; }
.pcd-toc a {
  display: block;
  padding: 6px 12px;
  color: var(--pcd-muted);
  text-decoration: none;
  border-left: 2px solid transparent;
  transition: color 0.15s, border-color 0.15s, background 0.15s;
  border-radius: 0 6px 6px 0;
  line-height: 1.4;
}
.pcd-toc a:hover { color: var(--pcd-accent); background: var(--pcd-accent-soft); }
.pcd-toc a.active {
  color: var(--pcd-accent-dark);
  border-left-color: var(--pcd-accent);
  font-weight: 600;
  background: var(--pcd-accent-soft);
}
.pcd-toc li.h3 a { font-size: 12px; padding: 4px 12px; }

/* Main content column — strip nbconvert's container chrome so our layout wins. */
#notebook, #notebook-container {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin: 0 !important;
  width: 100% !important;
  max-width: 100% !important;
}
.container { width: 100% !important; max-width: 100% !important; padding: 0 !important; }
.cell { padding: 0 !important; margin: 0 0 28px !important; background: transparent !important; }
.text_cell { background: transparent !important; }
.input_area, .output_area, .inner_cell { background: transparent !important; }

/* Headings */
h1, .text_cell h1 {
  font-family: var(--pcd-font-display);
  font-weight: 800;
  letter-spacing: -0.02em;
  font-size: 36px;
  line-height: 1.15;
  margin: 56px 0 12px;
  border: 0;
  padding: 0;
  color: var(--pcd-fg);
}
h2, .text_cell h2 {
  font-family: var(--pcd-font-display);
  font-weight: 700;
  letter-spacing: -0.02em;
  font-size: 30px;
  margin: 72px 0 18px;
  padding-bottom: 12px;
  border-bottom: 2px solid var(--pcd-accent);
  color: var(--pcd-fg);
  position: relative;
}
h2::before {
  content: '';
  display: block;
  width: 48px;
  height: 4px;
  background: var(--pcd-accent);
  border-radius: 2px;
  margin-bottom: 16px;
}
h3, .text_cell h3 {
  font-family: var(--pcd-font-display);
  font-weight: 700;
  letter-spacing: -0.01em;
  font-size: 22px;
  margin: 40px 0 14px;
  color: var(--pcd-accent-dark);
}

/* Body text */
.text_cell p, .text_cell li {
  font-size: 16.5px;
  line-height: 1.7;
  color: #2a2a2a;
}
.text_cell strong { color: var(--pcd-fg); font-weight: 700; }
.text_cell em { font-style: italic; }
.text_cell hr { border: 0; border-top: 1px dashed var(--pcd-border); margin: 56px 0; }

/* Markdown tables (the contents table in the cold open) */
.text_cell table, .rendered_html table {
  border-collapse: separate !important;
  border-spacing: 0;
  margin: 16px 0 !important;
  background: var(--pcd-card);
  border-radius: var(--pcd-radius);
  overflow: hidden;
  box-shadow: var(--pcd-shadow);
  font-size: 14.5px;
}
.text_cell thead th, .rendered_html thead th {
  background: var(--pcd-accent-soft) !important;
  color: var(--pcd-accent-dark) !important;
  font-weight: 600;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  padding: 11px 16px !important;
  text-align: left !important;
  border: 0 !important;
  border-bottom: 2px solid var(--pcd-accent) !important;
}
.text_cell tbody td, .rendered_html tbody td {
  padding: 9px 16px !important;
  border: 0 !important;
  border-bottom: 1px solid var(--pcd-border) !important;
  color: #2a2a2a;
}
.text_cell tbody tr:last-child td, .rendered_html tbody tr:last-child td {
  border-bottom: 0 !important;
}

/* Inline code */
.text_cell code, p code, li code, td code {
  font-family: var(--pcd-font-mono);
  font-size: 13.5px;
  background: var(--pcd-accent-soft);
  color: var(--pcd-accent-dark);
  padding: 2px 6px;
  border-radius: 4px;
  font-weight: 500;
}
.text_cell code { white-space: nowrap; }

/* Code blocks — dark themed (catches BOTH Jupyter code cells and
   markdown fenced code, which render through the same .highlight class). */
div.input { margin-bottom: 0 !important; }
.input_area {
  background: transparent !important;
  border: 0 !important;
  padding: 0 !important;
  box-shadow: none !important;
}
.highlight {
  background: var(--pcd-code-bg) !important;
  border-radius: var(--pcd-radius) !important;
  border: 0 !important;
  padding: 16px 18px !important;
  margin: 8px 0 !important;
  box-shadow: var(--pcd-shadow);
  overflow-x: auto;
}
.highlight pre, .highlight code {
  background: transparent !important;
  color: var(--pcd-code-fg) !important;
  font-family: var(--pcd-font-mono) !important;
  font-size: 13.5px !important;
  line-height: 1.6 !important;
  margin: 0 !important;
  padding: 0 !important;
  white-space: pre !important;
  border: 0 !important;
}
.highlight code { display: inline; white-space: pre; }

/* Catppuccin Mocha syntax-token palette */
.highlight .k, .highlight .kn, .highlight .kc, .highlight .kd,
.highlight .kr, .highlight .kp, .highlight .ow {
  color: #cba6f7 !important; font-weight: 500;
}
.highlight .s, .highlight .s1, .highlight .s2, .highlight .sb,
.highlight .sx, .highlight .sd { color: #a6e3a1 !important; }
.highlight .c, .highlight .c1, .highlight .cm, .highlight .ch,
.highlight .cp { color: #9399b2 !important; font-style: italic; }
.highlight .nf, .highlight .nc, .highlight .nn { color: #89b4fa !important; }
.highlight .nb { color: #f9e2af !important; }
.highlight .mi, .highlight .mf, .highlight .m, .highlight .il {
  color: #fab387 !important;
}
.highlight .o, .highlight .p { color: #bac2de !important; }
.highlight .n, .highlight .na, .highlight .nx { color: #cdd6f4 !important; }
.highlight .kc { color: #fab387 !important; }   /* True/False/None */
.highlight .se { color: #f5c2e7 !important; }   /* String escapes */

/* Output area */
.output_area { padding: 0 !important; margin-top: 10px !important; }
.output_text { font-family: var(--pcd-font-mono); font-size: 13px; }
.output_subarea { max-width: 100% !important; padding: 0 !important; }
.output_stream, .output_text pre {
  background: #f6f5f3 !important;
  border: 1px solid var(--pcd-border) !important;
  border-radius: var(--pcd-radius) !important;
  padding: 14px 18px !important;
  font-family: var(--pcd-font-mono);
  font-size: 13px;
  color: #3a3a3a;
  margin: 6px 0 !important;
  overflow-x: auto;
}

/* DataFrames + PySofra rendered HTML tables — banded, modern, breathing.
   PySofra emits its tables inside `.output_html .rendered_html` so we
   style both that and the legacy `.dataframe` shape. */
.dataframe, table.dataframe,
.rendered_html table.pysofra,
.rendered_html .pysofra {
  border-collapse: separate !important;
  border-spacing: 0;
  width: auto !important;
  max-width: 100%;
  margin: 12px 0 !important;
  background: var(--pcd-card);
  border-radius: var(--pcd-radius);
  overflow: hidden;
  box-shadow: var(--pcd-shadow);
  font-family: var(--pcd-font-body);
  font-size: 13.5px;
}
.dataframe thead th, .rendered_html table.pysofra thead th {
  background: var(--pcd-accent-soft) !important;
  color: var(--pcd-accent-dark) !important;
  font-weight: 600 !important;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  padding: 12px 16px !important;
  border: 0 !important;
  text-align: left !important;
  border-bottom: 2px solid var(--pcd-accent) !important;
}
.dataframe tbody td, .dataframe tbody th,
.rendered_html table.pysofra tbody td,
.rendered_html table.pysofra tbody th {
  padding: 10px 16px !important;
  border: 0 !important;
  border-bottom: 1px solid var(--pcd-border) !important;
  font-family: var(--pcd-font-mono);
  font-size: 12.5px;
  color: #2a2a2a;
}
.dataframe tbody tr:nth-child(even),
.rendered_html table.pysofra tbody tr:nth-child(even) {
  background: var(--pcd-table-band);
}
.dataframe tbody tr:hover,
.rendered_html table.pysofra tbody tr:hover {
  background: var(--pcd-accent-soft);
}
.dataframe tbody tr:last-child td, .dataframe tbody tr:last-child th,
.rendered_html table.pysofra tbody tr:last-child td {
  border-bottom: 0 !important;
}

/* Charts — framed, breathing. PySofra forest plots / KM curves embed
   as PNG; the gtsummary comparison figure also lands in .output_png. */
.output_svg, .output_png, .output_jpeg {
  display: block;
  margin: 12px 0 !important;
  padding: 18px !important;
  background: var(--pcd-card);
  border-radius: var(--pcd-radius);
  border: 1px solid var(--pcd-border);
  box-shadow: var(--pcd-shadow);
  overflow-x: auto;
  text-align: center;
}
.output_svg img, .output_svg svg, .output_png img {
  max-width: 100%;
  height: auto;
  display: inline-block;
  margin: 0 auto;
}

/* Anchor scroll offset */
h2, h3 { scroll-margin-top: 24px; }

/* Footer */
.pcd-footer {
  border-top: 1px solid var(--pcd-border);
  padding: 32px 0 12px;
  margin-top: 80px;
  color: var(--pcd-muted);
  font-size: 13px;
  text-align: center;
}
.pcd-footer a { color: var(--pcd-accent); text-decoration: none; }
.pcd-footer a:hover { text-decoration: underline; }
"""


POLISH_JS = """
(function () {
  // Build a TOC from h2/h3 in the rendered content.
  var main = document.querySelector('#pcd-main');
  if (!main) return;
  var headings = main.querySelectorAll('h2, h3');
  var tocList = document.querySelector('#pcd-toc-list');
  if (!tocList || !headings.length) return;
  var items = [];
  headings.forEach(function (h) {
    if (!h.id) return;
    var li = document.createElement('li');
    li.className = h.tagName.toLowerCase();
    var a = document.createElement('a');
    a.href = '#' + h.id;
    var label = h.textContent.replace(/\xb6.*$/, '').trim();
    a.textContent = label;
    a.dataset.target = h.id;
    li.appendChild(a);
    tocList.appendChild(li);
    items.push({ id: h.id, link: a, el: h });
  });

  // Scrollspy: highlight whichever heading the reader is currently on.
  var current = null;
  function onScroll() {
    var y = window.scrollY + 80;
    var active = items[0];
    for (var i = 0; i < items.length; i++) {
      if (items[i].el.offsetTop <= y) active = items[i];
    }
    if (active && active !== current) {
      if (current) current.link.classList.remove('active');
      active.link.classList.add('active');
      current = active;
    }
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
})();
"""


# nbconvert classic template URL-encodes the em-dash in heading IDs as
# %E2%80%94 — we mirror that for the chip targets.
DASH = "%E2%80%94"

HERO_HTML = f"""\
<div class="pcd-hero">
  <div class="pcd-hero-inner">
    <h1>pysofra</h1>
    <p class="tagline">
      The missing statistical reporting layer for Python — Table 1,
      regression with forest plots, survival summaries, multiple-imputation
      pooling, survey-weighted dispatch — from one immutable object to
      seven byte-deterministic output formats. Demonstrated below on a
      fully synthetic teaching dataset.
    </p>
    <div class="chips">
      <a class="chip" href="#Part-I-{DASH}-The-trial">I · the trial</a>
      <a class="chip" href="#Part-II-{DASH}-Did-randomisation-succeed?">II · Table 1</a>
      <a class="chip" href="#Part-III-{DASH}-Receipts:-every-test-against-the-reference-library">III · receipts</a>
      <a class="chip" href="#Part-V-{DASH}-Did-Drug-X-reduce-major-adverse-cardiac-events?">V · primary endpoint</a>
      <a class="chip" href="#Part-VII-{DASH}-Survival:-when-did-the-curves-diverge?">VII · survival</a>
      <a class="chip" href="#Part-IX-{DASH}-The-fanout">IX · the fanout</a>
      <a class="chip" href="#Part-XIV-{DASH}-Cross-validation:-PySofra-vs-gtsummary">XIV · vs gtsummary</a>
      <a class="chip" href="#Part-XV-{DASH}-Where-next">XV · cheat sheet</a>
    </div>
    <div class="meta">
      teaching demo · fully synthetic 500-row two-arm dataset · engineered treatment effect · not a real study
    </div>
  </div>
</div>
"""


FOOTER_HTML = """\
<footer class="pcd-footer">
  <p>
    Generated from
    <a href="pysofra_showcase.ipynb"><code>examples/pysofra_showcase.ipynb</code></a>
    via <code>scripts/render_showcase.py</code>.
    All charts and tables embedded inline — no CDN, no JavaScript runtime
    beyond the local scrollspy.
  </p>
</footer>
"""


def polish_html(html_path: Path) -> None:
    """Apply the recipe: title, drop CDN scripts, inject CSS, hero, TOC, JS."""
    html = html_path.read_text()

    # 1. Title
    html = re.sub(
        r"<title>[^<]*</title>",
        "<title>pysofra — the showcase</title>",
        html,
        count=1,
    )

    # 2. Drop the CDN script tags. nbconvert classic links jquery /
    # require.js from cdnjs; neither is needed once we've embedded all
    # the outputs and our scrollspy is vanilla JS.
    html = re.sub(
        r'<script[^>]*src="https://cdnjs\.cloudflare\.com[^"]+"[^>]*>\s*</script>',
        "",
        html,
    )
    # The mermaid esm-module import is loaded from cdnjs via a module
    # script — strip it too (we don't use mermaid diagrams).
    html = re.sub(
        r'<script type="module">\s*import mermaid.*?</script>',
        "",
        html,
        flags=re.DOTALL,
    )

    # 3. Inject CSS just before </head>.
    polish_block = f"<style>{POLISH_CSS}</style>"
    html = html.replace("</head>", polish_block + "</head>", 1)

    # 4+5. Wrap the notebook container with the hero + TOC + main grid.
    toc_block = (
        '<aside class="pcd-toc">'
        '<div class="pcd-toc-title">Contents</div>'
        '<ul id="pcd-toc-list"></ul>'
        "</aside>"
    )
    new_open = (
        HERO_HTML
        + '<div class="pcd-page">'
        + toc_block
        + '<main id="pcd-main">'
    )
    new_close = "</main></div>" + FOOTER_HTML

    body_open_match = re.search(r"<body[^>]*>", html)
    if body_open_match is None:
        raise RuntimeError("no <body> tag found in HTML")
    insert_after_idx = body_open_match.end()
    html = html[:insert_after_idx] + new_open + html[insert_after_idx:]
    html = html.replace(
        "</body>",
        new_close + f"<script>{POLISH_JS}</script></body>",
        1,
    )

    html_path.write_text(html)


def run(cmd: list[str]) -> None:
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main(argv: list[str]) -> int:
    if not NB.exists():
        sys.stderr.write(f"notebook not found: {NB}\n")
        return 1

    execute = "--execute" in argv

    # 1. Optionally re-execute the notebook (skip by default; use the
    # standalone nbconvert call when iterating on the post-process pass).
    if execute:
        run([
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute", str(NB),
            "--output", NB.name,
            "--ExecutePreprocessor.timeout=240",
        ])

    # 2. Render to HTML via the CLASSIC template — the one whose CSS
    # selectors our polish bundle targets (`#notebook`, `.cell`,
    # `.text_cell`, `.input_area`, `.output_area`, `.highlight`).
    run([
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "html",
        "--template", "classic",
        "--embed-images",
        "--output-dir", str(HTML.parent),
        "--output", HTML.name,
        str(NB),
    ])

    # 3. Surgical post-process.
    polish_html(HTML)
    size_kb = HTML.stat().st_size // 1024
    print(f"Wrote polished {HTML} ({size_kb} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
