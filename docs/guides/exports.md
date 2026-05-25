# Exports

Every `SofraTable` consumes the same internal schema, so all renderers
produce consistent output.

| Method | Format | Notes |
|---|---|---|
| `.to_html(sticky_header=False, max_height=None)` | HTML fragment | Scoped CSS, inherit colours |
| `.to_markdown()` | GitHub-flavored Markdown | Alignment markers preserved |
| `.to_docx(path)` | Word `.docx` | Publication-quality, python-docx fully abstracted |
| `.to_latex(booktabs=True, float_position='ht', centering=True)` | LaTeX string | Booktabs by default; falls back to `\hline` |
| `.to_pptx(path, slide_title=None)` | PowerPoint `.pptx` | Optional extra `pysofra[pptx]` |
| `.to_xlsx(path, sheet_name='Table')` | Excel `.xlsx` | Numeric cells written as numbers; captions + footnotes + spanning headers preserved |

## Notebook reprs

`SofraTable` exposes:

- `_repr_html_` — used by Jupyter, Colab, VS Code, Quarto
- `_repr_markdown_` — used by Markdown-first viewers
- `_repr_latex_` — used by LaTeX-first frontends

Frontends try `_repr_html_` first by default, so the rich notebook
render is what you see.

## Round-tripping

The internal `SofraTable` is the source of truth. Exporting the same
object to HTML and DOCX produces consistent content (text, alignment,
captions, footnotes, spanning headers). Theme decisions that are
HTML-specific (CSS classes, `color-mix`) degrade naturally to DOCX
(`font_name`, `font_size`, border hints).
