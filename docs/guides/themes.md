# Themes & styling

PySofra ships six themes. Apply with `.theme(name)`:

| Theme | Look |
|---|---|
| `default` | Generic sans-serif, modern, neutral |
| `clinical` | Slightly larger, soft header band, comfortable spacing |
| `compact` | Tight padding, smaller font — for data-heavy tables |
| `jama` | Times Roman, JAMA-style rules |
| `nejm` | Georgia serif, NEJM-style rules |
| `minimal` | Single under-header rule, no internal separators |

```python
ps.available_themes()
# ['clinical', 'compact', 'default', 'jama', 'minimal', 'nejm']
```

## Custom themes

```python
from pysofra.themes.registry import Theme, register_theme

house_style = Theme(
    name='house',
    css={
        'table': {'font-family': '"Inter", sans-serif', 'font-size': '13px'},
        'th': {'background': '#0b3d91', 'color': 'white'},
        'caption': {'color': '#0b3d91', 'font-weight': '700'},
    },
    docx={'font_name': 'Inter', 'font_size': 10},
)
register_theme(house_style)

ps.tbl_one(df, by='arm').theme('house')
```

## Conditional formatting

| Method | Effect |
|---|---|
| `.bold_p(threshold=0.05)` | bold rows with significant p-value |
| `.bold_if(predicate)` | bold rows where `predicate(row)` is true |
| `.highlight_if(predicate, color='#fff3cd')` | row background highlight |
| `.style_if(predicate, bold=, italic=, color=)` | combined |

```python
table.highlight_if(
    lambda r: r.cells[0].text.startswith('age'),
    color='#fff3cd',
)
```

## Sticky headers in notebooks

```python
table.to_html(sticky_header=True, max_height='60vh')
```

Used inside a scroll container, the header stays in view as the body
scrolls.
