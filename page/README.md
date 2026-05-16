# Project Landing Page

Static single-page presentation of the paper, matching the SİU 2026 / TEKNOFEST
poster's color palette (cream background, navy headers, pale-blue section cards,
gold accents).

## Files

- `index.html` — content (abstract, method, results, discussion, references, BibTeX).
- `styles.css` — palette, typography, layout, table + figure styling.
- `script.js` — BibTeX copy-to-clipboard + on-scroll reveal of section cards.
- `assets/` — copied figures from `visuals/` (architecture, qualitative grid,
  chroma recovery, MDN diversity, training curves, caption examples).

## Run

It's pure static HTML/CSS/JS — open the file directly:

```bash
open page/index.html
```

Or serve it locally if you prefer a real `http://` origin (recommended for the
`navigator.clipboard` API to work on Chrome):

```bash
python3 -m http.server -d page 8000
# → http://localhost:8000/
```
