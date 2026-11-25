"""Append the NX-1 nomenclature complement to the Agent Mechanics treatise PDF.

Usage:
    python scripts/append_nx1_to_treatise.py <treatise.pdf> [--out docs/Agent_Mechanics_Treatise_EN.pdf]
                                             [--md docs/NOMENCLATURE_COMPLEMENT.md] [--fonts <dir with DejaVu*.ttf>]

What it does:
* renders ``docs/NOMENCLATURE_COMPLEMENT.md`` as an appendix in a serif face, on the
  book's page size, with the book's running header and page numbers continuing
  after its last printed page;
* adds the appendix to the book's Contents page;
* writes PDF bookmarks for every part of the book and for the appendix.

The original PDF is never modified. Build-time requirements (not package
dependencies): ``pymupdf`` and ``markdown``, plus the DejaVu fonts (Serif, Sans
Mono, Math TeX Gyre), which ship with most Linux distributions and many Windows
installs.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import markdown
import pymupdf as fitz

# Printed page -> part, from the book's Contents. PDF page = printed + OFFSET.
PARTS = [
    ("Canonical glossary (agent, mass, energy)", None, 3),
    ("Normalised formal apparatus", 2, None),
    ("Corrigenda and revised criteria", 12, None),
    ("Part I · Agent Mechanics", 13, None),
    ("Part IV · Variety ranks and nomenclature of evolutions", 20, None),
    ("Part IV·bis · Compositional language of ensembles", 24, None),
    ("Taxonomy of flux chains and circuits (extension)", 35, None),
    ("Part IV·ter · Taxonomy of agents", 38, None),
    ("Part V · The world as a rate matrix", 48, None),
    ("Part VI · Electrodynamics of species", 52, None),
    ("Part VII · Topology of agency", 59, None),
    ("Part VIII · Proof obligations and open problems", 63, None),
    ("Volume II · Shells, rings and circuits", 66, None),
    ("Synthesis note · Inter-agent flow exchange", 74, None),
]
OFFSET = 3
APPENDIX_TITLE = "Nomenclature complement NX-1"

CSS = """
@font-face {{ font-family: serif1; src: url({serif}); }}
@font-face {{ font-family: serif1; src: url({serif_b}); font-weight: bold; }}
@font-face {{ font-family: serif1; src: url({serif_i}); font-style: italic; }}
@font-face {{ font-family: serif1; src: url({serif_bi}); font-weight: bold; font-style: italic; }}
@font-face {{ font-family: mono1; src: url({mono}); }}
@font-face {{ font-family: mono1; src: url({mono_b}); font-weight: bold; }}
@font-face {{ font-family: math1; src: url({math}); }}
* {{ font-family: serif1, math1; }}
body {{ font-size: 9.4pt; line-height: 1.38; color: #111; }}
p, li {{ text-align: justify; margin: 0 0 5pt 0; }}
.eyebrow {{ font-size: 9pt; letter-spacing: 2pt; text-transform: uppercase; color: #555; margin: 0 0 4pt 0; }}
h1 {{ font-size: 18pt; margin: 0 0 8pt 0; }}
h2 {{ font-size: 12.5pt; margin: 14pt 0 5pt 0; }}
h3 {{ font-size: 10.5pt; margin: 10pt 0 4pt 0; }}
code {{ font-family: mono1, math1; font-size: 8pt; }}
pre {{ font-family: mono1, math1; font-size: 7.2pt; line-height: 1.3; margin: 4pt 0 8pt 12pt; white-space: pre-wrap; color: #222; }}
pre code {{ font-size: 7.2pt; }}
table {{ border-collapse: collapse; margin: 4pt 0 9pt 0; width: 100%; }}
th, td {{ border: 0.4pt solid #999; padding: 2.5pt 4pt; font-size: 7.8pt; line-height: 1.25; text-align: left; vertical-align: top; }}
th {{ font-weight: bold; border-bottom: 0.8pt solid #555; }}
hr {{ border: none; border-top: 0.5pt solid #bbb; margin: 10pt 0; }}
blockquote {{ margin: 6pt 0 6pt 10pt; color: #333; }}
"""


def find_fonts(font_dir: str | None) -> dict[str, str]:
    names = {
        "serif": "DejaVuSerif.ttf",
        "serif_b": "DejaVuSerif-Bold.ttf",
        "serif_i": "DejaVuSerif-Italic.ttf",
        "serif_bi": "DejaVuSerif-BoldItalic.ttf",
        "mono": "DejaVuSansMono.ttf",
        "mono_b": "DejaVuSansMono-Bold.ttf",
        "math": "DejaVuMathTeXGyre.ttf",
    }
    dirs = (
        [font_dir]
        if font_dir
        else [
            r"C:\Windows\Fonts",
            "/usr/share/fonts/truetype/dejavu",
            "/usr/share/fonts/TTF",
            "/Library/Fonts",
        ]
    )
    for d in dirs:
        if d and all(Path(d, f).is_file() for f in names.values()):
            return {"dir": d, **names}
    raise SystemExit(f"DejaVu fonts not found in {dirs}; pass --fonts")


def render_pages(md_text: str, page_rect: fitz.Rect, fonts: dict[str, str]) -> fitz.Document:
    """Story layout via a DocumentWriter (the supported multi-page path)."""
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    html = f'<p class="eyebrow">Appendix</p>{body}'
    css = CSS.format(**{k: v for k, v in fonts.items() if k != "dir"})
    story = fitz.Story(html=html, user_css=css, archive=fitz.Archive(fonts["dir"]))
    tmp = Path(os.environ.get("TEMP", "/tmp"), "nx1_appendix.pdf")
    writer = fitz.DocumentWriter(str(tmp))
    margin_x, top, bottom = 72, 62, 70
    where = fitz.Rect(margin_x, top, page_rect.width - margin_x, page_rect.height - bottom)
    more = True
    while more:
        dev = writer.begin_page(page_rect)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
    writer.close()
    return fitz.open(str(tmp))


def last_printed_page(doc: fitz.Document) -> int:
    page = doc[-1]
    h = page.rect.height
    nums = [
        int(b[4].strip())
        for b in page.get_text("blocks")
        if b[1] > h - 80 and re.fullmatch(r"\s*\d+\s*", b[4])
    ]
    return max(nums) if nums else doc.page_count - OFFSET


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("treatise")
    ap.add_argument("--md", default="docs/NOMENCLATURE_COMPLEMENT.md")
    ap.add_argument("--out", default="docs/Agent_Mechanics_Treatise_EN.pdf")
    ap.add_argument("--fonts", default=None)
    args = ap.parse_args(argv)

    book = fitz.open(args.treatise)
    fonts = find_fonts(args.fonts)
    md_text = Path(args.md).read_text(encoding="utf-8")
    appendix = render_pages(md_text, book[0].rect, fonts)
    first_num = last_printed_page(book) + 1
    first_pdf = book.page_count + 1

    serif = str(Path(fonts["dir"], fonts["serif"]))
    for i, page in enumerate(appendix):
        page.insert_font(fontname="dvs", fontfile=serif)
        w = page.rect.width
        page.insert_text((54, 39), "Agent Mechanics", fontname="dvs", fontsize=8.6)
        tw = fitz.get_text_length(APPENDIX_TITLE, fontname="helv", fontsize=8.6)
        page.insert_text((w - 54 - tw * 1.08, 39), APPENDIX_TITLE, fontname="dvs", fontsize=8.6)
        num = str(first_num + i)
        page.insert_text((w / 2 - 3 * len(num), 745), num, fontname="dvs", fontsize=9.5)
    book.insert_pdf(appendix)

    # Contents row, in the Contents page's own face and rhythm.
    contents = book[1]
    rows = [
        (s["origin"][1], s["text"])
        for b in contents.get_text("dict")["blocks"]
        for line in b.get("lines", [])
        for s in line["spans"]
        if s["text"].strip()
    ]
    synth = max(y for y, t in rows if t.strip().startswith("Synthesis note"))
    step = 13.6
    y = synth + step  # next row after "Synthesis note"
    contents.insert_text((68, y), "Appendix", fontname="Times-Roman", fontsize=8.6)
    contents.insert_text(
        (132, y), f"{APPENDIX_TITLE} (for the RSI engine)", fontname="Times-Roman", fontsize=8.6
    )
    num = str(first_num)
    contents.insert_text(
        (544 - fitz.get_text_length(num, fontname="Times-Roman", fontsize=8.6), y),
        num,
        fontname="Times-Roman",
        fontsize=8.6,
    )

    toc = [[1, "Cover", 1], [1, "Contents", 2]]
    for title, printed, pdf_page in PARTS:
        toc.append([1, title, pdf_page if pdf_page else printed + OFFSET])
    toc.append([1, f"Appendix · {APPENDIX_TITLE}", first_pdf])
    book.set_toc(toc)
    meta = dict(book.metadata or {})
    meta.update(
        {
            "title": "Agent Mechanics",
            "author": "Andrés Ulloa de la Torre",
            "subject": "Complete treatise with the NX-1 nomenclature complement",
        }
    )
    book.set_metadata(meta)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    book.save(args.out, garbage=4, deflate=True, clean=True)
    print(
        f"wrote {args.out}: {book.page_count} pages ({appendix.page_count} appendix pages from printed page {first_num})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
