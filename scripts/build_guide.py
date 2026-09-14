#!/usr/bin/env python3
"""Assemble docs/guide/*.md into one styled PDF.

There is no pandoc, wkhtmltopdf or weasyprint on the target machine, and installing
a LaTeX toolchain to typeset a study guide would be absurd. Chromium is already
present (Brave), and Chromium's ``--print-to-pdf`` is a competent print engine, so
the pipeline is: Markdown -> HTML with print CSS -> headless Chromium -> PDF.

Usage:
    python scripts/build_guide.py [--open]
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import time
import sys
from pathlib import Path

try:
    import markdown
except ImportError:  # pragma: no cover
    sys.exit("missing dependency: pip install markdown pygments")

ROOT = Path(__file__).resolve().parent.parent
GUIDE_DIR = ROOT / "docs" / "guide"
BUILD_DIR = ROOT / "docs" / "build"
TITLE = "Sightline"
SUBTITLE = "A Field Guide to Permission-Aware Retrieval"

# Chromium candidates, in preference order. Brave is Chromium, so it prints
# identically; we take whatever the machine actually has.
BROWSERS = [
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
]

CSS = """
@page { size: A4; margin: 20mm 18mm 22mm 18mm; }

:root {
  --ink: #16181d;
  --muted: #5b6270;
  --rule: #d9dde5;
  --accent: #1f4e79;
  --code-bg: #f5f6f8;
  --callout: #fbfaf4;
}

* { box-sizing: border-box; }

body {
  font-family: "Charter", "Georgia", "Iowan Old Style", serif;
  font-size: 10.6pt;
  line-height: 1.58;
  color: var(--ink);
  margin: 0;
  -webkit-font-smoothing: antialiased;
}

/* ---------- title page ---------- */
.titlepage {
  height: 246mm;
  display: flex;
  flex-direction: column;
  justify-content: center;
  page-break-after: always;
  border-top: 3px solid var(--accent);
}
.titlepage h1 {
  font-size: 44pt; margin: 0 0 4pt 0; letter-spacing: -0.02em;
  border: none; color: var(--accent);
}
.titlepage .sub { font-size: 15pt; color: var(--muted); margin: 0 0 28pt 0; font-style: italic; }
.titlepage .blurb { font-size: 11pt; max-width: 118mm; color: var(--ink); }
.titlepage .meta { margin-top: 34pt; font-size: 9.5pt; color: var(--muted); }

/* ---------- table of contents ---------- */
.toc { page-break-after: always; }
.toc h2 { border: none; margin-bottom: 14pt; }
.toc ol { list-style: none; padding-left: 0; counter-reset: ch; }
.toc li { counter-increment: ch; padding: 3.5pt 0; border-bottom: 1px dotted var(--rule); font-size: 10.5pt; }
.toc li::before { content: counter(ch) ". "; color: var(--accent); font-weight: 600; }

/* ---------- headings ---------- */
h1, h2, h3, h4 { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; line-height: 1.25; }
h2 {
  font-size: 19pt; margin: 0 0 12pt 0; padding-top: 4pt;
  color: var(--accent); border-bottom: 2px solid var(--rule); padding-bottom: 5pt;
  page-break-before: always; page-break-after: avoid;
}
h2:first-of-type { page-break-before: avoid; }
h3 {
  font-size: 12.4pt; margin: 17pt 0 6pt 0; color: var(--ink);
  page-break-after: avoid;
}
/* The six recurring section headings get a quiet marker so the structure is
   scannable when flipping through the printed guide. */
h3 { border-left: 3px solid var(--accent); padding-left: 7pt; }
h4 { font-size: 10.8pt; margin: 12pt 0 4pt 0; page-break-after: avoid; }

p { margin: 0 0 8pt 0; orphans: 3; widows: 3; }

/* ---------- code ---------- */
code {
  font-family: "SF Mono", Menlo, Consolas, monospace;
  font-size: 8.9pt; background: var(--code-bg);
  padding: 1pt 3pt; border-radius: 3px;
}
pre {
  background: var(--code-bg); border: 1px solid var(--rule); border-left: 3px solid var(--accent);
  padding: 8pt 10pt; border-radius: 4px; overflow-x: auto;
  page-break-inside: avoid; margin: 9pt 0;
}
pre code { background: none; padding: 0; font-size: 8.4pt; line-height: 1.45; }

/* ---------- tables ---------- */
table {
  border-collapse: collapse; width: 100%; margin: 10pt 0;
  font-size: 9.3pt; page-break-inside: avoid;
}
th {
  background: var(--accent); color: #fff; text-align: left;
  padding: 5pt 7pt; font-family: Helvetica, Arial, sans-serif; font-size: 9pt;
}
td { padding: 4.5pt 7pt; border-bottom: 1px solid var(--rule); vertical-align: top; }
tr:nth-child(even) td { background: #fafbfc; }

blockquote {
  margin: 10pt 0; padding: 7pt 12pt; background: var(--callout);
  border-left: 3px solid #c9a227; color: #3b3a32; page-break-inside: avoid;
}
blockquote p:last-child { margin-bottom: 0; }

ul, ol { margin: 0 0 8pt 0; padding-left: 19pt; }
li { margin-bottom: 3.5pt; }

hr { border: none; border-top: 1px solid var(--rule); margin: 14pt 0; }
a { color: var(--accent); text-decoration: none; }
strong { font-weight: 700; }

/* Interview Q&A blocks read better with a little air and must not split badly. */
h3[id*="interview"] + ol > li,
h3[id*="interview"] + ul > li { page-break-inside: avoid; margin-bottom: 7pt; }
"""


def chapter_files() -> list[Path]:
    files = sorted(GUIDE_DIR.glob("*.md"))
    if not files:
        sys.exit(f"no chapters found in {GUIDE_DIR} - has the writing job finished?")
    return files


def render() -> tuple[str, list[str]]:
    """Return (body_html, chapter_titles)."""
    md = markdown.Markdown(
        extensions=["extra", "tables", "fenced_code", "codehilite", "sane_lists", "attr_list"],
        extension_configs={"codehilite": {"guess_lang": False, "noclasses": True}},
    )
    parts: list[str] = []
    titles: list[str] = []
    for path in chapter_files():
        text = path.read_text(encoding="utf-8")
        # Collect every level-2 heading: one file may hold several chapters.
        titles.extend(m.strip() for m in re.findall(r"^##\s+(.+)$", text, flags=re.M))
        md.reset()
        parts.append(md.convert(text))
    return "\n".join(parts), titles


def build_html() -> Path:
    body, titles = render()
    toc = "\n".join(f"<li>{html.escape(t)}</li>" for t in titles)
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{TITLE} - {SUBTITLE}</title>
<style>{CSS}</style></head>
<body>
<section class="titlepage">
  <h1>{TITLE}</h1>
  <p class="sub">{SUBTITLE}</p>
  <p class="blurb">Internal AI search that can only see what you are allowed to see &mdash;
  and proves it. This guide explains every idea the system is built on, starting from
  first principles, and ends each chapter with the questions an interviewer will ask.</p>
  <p class="meta">{len(titles)} chapters &middot; intuition, mechanics, failure modes,
  and interview answers</p>
</section>
<section class="toc"><h2>Contents</h2><ol>{toc}</ol></section>
{body}
</body></html>"""
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    out = BUILD_DIR / "guide.html"
    out.write_text(doc, encoding="utf-8")
    return out


def find_browser() -> str:
    for candidate in BROWSERS:
        if Path(candidate).exists():
            return candidate
    for name in ("chromium", "google-chrome", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    sys.exit("no Chromium-based browser found; install Chrome or Brave to print the PDF")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true", help="open the PDF when done")
    args = ap.parse_args()

    html_path = build_html()
    pdf_path = BUILD_DIR / "Sightline-Field-Guide.pdf"
    browser = find_browser()
    if pdf_path.exists():
        pdf_path.unlink()

    # Headless Chromium writes the PDF and then, on some macOS builds, declines to
    # exit. Waiting on the process is therefore the wrong signal entirely. We watch
    # for the output file to appear and stop growing, then terminate the browser
    # ourselves. An isolated --user-data-dir stops it attaching to a running Brave.
    # Deliberately NOT passing --user-data-dir. It looks like the safe choice, but on
    # this Brave build an isolated profile never finishes initialising headless and no
    # PDF is ever written. Using the default profile works and takes about 40 seconds.
    proc = subprocess.Popen(
        [
            browser,
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            "--virtual-time-budget=20000",
            html_path.as_uri(),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 420
    stable_for = 0.0
    last_size = -1
    try:
        while time.monotonic() < deadline:
            size = pdf_path.stat().st_size if pdf_path.exists() else 0
            # Two consecutive identical, non-trivial sizes means the write finished.
            stable_for = stable_for + 2.0 if size > 20_000 and size == last_size else 0.0
            last_size = size
            if stable_for >= 4.0:
                break
            if proc.poll() is not None and size > 20_000:
                break
            time.sleep(2.0)
    finally:
        # Headless Chromium writes the PDF and then declines to exit on some macOS
        # builds, so waiting on the process is the wrong signal. We stop it ourselves.
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    # Chromium's exit code is unreliable in headless print mode: judge by the output.
    if not pdf_path.exists() or pdf_path.stat().st_size < 5_000:
        sys.exit("PDF was not produced")

    size_kb = pdf_path.stat().st_size // 1024
    print(f"built {pdf_path} ({size_kb} KB)")
    if args.open:
        subprocess.run(["open", str(pdf_path)], check=False)


if __name__ == "__main__":
    main()
