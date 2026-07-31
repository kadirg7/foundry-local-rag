import re
import pdfplumber

PDF_PATH = "docs/faocoldchain.pdf"
OUT_PATH = "docs/fao_sample.txt"
START_PAGE = 20
END_PAGE = 70


def clean(text):
    if not text:
        return ""
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if re.fullmatch(r"\d+", line):
            continue
        if len(line) < 3:
            continue
        lines.append(line)
    return " ".join(lines)


parts = []
with pdfplumber.open(PDF_PATH) as pdf:
    for i in range(START_PAGE, min(END_PAGE, len(pdf.pages))):
        page = pdf.pages[i]
        mid = page.width / 2
        left = page.crop((0, 0, mid, page.height)).extract_text()
        right = page.crop((mid, 0, page.width, page.height)).extract_text()
        parts.append(clean(left))
        parts.append(clean(right))
        print(f"\rPage {i}", end="", flush=True)

print()
text = " ".join(p for p in parts if p)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write(text)

print(f"{len(text)} characters written to {OUT_PATH}")