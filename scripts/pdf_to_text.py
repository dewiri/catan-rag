# scripts/pdf_to_text.py
import pdfplumber, pathlib

RAW = pathlib.Path("data/raw")
OUT = pathlib.Path("data/text")
OUT.mkdir(exist_ok=True, parents=True)

for pdf in RAW.glob("*.pdf"):
    text = ""
    with pdfplumber.open(pdf) as doc:
        for p in doc.pages:
            text += p.extract_text() + "\n"
    (OUT / f"{pdf.stem}.txt").write_text(text, encoding="utf-8")