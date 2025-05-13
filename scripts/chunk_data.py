import json
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

TXT_DIR = Path("data/text")
OUT_FILE = Path("data/chunks.jsonl")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

with OUT_FILE.open("w", encoding="utf-8") as fout:
    for txt in TXT_DIR.glob("*.txt"):
        content = txt.read_text(encoding="utf-8")
        for idx, chunk in enumerate(splitter.split_text(content)):
            record = {"source": txt.name, "chunk_id": idx, "text": chunk}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")