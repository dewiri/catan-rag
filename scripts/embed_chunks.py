#!/usr/bin/env python
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# —————————————————————————————
# 1) .env laden (in env/.env)
# —————————————————————————————
root     = Path(__file__).parent.parent
env_path = root / "env" / ".env"
load_dotenv(dotenv_path=env_path)

# —————————————————————————————
# 2) API-Key holen und validieren
# —————————————————————————————
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(f"OPENAI_API_KEY not found in {env_path}")

# —————————————————————————————
# 3) LangChain-Module importieren
# —————————————————————————————
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# —————————————————————————————
# 4) Chunks einlesen
# —————————————————————————————
chunks_file = root / "data" / "chunks.jsonl"
chunks = []
with open(chunks_file, encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

# Baue Listen mit Texten und Metadaten
texts = [c["text"] for c in chunks]
metas = [{"source": c["source"], "chunk_id": c["chunk_id"]} for c in chunks]

# —————————————————————————————
# 5) Embeddings erzeugen
# —————————————————————————————
emb = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=api_key
)

# —————————————————————————————
# 6) FAISS-Index bauen und speichern
# —————————————————————————————
store = FAISS.from_texts(
    texts=texts,         # Texte
    embedding=emb,       # korrektes Argument: embedding, nicht embeddings
    metadatas=metas      # Metadaten
)

vector_store_path = os.getenv("VECTOR_STORE_PATH", root / "data" / "faiss_index")
store.save_local(vector_store_path)

print(f"✅ FAISS index saved to {vector_store_path}")