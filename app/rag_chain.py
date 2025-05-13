#!/usr/bin/env python
import os
from pathlib import Path
from dotenv import load_dotenv

# —————————————————————————————
# 1) .env aus env/.env laden
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
# 3) Aktuelle Importe aus langchain-openai
# —————————————————————————————
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# —————————————————————————————
# 4) LLM- und Embedding-Instanzen erstellen
# —————————————————————————————
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.0,
    openai_api_key=api_key
)

emb = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=api_key
)

# —————————————————————————————
# 5) FAISS-Index laden (mit Deserialisierung explizit erlauben)
# —————————————————————————————
vector_path = root / os.getenv("VECTOR_STORE_PATH", "data/faiss_index")
store = FAISS.load_local(
    folder_path=vector_path,
    embeddings=emb,
    allow_dangerous_deserialization=True
)

# —————————————————————————————
# 6) RAG-Chain aufsetzen
# —————————————————————————————
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",                 # packt alle relevanten Chunks in den Prompt
    retriever=store.as_retriever(search_kwargs={"k": 5})
)

# —————————————————————————————
# 7) Hilfsfunktion
# —————————————————————————————
def answer_query(query: str) -> str:
    """Return an answer string for the given Catan question."""
    return qa.invoke({"query": query}).get("result")

# —————————————————————————————
# 8) Testaufruf (optional)
# —————————————————————————————
if __name__ == "__main__":
    sample = "When will the game end?"
    print("Question:", sample)
    print("Answer:", answer_query(sample))