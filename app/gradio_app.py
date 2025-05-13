#!/usr/bin/env python
import os
from dotenv import load_dotenv
import gradio as gr

# 1) dotenv laden
from pathlib import Path
root = Path(__file__).parent.parent
load_dotenv(dotenv_path=root/"env"/".env")

# 2) RAG-Pipeline importieren
from app.rag_chain import answer_query

# 3) Gradio-Funktion definieren
def get_answer(question: str) -> str:
    if not question:
        return "Please enter a question."
    return answer_query(question)

# 4) Gradio Interface
iface = gr.Interface(
    fn=get_answer,
    inputs=gr.Textbox(lines=2, placeholder="Ask a Catan rule question..."),
    outputs="text",
    title="Catan RAG Assistant",
    description="Ask any Settlers of Catan rule question and get a context-aware answer."
)

if __name__ == "__main__":
    # Port 7860 ist der Standard bei HF Spaces
    iface.launch(server_name="0.0.0.0", server_port=7860)
