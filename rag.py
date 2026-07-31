import re
import sqlite3
import numpy as np
from foundry_local_sdk import Configuration, FoundryLocalManager

DB_PATH = "coldchain.db"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"
CHAT_MODEL = "phi-3.5-mini"

_emb_client = None
_chat_client = None
_sources = []
_documents = []
_doc_embeddings = []


def load_chunks():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT source, content, embedding FROM chunks").fetchall()
    conn.close()
    sources = [r[0] for r in rows]
    documents = [r[1] for r in rows]
    embeddings = [np.frombuffer(r[2], dtype=np.float32) for r in rows]
    return sources, documents, embeddings


def load_models():
    config = Configuration(app_name="cold_chain_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    emb_model = manager.catalog.get_model(EMBEDDING_MODEL)
    emb_model.download()
    emb_model.load()

    chat_model = manager.catalog.get_model(CHAT_MODEL)
    chat_model.download()
    chat_model.load()

    return emb_model.get_embedding_client(), chat_model.get_chat_client()


def init():
    global _emb_client, _chat_client, _sources, _documents, _doc_embeddings
    _sources, _documents, _doc_embeddings = load_chunks()
    _emb_client, _chat_client = load_models()
    return _sources, _documents


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def get_top_chunks(query_embedding, top_k):
    scores = [(i, cosine_similarity(query_embedding, e))
              for i, e in enumerate(_doc_embeddings)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


SCORE_THRESHOLD = 0.35

def answer_query(question, top_k=3):
    q_vec = _emb_client.generate_embedding(question).data[0].embedding
    results = get_top_chunks(q_vec, top_k)

    # Guard: if the best match is too weak, refuse before asking the model
    if not results or results[0][1] < SCORE_THRESHOLD:
        return "I don't have that information in my knowledge base.", results

    context = "\n".join(f"- {_documents[i]}" for i, _ in results)

    messages = [
        {"role": "system", "content":
            "Answer the question using only the context below. "
            "Always answer in English. "
            "If the context does not contain the answer, say exactly: "
            "I don't have that information in my knowledge base.\n\n"
            f"Context:\n{context}"},
        {"role": "user", "content": question},
    ]
    response = _chat_client.complete_chat(messages)
    return strip_think(response.choices[0].message.content), results