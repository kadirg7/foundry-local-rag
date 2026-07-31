import re
import sqlite3
import numpy as np
from foundry_local_sdk import Configuration, FoundryLocalManager

DB_PATH = "coldchain.db"
TOP_K = 3
SHOW_SCORES = True

# 1 - Load chunks from the database
conn = sqlite3.connect(DB_PATH)
rows = conn.execute("SELECT source, content, embedding FROM chunks").fetchall()
conn.close()

if not rows:
    raise SystemExit("Database is empty. Run: python ingest.py")

sources = [r[0] for r in rows]
documents = [r[1] for r in rows]
doc_embeddings = [np.frombuffer(r[2], dtype=np.float32) for r in rows]

print(f"{len(documents)} chunks loaded.")

# 2 - Load models
config = Configuration(app_name="cold_chain_rag")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance

emb_model = manager.catalog.get_model("qwen3-embedding-0.6b")
emb_model.download(lambda p: print(f"\rEmbedding model: {p:.0f}%", end="", flush=True))
print()
emb_model.load()
emb_client = emb_model.get_embedding_client()

chat_model = manager.catalog.get_model("phi-3.5-mini")
chat_model.download(lambda p: print(f"\rChat model: {p:.0f}%", end="", flush=True))
print()
chat_model.load()
chat_client = chat_model.get_chat_client()
print("Models ready.")


# 3 - Retrieval
def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_top_chunks(query_embedding, top_k=TOP_K):
    scores = [(i, cosine_similarity(query_embedding, e))
              for i, e in enumerate(doc_embeddings)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# 4 - Generate answer
def answer_query(question, top_k=TOP_K):
    q_vec = emb_client.generate_embedding(question).data[0].embedding
    results = get_top_chunks(q_vec, top_k)

    context = "\n".join(f"- {documents[i]}" for i, _ in results)

    messages = [
        {"role": "system", "content":
            "Answer the question using only the context below. "
            "Answer in the same language as the question. "
            "If the context does not contain the answer, say exactly: "
            "I don't have that information.\n\n"
            f"Context:\n{context}"},
        {"role": "user", "content": question},
    ]
    response = chat_client.complete_chat(messages)

    return strip_think(response.choices[0].message.content), results


# 5 - Interactive loop
print("\nAsk a question about the cold chain. Type 'quit' to exit.\n")

try:
    while True:
        question = input("> ").strip()

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        try:
            answer, results = answer_query(question)
        except Exception as e:
            print(f"\nError: {e}")
            print("Try asking again.\n")
            continue
        print(f"\n{answer}\n")

        if SHOW_SCORES:
            for i, score in results:
                print(f"   [{score:.3f}] {sources[i]} | {documents[i][:70]}...")
            print()

except KeyboardInterrupt:
    print("\nInterrupted.")

finally:
    emb_model.unload()
    chat_model.unload()
    print("Models unloaded.")