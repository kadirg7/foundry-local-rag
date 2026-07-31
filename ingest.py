import os
import sqlite3
import numpy as np
from foundry_local_sdk import Configuration, FoundryLocalManager

DB_PATH = "coldchain.db"
DOCS_DIR = "docs"
CHUNK_SIZE = 200
OVERLAP = 30
BATCH_SIZE = 16


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(words), step):
        piece = words[i:i + chunk_size]
        if len(piece) >= 30:
            chunks.append(" ".join(piece))
    return chunks


# 1 - Read .txt files and chunk them
all_chunks = []

for name in os.listdir(DOCS_DIR):
    if not name.endswith(".txt"):
        continue
    with open(os.path.join(DOCS_DIR, name), encoding="utf-8") as f:
        text = f.read()
    pieces = chunk_text(text)
    all_chunks.extend((name, p) for p in pieces)
    print(f"{name}: {len(pieces)} chunks")

print(f"Total: {len(all_chunks)} chunks")

# 2 - Load the embedding model
config = Configuration(app_name="cold_chain_rag")
FoundryLocalManager.initialize(config)
manager = FoundryLocalManager.instance

model = manager.catalog.get_model("qwen3-embedding-0.6b")
model.download()
model.load()
client = model.get_embedding_client()
print("Embedding model ready.")

# 3 - Generate embeddings in batches
embeddings = []
for i in range(0, len(all_chunks), BATCH_SIZE):
    batch_texts = [c for _, c in all_chunks[i:i + BATCH_SIZE]]
    res = client.generate_embeddings(batch_texts)
    embeddings.extend(item.embedding for item in res.data)
    print(f"\rEmbedded {len(embeddings)}/{len(all_chunks)}", end="", flush=True)

print()

# 4 - Write to database
conn = sqlite3.connect(DB_PATH)
conn.execute("PRAGMA journal_mode=WAL;")
conn.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id        INTEGER PRIMARY KEY,
        source    TEXT,
        content   TEXT NOT NULL,
        embedding BLOB NOT NULL
    );
""")
conn.execute("DELETE FROM chunks;")

for (source, content), vec in zip(all_chunks, embeddings):
    blob = np.array(vec, dtype=np.float32).tobytes()
    conn.execute(
        "INSERT INTO chunks (source, content, embedding) VALUES (?, ?, ?)",
        (source, content, blob)
    )

conn.commit()
count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
print(f"{count} chunks saved.")

conn.close()
model.unload()