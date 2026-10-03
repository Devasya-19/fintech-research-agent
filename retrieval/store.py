import json

import chromadb
import numpy as np

from config import CHROMA_PATH, CHROMA_COLLECTION, EMBEDDING_MODEL, PROCESSED_DIR
from ingestion.io_utils import read_jsonl


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    # If your Chroma version rejects "hnsw:space", check its docs for the cosine setting.
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL},
    )
    stored = (collection.metadata or {}).get("embedding_model")
    if stored != EMBEDDING_MODEL:
        raise RuntimeError(
            f"Collection was built with '{stored}' but EMBEDDING_MODEL is '{EMBEDDING_MODEL}'. "
            "Use a new collection name or delete data/chroma and re-index."
        )
    return collection


def index_chunks(batch_size=500):
    chunks = read_jsonl(PROCESSED_DIR / "chunks.jsonl")
    vectors = np.load(PROCESSED_DIR / "embeddings.npy")
    meta = json.loads((PROCESSED_DIR / "embeddings_meta.json").read_text())

    if meta["model"] != EMBEDDING_MODEL:
        raise RuntimeError("Saved embeddings were made with a different model. Re-run retrieval.embed.")
    if meta["chunk_ids"] != [c["chunk_id"] for c in chunks]:
        raise RuntimeError("Embeddings are out of sync with chunks.jsonl. Re-run retrieval.embed.")

    collection = get_collection()
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=vectors[i:i + batch_size].tolist(),
            metadatas=[
                {
                    "doc_id": c["doc_id"],
                    "chunk_index": c["chunk_index"],
                    "has_table": c["has_table"],
                }
                for c in batch
            ],
        )

    print(f"collection count: {collection.count()}  (chunks: {len(chunks)})")
    sample = collection.get(ids=[chunks[0]["chunk_id"], chunks[-1]["chunk_id"]])
    for cid, text in zip(sample["ids"], sample["documents"]):
        print(f"{cid}: {text[:120]!r}")


if __name__ == "__main__":
    index_chunks()