import json

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, PROCESSED_DIR
from ingestion.io_utils import read_jsonl


def get_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(model, texts, batch_size=64):
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def main():
    chunks = read_jsonl(PROCESSED_DIR / "chunks.jsonl")
    model = get_model()
    print("max_seq_length:", model.max_seq_length)

    vectors = embed_texts(model, [c["text"] for c in chunks])
    np.save(PROCESSED_DIR / "embeddings.npy", vectors)
    (PROCESSED_DIR / "embeddings_meta.json").write_text(
        json.dumps(
            {
                "model": EMBEDDING_MODEL,
                "count": len(chunks),
                "chunk_ids": [c["chunk_id"] for c in chunks],
            }
        )
    )
    print(f"saved {vectors.shape} embeddings")


if __name__ == "__main__":
    main()