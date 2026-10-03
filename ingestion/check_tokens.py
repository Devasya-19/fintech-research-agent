import statistics

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, PROCESSED_DIR
from ingestion.io_utils import read_jsonl


def main():
    model = SentenceTransformer(EMBEDDING_MODEL)
    tok = model.tokenizer
    limit = model.max_seq_length
    chunks = read_jsonl(PROCESSED_DIR / "chunks.jsonl")
    lengths = sorted(
        len(tok(c["text"], add_special_tokens=True, truncation=False)["input_ids"])
        for c in chunks
    )
    over = sum(1 for n in lengths if n > limit)
    print(f"model limit: {limit} tokens")
    print(f"chunk tokens: median {int(statistics.median(lengths))}, "
          f"p95 {lengths[int(0.95 * len(lengths))]}, max {lengths[-1]}")
    print(f"chunks over the limit: {over} of {len(lengths)} ({100 * over / len(lengths):.1f}%)")


if __name__ == "__main__":
    main()