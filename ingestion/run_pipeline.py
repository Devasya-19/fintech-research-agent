import hashlib

from config import PROCESSED_DIR, CHUNK_MAX_WORDS, CHUNK_OVERLAP_SENTENCES
from ingestion.load import load_finqa
from ingestion.validate import validate
from ingestion.clean import clean_text, digit_count
from ingestion.chunk import chunk_passage
from ingestion.io_utils import write_jsonl


def main():
    corpus, queries, relevance = load_finqa()
    validate(corpus, queries, relevance)

    # Clean
    before = {r["id"]: len(r["text"]) for r in corpus}
    digit_changed = []
    for r in corpus:
        original = r["text"]
        r["text"] = clean_text(original)
        if digit_count(original) != digit_count(r["text"]):
            digit_changed.append(r["id"])
    for pid in ("000000001", "000000003"):
        row = next((r for r in corpus if r["id"] == pid), None)
        if row:
            print(f"passage {pid}: {before[pid]} -> {len(row['text'])} chars")
    print(f"passages whose digit count changed: {len(digit_changed)} {digit_changed[:10]}")

    # Chunk
    all_chunks, total_tables, failed_tables = [], 0, 0
    for r in corpus:
        chunks, n_tables, n_failed = chunk_passage(
            r["id"], r["text"], CHUNK_MAX_WORDS, CHUNK_OVERLAP_SENTENCES
        )
        if not chunks:
            raise ValueError(f"passage {r['id']} produced no chunks")
        all_chunks.extend(chunks)
        total_tables += n_tables
        failed_tables += n_failed
    print(f"chunks: {len(all_chunks)} from {len(corpus)} passages")
    print(f"tables: {total_tables} found, {failed_tables} failed to parse")

    # Write
    write_jsonl(PROCESSED_DIR / "corpus.jsonl", corpus)
    write_jsonl(PROCESSED_DIR / "queries.jsonl", queries)
    write_jsonl(PROCESSED_DIR / "relevance.jsonl", relevance)
    write_jsonl(PROCESSED_DIR / "chunks.jsonl", all_chunks)

    digest = hashlib.md5((PROCESSED_DIR / "chunks.jsonl").read_bytes()).hexdigest()
    print(f"chunks.jsonl md5: {digest}  (should be identical on every run)")


if __name__ == "__main__":
    main()