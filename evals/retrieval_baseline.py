from collections import defaultdict

from config import PROCESSED_DIR
from ingestion.io_utils import read_jsonl
from retrieval.embed import get_model
from retrieval.store import get_collection

KS = (1, 3, 5, 10)


def main():
    queries = read_jsonl(PROCESSED_DIR / "queries.jsonl")
    relevance = read_jsonl(PROCESSED_DIR / "relevance.jsonl")
    relevant = defaultdict(set)
    for r in relevance:
        relevant[r["query_id"]].add(r["corpus_id"])

    model = get_model()
    collection = get_collection()

    hits = {k: 0 for k in KS}
    failures = []
    batch = 128
    for i in range(0, len(queries), batch):
        qs = queries[i:i + batch]
        vecs = model.encode([q["text"] for q in qs], normalize_embeddings=True)
        res = collection.query(query_embeddings=vecs.tolist(), n_results=max(KS))
        for q, metas, docs in zip(qs, res["metadatas"], res["documents"]):
            ranked = [m["doc_id"] for m in metas]
            truth = relevant[q["id"]]
            for k in KS:
                if truth & set(ranked[:k]):
                    hits[k] += 1
            if not (truth & set(ranked[:5])) and len(failures) < 3:
                failures.append((q["text"], sorted(truth), ranked[:3], docs[0][:300]))

    print(f"queries evaluated: {len(queries)}")
    for k in KS:
        print(f"hit@{k}: {hits[k] / len(queries):.3f}")

    print("\nSample failures (hit@5 missed):")
    for text, truth, top3, snippet in failures:
        print(f"\nQ: {text}\n  labeled passage: {truth}\n  retrieved top3:  {top3}\n  top chunk: {snippet!r}")


if __name__ == "__main__":
    main()