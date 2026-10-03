import statistics
from collections import Counter


def validate(corpus, queries, relevance):
    errors = []

    for name, rows in (("corpus", corpus), ("queries", queries)):
        ids = [r["id"] for r in rows]
        dupes = [i for i, c in Counter(ids).items() if c > 1]
        if dupes:
            errors.append(f"{name}: {len(dupes)} duplicate ids, e.g. {dupes[:3]}")
        empty = [r["id"] for r in rows if not r["text"] or not r["text"].strip()]
        if empty:
            errors.append(f"{name}: {len(empty)} empty texts, e.g. {empty[:3]}")

    corpus_ids = {r["id"] for r in corpus}
    query_ids = {r["id"] for r in queries}
    rel_queries = {r["query_id"] for r in relevance}
    rel_passages = {r["corpus_id"] for r in relevance}

    missing_q = rel_queries - query_ids
    if missing_q:
        errors.append(f"relevance: {len(missing_q)} query ids not in queries")
    missing_c = rel_passages - corpus_ids
    if missing_c:
        errors.append(f"relevance: {len(missing_c)} corpus ids not in corpus")
    unlabeled = query_ids - rel_queries
    if unlabeled:
        errors.append(f"{len(unlabeled)} queries have no relevance label")

    if errors:
        raise ValueError("Validation failed:\n- " + "\n- ".join(errors))

    lengths = sorted(len(r["text"]) for r in corpus)
    print("Validation passed")
    print(f"  corpus passages:             {len(corpus)}")
    print(f"  queries:                     {len(queries)}")
    print(f"  relevance rows:              {len(relevance)}")
    print(f"  distinct passages labeled:   {len(rel_passages)}")
    print(f"  avg labeled rows/passage:    {len(relevance) / len(rel_passages):.2f}")
    print(
        f"  passage length (chars):      min {lengths[0]}, "
        f"median {int(statistics.median(lengths))}, max {lengths[-1]}"
    )