from datasets import load_dataset

DATASET = "embedding-benchmark/FinQA"


def _first_split(config_name):
    ds = load_dataset(DATASET, config_name)
    return next(iter(ds.values()))


def load_finqa():
    corpus = [{"id": str(r["id"]), "text": r["text"]} for r in _first_split("corpus")]
    queries = [{"id": str(r["id"]), "text": r["text"]} for r in _first_split("queries")]
    relevance = [
        {
            "query_id": str(r["query-id"]),
            "corpus_id": str(r["corpus-id"]),
            "score": float(r["score"]),
        }
        for r in _first_split("default")
    ]
    return corpus, queries, relevance