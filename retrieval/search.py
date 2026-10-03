def search(collection, model, query, k=5):
    vec = model.encode([query], normalize_embeddings=True)
    res = collection.query(query_embeddings=vec.tolist(), n_results=k)
    return [
        {
            "chunk_id": cid,
            "doc_id": meta["doc_id"],
            "text": doc,
            "distance": dist,
        }
        for cid, meta, doc, dist in zip(
            res["ids"][0], res["metadatas"][0], res["documents"][0], res["distances"][0]
        )
    ]