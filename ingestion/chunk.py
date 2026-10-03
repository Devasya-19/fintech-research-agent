from ingestion.tables import split_segments, render_table


def _split_long(line, max_words):
    words = line.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)] or [line]


def _sentences(text, max_words):
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            out.extend(_split_long(line, max_words))
    return out


def _pack(sentences, max_words, overlap):
    chunks, current, count = [], [], 0
    for s in sentences:
        w = len(s.split())
        if current and count + w > max_words:
            chunks.append(current)
            current = current[-overlap:] if overlap else []
            count = sum(len(x.split()) for x in current)
        current.append(s)
        count += w
    if current:
        chunks.append(current)
    return [" ".join(c) for c in chunks]


def _table_chunks(rendered, caption, max_words):
    rows = [r for r in rendered.split("\n") if r.strip()]
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    head = ([caption] if caption else []) + [header]
    head_words = sum(len(x.split()) for x in head)

    chunks, current, count = [], [], head_words
    for row in body:
        w = len(row.split())
        if current and count + w > max_words:
            chunks.append("\n".join(head + current))
            current, count = [], head_words
        current.append(row)
        count += w
    chunks.append("\n".join(head + current))
    return chunks


def chunk_passage(doc_id, text, max_words, overlap):
    """Returns (chunks, number_of_tables, number_of_tables_that_failed_to_parse)."""
    pieces = []
    last_sentence = ""
    n_tables = n_failed = 0

    for kind, content in split_segments(text):
        if kind == "text":
            sentences = _sentences(content, max_words)
            pieces += [(t, False) for t in _pack(sentences, max_words, overlap)]
            if sentences:
                last_sentence = sentences[-1]
        else:
            n_tables += 1
            rendered, ok = render_table(content)
            if not ok:
                n_failed += 1
            pieces += [(t, True) for t in _table_chunks(rendered, last_sentence, max_words)]

    chunks = [
        {
            "chunk_id": f"{doc_id}_{i}",
            "doc_id": doc_id,
            "chunk_index": i,
            "text": t,
            "has_table": has_table,
            "word_count": len(t.split()),
        }
        for i, (t, has_table) in enumerate(pieces)
    ]
    return chunks, n_tables, n_failed