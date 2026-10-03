import ast


def split_segments(text):
    """Split a passage into ordered ('text', str) and ('table', str) segments."""
    segments = []
    pos = 0
    while True:
        start = text.find("[[", pos)
        if start == -1:
            break
        end = text.find("]]", start)
        if end == -1:
            break
        end += 2
        before = text[pos:start]
        if before.strip():
            segments.append(("text", before))
        segments.append(("table", text[start:end]))
        pos = end
    rest = text[pos:]
    if rest.strip():
        segments.append(("text", rest))
    return segments


def render_table(raw):
    """Turn the table's list-of-lists text into one line per row. Returns (text, parsed_ok)."""
    try:
        rows = ast.literal_eval(raw)
        lines = [" | ".join(str(c).strip() for c in row) for row in rows]
        return "\n".join(lines), True
    except (ValueError, SyntaxError):
        return raw, False