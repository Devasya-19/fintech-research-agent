import re

# Typesetting artifact such as "%%transmsg*** ... ***%%pcmsg|64 |...", removed to end of line.
ARTIFACT = re.compile(r"%%\w*msg.*")


def clean_text(text):
    text = ARTIFACT.sub("", text)
    # Drop lines that contain only a single dot (table-of-contents leaders).
    lines = [ln for ln in text.split("\n") if ln.strip() != "."]
    return "\n".join(lines).strip()


def digit_count(text):
    return sum(ch.isdigit() for ch in text)