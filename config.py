import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent


def _path(name, default):
    p = Path(os.getenv(name, default))
    return p if p.is_absolute() else ROOT / p


DATA_DIR = _path("DATA_DIR", "data")
PROCESSED_DIR = _path("PROCESSED_DIR", "data/processed")
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_MAX_WORDS = int(os.getenv("CHUNK_MAX_WORDS", "180"))
CHUNK_OVERLAP_SENTENCES = int(os.getenv("CHUNK_OVERLAP_SENTENCES", "1"))

CHROMA_PATH = _path("CHROMA_PATH", "data/chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "finqa_chunks")
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))