# vector_store_loader.py
import pickle
from pathlib import Path

def load_vector_store(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Vector store not found at {path}")

    with open(path, "rb") as f:
        vector_store = pickle.load(f)

    if "embeddings" not in vector_store or "documents" not in vector_store:
        raise ValueError("Invalid vector store format")

    return vector_store
