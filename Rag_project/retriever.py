# retriever.py
from vector_store_loader import load_vector_store
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def retrieve_top_k(question, embedder, vector_store, k):
    question_embedding = embedder.encode([question])
    similarities = cosine_similarity(
        question_embedding, vector_store["embeddings"]
    )[0]

    top_indices = np.argsort(similarities)[::-1][:k]

    results = [
        {
            "text": vector_store["documents"][i],
            "score": similarities[i]
        }
        for i in top_indices
    ]

    return results


class ComplaintRetriever:
    def __init__(self, vector_store_path, embedder, top_k=5):
        self.vector_store = load_vector_store(vector_store_path)
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, question):
        if not question or not isinstance(question, str):
            raise ValueError("Question must be a non-empty string")

        return retrieve_top_k(
            question,
            self.embedder,
            self.vector_store,
            self.top_k
        )
