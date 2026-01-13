# config.py
from sentence_transformers import SentenceTransformer
import random
import numpy as np
import torch

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5
RANDOM_SEED = 42

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)
