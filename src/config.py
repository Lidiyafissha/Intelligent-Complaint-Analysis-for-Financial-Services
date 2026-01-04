import random
import numpy as np

# Reproducibility
RANDOM_SEED = 42

# Sampling
SAMPLE_SIZE = 12000
# Column in the processed CSV that contains product information
STRATIFY_COLUMN = "Product"

# Text
# Column in the processed CSV that contains cleaned complaint text
TEXT_COLUMN = "clean_narrative"
# Identifier column; if your processed CSV lacks an ID column we create one in the notebook
ID_COLUMN = "complaint_id"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Embeddings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Paths
VECTOR_DB_PATH = "vector_db/faiss_index"
