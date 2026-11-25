import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def initialize_models(embedding_model: str = "Qwen/Qwen3-Embedding-0.6B") -> SentenceTransformer:
    """Initializes and returns the bi-encoder model."""
    return SentenceTransformer(embedding_model)


def prepare_data(chunks: list[dict]) -> list[str]:
    """Convert textual chunks with metadata into a list of strings."""
    return [chunk["text"] for chunk in chunks]


def generate_embeddings(text: list[str], embedding_model: SentenceTransformer) -> np.ndarray:
    """Generates L2-normalized float32 embeddings for document chunks."""
    return embedding_model.encode_document(
        text,
        batch_size=16,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")


def store_embeddings(document_embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Stores document embeddings in a FAISS exact inner-product index."""
    dim = document_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(document_embeddings)
    return index


def process_query(query: str, embedding_model: SentenceTransformer) -> np.ndarray:
    """Encodes a user query into a 2D normalized float32 embedding."""
    return embedding_model.encode_query(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")


def find_similarity(
        query_embedding: np.ndarray,
        index: faiss.IndexFlatIP,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
    """Finds top-k most similar chunks for a query embedding."""
    if not chunks:
        return []

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    matched_chunks = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        matched_chunks.append(
            {
                "idx": int(idx),
                "score": float(score),
                "source": chunks[idx].get("source_name", ""),
                "page_no": chunks[idx]["page_no"],
                "text": chunks[idx]["text"],
            }
        )
    return matched_chunks