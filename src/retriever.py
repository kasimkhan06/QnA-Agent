import torch
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings


def initialize_embeddings(
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
) -> HuggingFaceEmbeddings:
    """Initialize the bi-encoder with asymmetric query prompt and L2 normalization."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={
            "batch_size": 16,
            "normalize_embeddings": True,
        },
        query_encode_kwargs={
            "prompt_name": "query",
            "normalize_embeddings": True,
        },
    )


def build_vectorstore(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
) -> FAISS:
    """Embed Document chunks and store them in a LangChain FAISS vector store."""
    return FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )


def get_retriever(
    vectorstore: FAISS,
    top_k: int = 5,
) -> VectorStoreRetriever:
    """Return a LangChain Runnable retriever over the FAISS vector store."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )


def search_with_scores(
    question: str,
    vectorstore: FAISS,
    top_k: int = 5,
) -> list[tuple[Document, float]]:
    """Retrieve top-k Documents along with their cosine similarity scores."""
    return vectorstore.similarity_search_with_score(question, k=top_k)
