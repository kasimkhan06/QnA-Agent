import faiss
from sentence_transformers import SentenceTransformer

from src.chunker import chunk_pages, extract_pdf_pages
from src.retriever import (
    find_similarity,
    generate_embeddings,
    initialize_models,
    prepare_data,
    process_query,
    store_embeddings,
)


def ingest_pdf(
    file_obj,
    source_name: str,
    embedding_model: SentenceTransformer,
    chunk_size: int = 800,
    overlap: int = 120,
) -> tuple[list[dict], faiss.IndexFlatIP]:
    """Run full ingestion: PDF -> pages -> chunks -> embeddings -> FAISS index."""
    pages = extract_pdf_pages(file_obj)
    chunks = chunk_pages(
        pages,
        source_name=source_name,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    if not chunks:
        raise ValueError(f"No extractable text found in {source_name}.")

    text_chunks = prepare_data(chunks)
    document_embeddings = generate_embeddings(text_chunks, embedding_model)
    index = store_embeddings(document_embeddings)

    return chunks, index


def query_documents(
    question: str,
    chunks: list[dict],
    index: faiss.IndexFlatIP,
    embedding_model: SentenceTransformer,
    top_k: int = 5,
) -> list[dict]:
    """Encode a user question and retrieve the top-k matching chunks."""
    query_embedding = process_query(question, embedding_model)
    return find_similarity(
        query_embedding=query_embedding,
        index=index,
        chunks=chunks,
        top_k=top_k,
    )


if __name__ == "__main__":
    pdf_path = "data/English_ArRaheeq_AlMakhtum_THE_SEALED_NECTAR.pdf"

    # 1. Load model once
    model = initialize_models()

    # 2. Ingest PDF into FAISS
    chunks, index = ingest_pdf(
        file_obj=pdf_path,
        source_name="Sealed Nectar",
        embedding_model=model,
        chunk_size=1000,
        overlap=150,
    )
    print(f"Indexed {index.ntotal} chunks.")

    # 3. Query the pipeline
    question = "What happened during the Treaty of Hudaybiyyah?"
    results = query_documents(question, chunks, index, model, top_k=5)

    print(f"\nQuery: {question}\n")
    for rank, res in enumerate(results, start=1):
        print(f"#{rank} | Score: {res['score']:.4f} | Source: {res['source']} | Page: {res['page_no']}")
        print(res["text"][:250] + "...\n")