from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings

from src.chunker import chunk_document
from src.retriever import (
    build_vectorstore,
    get_retriever,
    initialize_embeddings,
    search_with_scores,
)


def ingest_pdf(
    document_path: str,
    embeddings: HuggingFaceEmbeddings,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> tuple[list[Document], FAISS]:
    """Load a PDF, recursively split it into Documents, and build a FAISS vector store."""
    chunks = chunk_document(
        document_path=document_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        raise ValueError(f"No extractable text found in {document_path}.")

    vectorstore = build_vectorstore(chunks=chunks, embeddings=embeddings)
    return chunks, vectorstore


def format_docs(docs: list[Document]) -> str:
    """Format retrieved Document chunks into a single readable context block."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_retrieval_chain(vectorstore: FAISS, top_k: int = 5) -> Runnable:
    """Build an LCEL retrieval chain returning question, raw docs, and formatted context."""
    retriever = get_retriever(vectorstore=vectorstore, top_k=top_k)

    retrieval_chain = {
        "docs": retriever,
        "question": RunnablePassthrough(),
    } | RunnableLambda(
        lambda x: {
            "question": x["question"],
            "docs": x["docs"],
            "context": format_docs(x["docs"]),
        }
    )
    return retrieval_chain


if __name__ == "__main__":
    document_path = "data/English_ArRaheeq_AlMakhtum_THE_SEALED_NECTAR.pdf"

    # 1. Initialize embeddings once
    embeddings = initialize_embeddings()

    # 2. Ingest PDF into LangChain FAISS vector store
    chunks, vectorstore = ingest_pdf(
        document_path=document_path,
        embeddings=embeddings,
        chunk_size=800,
        chunk_overlap=120,
    )
    print(f"Indexed {len(chunks)} chunks into FAISS.")

    # 3. Invoke the LCEL retrieval chain
    question = "What happened during the Treaty of Hudaybiyyah?"
    chain = build_retrieval_chain(vectorstore=vectorstore, top_k=5)
    response = chain.invoke(question)

    print(f"\nQuery: {response['question']}\n")
    for rank, doc in enumerate(response["docs"], start=1):
        print(f"#{rank} | Metadata: {doc.metadata}")
        print(doc.page_content[:250] + "...\n")

    # Optional: Inspect cosine similarity scores
    scored_results = search_with_scores(question, vectorstore=vectorstore, top_k=5)
    print("Top cosine scores:", [round(float(score), 4) for _, score in scored_results])
