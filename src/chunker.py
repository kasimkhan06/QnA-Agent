from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_document(document_path: str, chunk_size: int = 800, chunk_overlap: int = 120):
    """preprocesses the document"""

    loader = PyPDFLoader(
        document_path, mode="single"
    )  # import the entire document as single document instead of pages mode='single'
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)

    return chunks


if __name__ == "__main__":

    document_path = "data/English_ArRaheeq_AlMakhtum_THE_SEALED_NECTAR.pdf"
    chunks = chunk_document(document_path)
