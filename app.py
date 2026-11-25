import streamlit as st

from src.pipeline import ingest_pdf, query_documents
from src.retriever import initialize_models

st.set_page_config(page_title="QnA-Agent — RAG Sandbox", layout="wide")


@st.cache_resource
def get_embedding_model():
    """Load the bi-encoder once and keep it cached across reruns."""
    return initialize_models()


# Initialize session state variables
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "index" not in st.session_state:
    st.session_state.index = None
if "indexed_meta" not in st.session_state:
    st.session_state.indexed_meta = ""

st.title("PDF Retrieval Sandbox (Exact k-NN)")

# --- Sidebar: Document & Chunking Controls ---
with st.sidebar:
    st.header("1. Document & Chunking")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    chunk_size = st.slider(
        "Chunk Size (characters)",
        min_value=200,
        max_value=2000,
        value=800,
        step=100,
    )
    overlap = st.slider(
        "Chunk Overlap (characters)",
        min_value=0,
        max_value=500,
        value=120,
        step=20,
    )

    if overlap >= chunk_size:
        st.error("Overlap must be smaller than chunk size.")

    build_clicked = st.button(
        "Build / Rebuild Index",
        type="primary",
        disabled=(uploaded_file is None or overlap >= chunk_size),
        use_container_width=True,
    )

    st.divider()
    st.header("2. Search Settings")
    top_k = st.slider("Top-K Results", min_value=1, max_value=15, value=5)

# Load cached model
with st.spinner("Loading embedding model..."):
    model = get_embedding_model()

# --- Ingestion Trigger ---
if build_clicked and uploaded_file is not None:
    with st.spinner(f"Chunking and embedding '{uploaded_file.name}'..."):
        uploaded_file.seek(0)  # Reset file pointer before reading
        chunks, index = ingest_pdf(
            file_obj=uploaded_file,
            source_name=uploaded_file.name,
            embedding_model=model,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.indexed_meta = (
            f"{uploaded_file.name} | {len(chunks)} chunks "
            f"(size={chunk_size}, overlap={overlap})"
        )

# --- Status Banner ---
if st.session_state.index is not None:
    st.success(f"Active Index: **{st.session_state.indexed_meta}**")
else:
    st.info("Upload a PDF in the sidebar and click **Build / Rebuild Index** to start.")

# --- Query & Results Area ---
question = st.text_input(
    "Ask a question about your document:",
    placeholder="e.g., What happened during the Treaty of Hudaybiyyah?",
)

if question and st.session_state.index is not None:
    with st.spinner("Searching..."):
        results = query_documents(
            question=question,
            chunks=st.session_state.chunks,
            index=st.session_state.index,
            embedding_model=model,
            top_k=top_k,
        )

    st.subheader(f"Top {len(results)} Retrieved Chunks")
    for rank, res in enumerate(results, start=1):
        header = (
            f"#{rank} — Cosine Score: {res['score']:.4f} | "
            f"Page {res['page_no']} | Chunk #{res['idx']}"
        )
        with st.expander(header, expanded=(rank <= 3)):
            st.write(res["text"])