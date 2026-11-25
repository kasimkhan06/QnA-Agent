# QnA-Agent (`v0.1.0` — Baseline PDF Retrieval Sandbox)

A modular, local-first Retrieval-Augmented Generation (RAG) pipeline built from scratch in Python.  
This baseline release (`v0.1.0`) implements end-to-end PDF ingestion, character-level sliding-window chunking, dense asymmetric embeddings (`Qwen/Qwen3-Embedding-0.6B`), exact k-NN vector search (`FAISS`), and an interactive Streamlit sandbox to experiment with chunking hyperparameters.

---

## Architecture & Project Structure

```text
QnA-Agent/
├── app.py                 # Streamlit UI for testing chunking & retrieval interactively
├── pyproject.toml         # Project dependencies managed via uv
├── uv.lock                # Locked dependency versions
├── pyrightconfig.json     # Pylance exclusions to prevent .venv indexing spikes
└── src/
    ├── chunker.py         # PDF page extraction & overlapping character-based chunking
    ├── retriever.py       # Bi-encoder initialization, embedding generation & FAISS IndexFlatIP search
    └── pipeline.py        # Orchestration connecting ingestion and query workflows