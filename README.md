# DocuMind AI

<div align="center">

**Intelligent document analysis and retrieval — powered by RAG, Gemini, and ChromaDB**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-Orchestration-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com/)
[![Google Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-LLM-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorDB-FF6B35?style=for-the-badge)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](./LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Prajwallnaik/DocuMind-AI?style=for-the-badge&logo=github)](https://github.com/Prajwallnaik/DocuMind-AI/stargazers)

</div>

---

## Overview / About the Project

**DocuMind AI** is a production-grade **Retrieval-Augmented Generation (RAG)** application that enables natural language querying over user-supplied documents. Upload any PDF or plain-text file and ask questions — DocuMind retrieves the most relevant passages and grounds every answer strictly in the provided content, eliminating hallucination by design.

### Why DocuMind AI?

Large Language Models are powerful, but they fabricate facts when queried on information outside their training data. DocuMind AI addresses this by:

- **Retrieving** only the semantically relevant chunks from the uploaded documents
- **Grounding** every response to retrieved context (temperature = 0)
- **Refusing** to speculate when the answer is not present in the source material

> Built as a full-stack AI portfolio project demonstrating end-to-end RAG pipeline engineering — from raw document ingestion to context-aware answer synthesis.

---

## Architecture / How It Works

```
┌─────────────┐     ┌──────────┐     ┌──────────────┐     ┌────────────┐
│  User Upload │────▶│  Parser  │────▶│   Chunker    │────▶│  Embedder  │
│  (PDF / TXT) │     │ (PyPDF)  │     │ (500 / 50 ov)│     │ MiniLM-L6  │
└─────────────┘     └──────────┘     └──────────────┘     └─────┬──────┘
                                                                  │
                                                          ┌───────▼───────┐
                                                          │   ChromaDB    │
                                                          │ (Vector Store)│
                                                          └───────┬───────┘
┌─────────────┐     ┌──────────────┐     ┌──────────┐           │
│    Answer   │◀────│ Gemini 2.5   │◀────│ Top-K=10 │◀──────────┘
│  (Grounded) │     │ Flash (LLM)  │     │ Retrieval│
└─────────────┘     └──────────────┘     └──────────┘
        ▲                                      ▲
        │              User Query              │
        └──────────────────────────────────────┘
```

### Step-by-Step Pipeline

| Step | Module | Description |
|------|--------|-------------|
| **1. Upload** | `app.py` | User provides PDF or TXT files via the Streamlit sidebar |
| **2. Parse** | `rag/loader.py` | Raw text is extracted from document binaries using `PyPDF` |
| **3. Chunk** | `rag/chunker.py` | Text is segmented into 500-character chunks with 50-character overlap via `RecursiveCharacterTextSplitter` |
| **4. Embed** | `rag/embedder.py` | Chunks are converted to dense vectors using `all-MiniLM-L6-v2` (HuggingFace, local, free) |
| **5. Store** | `rag/vectorstore.py` | Vectors are indexed in an ephemeral `ChromaDB` collection with a unique UUID per session |
| **6. Retrieve** | `rag/vectorstore.py` | The user's query is embedded and the top-10 most similar chunks are fetched |
| **7. Answer** | `rag/chain.py` | Retrieved context and query are passed to `Gemini 2.5 Flash` (temperature=0) via a `RetrievalQA` chain |

---

## Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| **UI** | [Streamlit](https://streamlit.io/) | Reactive web interface, chat history, and file uploader |
| **LLM** | [Google Gemini 2.5 Flash](https://ai.google.dev/) | Answer synthesis with zero-temperature grounding |
| **Embeddings** | `all-MiniLM-L6-v2` via `sentence-transformers` | Free, local, high-quality semantic vector representations |
| **Vector DB** | [ChromaDB](https://www.trychroma.com/) | In-process similarity search and vector indexing |
| **Orchestration** | [LangChain](https://langchain.com/) | Pipeline assembly, `RetrievalQA` chain, and prompt templating |
| **File Parsing** | `PyPDF` | Extraction of text from PDF binaries |
| **Text Splitting** | `RecursiveCharacterTextSplitter` | Context-aware document segmentation |
| **Config** | `python-dotenv` | Secure API key and environment variable management |

---

## Getting Started

### Prerequisites

- Python **3.9+**
- A **Google API key** with access to Gemini (free tier available at [ai.google.dev](https://ai.google.dev/))

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Prajwallnaik/DocuMind-AI.git
cd DocuMind-AI

# 2. Create and activate a virtual environment
python -m venv venv

# On macOS / Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root and add your Google API key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

> **Note:** The `OPENAI_API_KEY` field is optional. If it contains a placeholder value, it is automatically ignored at startup.

### Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

### Usage

1. **Upload** one or more PDF or `.txt` files using the sidebar.
2. Click **"Process Documents"** — the pipeline will parse, chunk, embed, and index the files.
3. Once processing is complete, type a question into the chat input.
4. DocuMind AI will return a concise, grounded answer sourced directly from the uploaded documents.

---

## Project Structure

```text
DocuMind-AI/
│
├── app.py                  # Streamlit UI — main entry point & session management
│
├── rag/                    # Core RAG pipeline package
│   ├── __init__.py
│   ├── loader.py           # Document loading & text extraction (PDF / TXT)
│   ├── chunker.py          # RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
│   ├── embedder.py         # HuggingFace embedding model setup (all-MiniLM-L6-v2)
│   ├── vectorstore.py      # ChromaDB vectorstore creation & top-k retriever
│   └── chain.py            # LangChain RetrievalQA chain with Gemini 2.5 Flash
│
├── assets/                 # Static assets (architecture diagrams, screenshots)
├── chroma_db/              # Local ChromaDB persistence directory
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not committed)
├── .gitignore
└── README.md
```

---

## Features

- **Multi-format support** — Upload and query PDF and plain-text (`.txt`) files
- **Multi-file upload** — Process several documents simultaneously in one session
- **Semantic search** — Dense vector similarity via `all-MiniLM-L6-v2` embeddings
- **Grounded answers** — Gemini 2.5 Flash with temperature=0 for factual, non-hallucinated responses
- **Source transparency** — `return_source_documents=True` exposes exactly which chunks informed each answer
- **Persistent chat history** — Full conversation context is maintained within a session via `st.session_state`
- **Local vector storage** — ChromaDB runs in-process; no external database or cloud dependency required
- **Secure configuration** — API keys managed entirely through `.env` and `python-dotenv`
- **No frontend code** — Entire UI delivered through Streamlit's Python API

---

## Limitations & Known Issues

| Issue | Details |
|-------|---------|
| **No session persistence** | The ChromaDB collection is ephemeral — documents must be re-uploaded after each app restart |
| **Large document performance** | PDFs with 100+ pages may cause slow processing due to sequential embedding |
| **Answer quality ceiling** | Responses are bounded by the quality, completeness, and formatting of the source documents |
| **Context window limits** | Very high `TOP_K` values may exceed the LLM's context window, causing truncation or errors |
| **Single language enforcement** | Queries are prefixed with an English instruction; multilingual documents may yield inconsistent results |
| **No DOCX / CSV support** | Only `.pdf` and `.txt` file formats are currently supported |

---

## Roadmap / Future Improvements

```text
Core Pipeline
  [x] PDF and TXT file ingestion
  [x] ChromaDB local vector store
  [x] LangChain RetrievalQA chain
  [x] Google Gemini 2.5 Flash integration
  [x] Multi-file upload in a single session
  [x] Persistent chat history within session

Planned Enhancements
  [ ] Persistent cross-session memory (database-backed)
  [ ] FAISS as an alternative / faster vector store
  [ ] DOCX and CSV file format support
  [ ] Configurable embedding model selection in the UI
  [ ] Source chunk highlights rendered inline with answers
  [ ] Docker containerization for one-command deployment
  [ ] Deployment to Streamlit Community Cloud
  [ ] Evaluation harness (RAGAS / TruLens) for answer quality scoring
  [ ] Conversational memory with follow-up question support
```

---

## License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute it.

```
MIT License — Copyright (c) 2026 Prajwal Naik
```

See the [LICENSE](./LICENSE) file for full details.


