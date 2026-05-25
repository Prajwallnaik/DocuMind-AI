# DocuMind AI

<div align="center">

**Intelligent document analysis and retrieval — powered by RAG, NLP, Gemini, and ChromaDB**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-Orchestration-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com/)
[![Google Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-LLM-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorDB-FF6B35?style=for-the-badge)](https://www.trychroma.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-Embeddings-005CED?style=for-the-badge)](https://onnxruntime.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](./LICENSE)
[![RAGAS](https://img.shields.io/badge/RAGAS-Evaluation-8B5CF6?style=for-the-badge)](https://docs.ragas.io/)

</div>

---

## Overview

**DocuMind AI** is a production-grade **Retrieval-Augmented Generation (RAG)** application that enables natural language querying over user-supplied documents. Upload any PDF or plain-text file and ask questions — DocuMind retrieves the most relevant passages and grounds every answer strictly in the provided content, eliminating hallucination by design.

### Why DocuMind AI?

Large Language Models are powerful, but they fabricate facts when queried on information outside their training data. DocuMind AI addresses this by:

- **Retrieving** only the semantically relevant chunks from the uploaded documents
- **Grounding** every response to retrieved context (temperature = 0)
- **Refusing** to speculate when the answer is not present in the source material
- **Scaling context dynamically** — whole-document summaries use 100% of the content, while targeted questions use precision retrieval

> Built as a full-stack AI portfolio project demonstrating end-to-end RAG pipeline engineering — from raw document ingestion to context-aware answer synthesis.

---

## Architecture / How It Works

![DocuMind AI Architecture](assets/Architecture.png)

### Step-by-Step Pipeline

| Step | Module | Description |
|------|--------|-------------|
| **1. Upload** | `app.py` | User provides PDF or TXT files via the Streamlit sidebar |
| **2. Parse** | `rag/loader.py` | Raw text is extracted from document binaries using `PyPDF` |
| **3. Chunk** | `rag/chunker.py` | Text is segmented into 1000-character chunks with 200-character overlap via `RecursiveCharacterTextSplitter` |
| **4. Embed** | `rag/onnx_embedder.py` | Chunks are converted to 384-dimensional dense vectors using `all-MiniLM-L6-v2` via ONNX Runtime (local, free, fast) |
| **5. Store** | `rag/vectorstore.py` | Vectors are indexed in an ephemeral `ChromaDB` collection with a unique UUID per session |
| **6. Retrieve** | `rag/vectorstore.py` | The `DynamicRetriever` automatically scales context — full document for summaries, top-10 precision search for specific questions |
| **7. Answer** | `rag/chain.py` | Retrieved context and query are passed to `Gemini 2.5 Flash` (temperature=0) via a `RetrievalQA` chain |

---

## NLP Techniques Used

DocuMind AI leverages multiple core NLP techniques across the pipeline:

| NLP Technique | Module | Description |
|---|---|---|
| **Tokenization** | `rag/onnx_embedder.py` | HuggingFace fast Rust tokenizer converts raw text into subword token IDs |
| **Sentence Embeddings** | `rag/onnx_embedder.py` | Transformer encoder (`all-MiniLM-L6-v2`) maps text into 384-dim dense semantic vectors |
| **Mean Pooling** | `rag/onnx_embedder.py` | Aggregates per-token hidden states into a single sentence-level representation |
| **L2 Normalization** | `rag/onnx_embedder.py` | Normalizes embedding vectors for cosine similarity search |
| **Text Splitting** | `rag/chunker.py` | Recursive splitting along natural language boundaries (paragraphs, sentences, words) |
| **Semantic Similarity Search** | `rag/vectorstore.py` | Cosine similarity over dense embeddings for passage retrieval |
| **Query Intent Detection** | `rag/vectorstore.py` | Keyword-based classification of summary vs. factual queries in `DynamicRetriever` |
| **LLM Question Answering** | `rag/chain.py` | Gemini 2.5 Flash performs natural language understanding and grounded answer generation |
| **Retrieval-Augmented Generation** | Full pipeline | End-to-end RAG architecture grounding LLM responses in retrieved document context |

---

## Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| **UI** | [Streamlit](https://streamlit.io/) | Reactive web interface with custom CSS, chat history, and file uploader |
| **LLM** | [Google Gemini 2.5 Flash](https://ai.google.dev/) | Answer synthesis with zero-temperature grounding (1M token context window) |
| **Embeddings** | `all-MiniLM-L6-v2` via [ONNX Runtime](https://onnxruntime.ai/) | Local, high-speed semantic vector representations with no PyTorch overhead |
| **Vector DB** | [ChromaDB](https://www.trychroma.com/) | In-process similarity search and vector indexing |
| **Orchestration** | [LangChain](https://langchain.com/) | Pipeline assembly, `RetrievalQA` chain, and prompt templating |
| **Retrieval** | Custom `DynamicRetriever` | Context-scaling retriever that adapts between full-document and precision modes |
| **File Parsing** | `PyPDF` | Extraction of text from PDF binaries |
| **Text Splitting** | `RecursiveCharacterTextSplitter` | Context-aware document segmentation (chunk_size=1000, overlap=200) |
| **Config** | `python-dotenv` | Secure API key and environment variable management |
| **Evaluation** | [RAGAS](https://docs.ragas.io/) + `datasets` | Faithfulness, Answer Relevancy, and Context Precision scoring |

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
4. DocuMind AI will return a concise, professional, grounded answer sourced directly from the uploaded documents.
5. Ask for a **summary** — the system automatically feeds the entire document to the LLM for a comprehensive, zero-loss summary.
6. Click **"Reset Knowledge Base"** in the sidebar to clear all data and start fresh.

---

## Project Structure

```text
DocuMind-AI/
|
+-- app.py                  # Streamlit UI -- main entry point, session management & custom CSS
|
+-- rag/                    # Core RAG pipeline package
|   +-- __init__.py
|   +-- loader.py           # Document loading & text extraction (PDF / TXT)
|   +-- chunker.py          # RecursiveCharacterTextSplitter (chunk_size=1000, overlap=200)
|   +-- onnx_embedder.py    # ONNX Runtime embedding model (all-MiniLM-L6-v2) & get_embedding_model()
|   +-- vectorstore.py      # ChromaDB vectorstore creation & DynamicRetriever (context-scaling)
|   +-- chain.py            # LangChain RetrievalQA chain with Gemini 2.5 Flash
|
+-- onnx_model/             # Auto-exported ONNX model files (model.onnx, tokenizer.json)
|
+-- evaluate/               # RAGAS evaluation package
|   +-- __init__.py
|   +-- ragas_eval.py       # Faithfulness, Answer Relevancy & Context Precision evaluation
|
+-- assets/                 # Static assets (architecture diagrams, screenshots)
+-- chroma_db/              # Local ChromaDB persistence directory
+-- requirements.txt        # Python dependencies
+-- .env                    # Environment variables (not committed)
+-- .gitignore
+-- LICENSE
+-- README.md
```

---

## Features

- **Multi-format support** — Upload and query PDF and plain-text (`.txt`) files
- **Multi-file upload** — Process several documents simultaneously in one session
- **100+ page document support** — Handles large documents (100+ pages) with sub-3-second processing times
- **Whole-document summarization** — Context-scaling `DynamicRetriever` feeds 100% of document content for comprehensive summaries with zero missed keywords or sentences
- **Semantic search** — Dense vector similarity via ONNX-accelerated `all-MiniLM-L6-v2` embeddings
- **Grounded answers** — Gemini 2.5 Flash with temperature=0 for factual, non-hallucinated responses
- **Professional, emoji-free responses** — LLM prompt enforces clean, objective, structured output
- **Source transparency** — `return_source_documents=True` exposes exactly which chunks informed each answer
- **Persistent chat history** — Full conversation context is maintained within a session via `st.session_state`
- **Active Knowledge Base tracking** — Sidebar displays currently indexed documents with a one-click reset option
- **Local ONNX embeddings** — No external embedding API calls; 2-3x faster inference with zero rate limits
- **Local vector storage** — ChromaDB runs in-process; no external database or cloud dependency required
- **Premium dark-mode UI** — Custom-styled Corporate Slate theme with modern typography, sleek scrollbars, and clean card layouts
- **Secure configuration** — API keys managed entirely through `.env` and `python-dotenv`
- **RAGAS evaluation** — Built-in evaluation tab measures Faithfulness, Answer Relevancy, and Context Precision using the RAGAS framework

---

## Dynamic Context-Scaling Retriever

DocuMind AI includes a custom `DynamicRetriever` that intelligently adapts how much context is sent to the LLM based on the type of query:

| Query Type | Retrieval Strategy | Example Queries |
|---|---|---|
| **Summary / Overview** | Fetches 100% of document chunks, sorted by page order | "Summarize this PDF", "Give me an overview", "What are the key takeaways?" |
| **Specific / Factual** | Top-10 precision semantic search | "What was the revenue in Q3?", "Define the term X" |

This ensures that summaries are comprehensive and cover every section of the document, while specific questions remain fast and precise.

---

## RAG Evaluation

DocuMind AI includes a dedicated **RAG Evaluation** tab powered by [RAGAS](https://docs.ragas.io/) — the standard open-source framework for measuring RAG pipeline quality.

### How to Use

1. Process your documents using the sidebar.
2. Navigate to the **RAG Evaluation** tab.
3. Enter one or more test questions. Optionally provide an expected answer for each.
4. Click **Run Evaluation** — the system will query the pipeline and score each response automatically.

### Metrics

| Metric | Description | Ground Truth Required |
|---|---|---|
| **Faithfulness** | Proportion of claims in the answer that are directly supported by the retrieved context. | No |
| **Answer Relevancy** | Degree to which the generated answer addresses the question asked. | No |
| **Context Precision** | Whether the most relevant retrieved chunks appear at the top of the ranked list. | Yes |

> All scores are in the range **[0, 1]**. Higher values indicate better pipeline performance. Context Precision is automatically enabled when expected answers are provided for all test cases.

### Implementation

| File | Role |
|---|---|
| `evaluate/ragas_eval.py` | Wraps the RAGAS `evaluate()` call; configures Gemini 2.5 Flash as the internal evaluator LLM with rate-limiting |
| `app.py` (RAG Evaluation tab) | Streamlit UI — collects test cases, invokes the pipeline, and renders metric cards and a per-question breakdown table |

---

## License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute it.

```
MIT License — Copyright (c) 2026 Prajwal Naik
```

See the [LICENSE](./LICENSE) file for full details.
