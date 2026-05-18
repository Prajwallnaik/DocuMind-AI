# Professional Project Prompt: RAG Application

---

## Project Overview

Build a **Retrieval-Augmented Generation (RAG) application** that allows users to upload a PDF or plain-text document and ask natural language questions about its contents. The system must retrieve only the most semantically relevant sections of the document and pass them as context to a large language model (LLM) to generate accurate, grounded answers — avoiding hallucination and staying within the scope of the uploaded document.

---

## Functional Requirements

- Accept file uploads in `.pdf` and `.txt` formats via a web UI
- Parse, chunk, and embed the document upon upload
- Store embeddings in a local vector database (ChromaDB)
- Accept a natural language question from the user
- Retrieve the top-k most relevant chunks using vector similarity search
- Pass retrieved chunks + question to an LLM to generate a final answer
- Display the answer and optionally expose the source chunks used

---

## Technical Specifications

| Component        | Technology                                      |
|------------------|-------------------------------------------------|
| UI Framework     | Streamlit                                       |
| LLM              | OpenAI GPT-3.5-turbo / GPT-4 (via LangChain)   |
| Embeddings       | OpenAI `text-embedding-ada-002` or `all-MiniLM-L6-v2` (HuggingFace, free) |
| Vector Database  | ChromaDB (local, in-process)                    |
| Orchestration    | LangChain `RetrievalQA` chain                   |
| File Parsing     | LangChain `PyPDFLoader`, `TextLoader`           |
| Text Splitting   | `RecursiveCharacterTextSplitter` (500 chars, 50 overlap) |
| Environment      | Python 3.9+, `.env` for secrets                 |

---

## Project File Structure

```
rag-app/
│
├── app.py                  # Streamlit entry point
├── rag/
│   ├── __init__.py
│   ├── loader.py           # Document loading (PDF / TXT)
│   ├── chunker.py          # Text splitting into passages
│   ├── embedder.py         # Embedding model setup
│   ├── vectorstore.py      # ChromaDB build & retrieval
│   └── chain.py            # LangChain RetrievalQA chain
│
├── requirements.txt
└── .env                    # API keys (never commit)
```

---

## RAG Pipeline (chunk → embed → retrieve → answer)

```
[User uploads file]
       ↓
[Parse document]  →  [Split into chunks (500 chars)]
                              ↓
                    [Embed each chunk → vectors]
                              ↓
                    [Store in ChromaDB]

[User asks question]
       ↓
[Embed question]  →  [Vector similarity search]
                              ↓
                    [Retrieve top-3 relevant chunks]
                              ↓
                    [LLM: chunks + question → answer]
                              ↓
                    [Display answer + source chunks]
```

---



## Expected Output

- A clean web UI at `localhost:8501`
- File upload widget for PDF or TXT
- Text input for natural language questions
- AI-generated answer grounded in the document
- Expandable panel showing the exact source chunks used

---

## Optional Enhancements

| Feature                    | How to implement                                      |
|----------------------------|-------------------------------------------------------|
| Multi-turn chat memory     | Add `ConversationBufferMemory` to the LangChain chain |
| Model selector             | Streamlit `selectbox` to toggle GPT-3.5 vs GPT-4     |
| Faster local vector search | Swap ChromaDB for FAISS                               |
| Cloud deployment           | Deploy free on Streamlit Community Cloud              |
| Multiple file uploads      | Loop `load_document()` over a list of files           |

---

## Key Concepts You Will Learn

- What embeddings are and how they represent text semantically
- How vector similarity search finds contextually relevant passages
- The full RAG pipeline: chunk → embed → retrieve → answer
- How LangChain orchestrates LLMs, retrievers, and chains
- How to build and deploy a production-ready AI app with Streamlit

---

*Prompt authored for professional AI-assisted development. Adapt tech stack as needed for your environment.*
