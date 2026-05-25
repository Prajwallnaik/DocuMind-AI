import uuid
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

class DynamicRetriever(BaseRetriever):
    vectorstore: any
    k_default: int = 10
    k_summary: int = 200

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        # Detect if the query is a high-level summary or overview request
        q_lower = query.lower()
        summary_keywords = [
            "summar", "overview", "takeaway", "main point", "tldr", "tl;dr", 
            "abstract", "about the pdf", "about the doc", "executive summary", 
            "key points", "outline", "synopsis"
        ]
        is_summary = any(kw in q_lower for kw in summary_keywords)
        
        if is_summary:
            try:
                # Retrieve ALL document chunks to ensure zero missing key words or sentences
                all_data = self.vectorstore.get()
                documents = []
                if all_data and "documents" in all_data and all_data["documents"]:
                    for doc_text, metadata in zip(all_data["documents"], all_data["metadatas"]):
                        documents.append(Document(page_content=doc_text, metadata=metadata))
                if documents:
                    # Sort chunks chronologically by original page index so the context is fed in correct reading order
                    documents.sort(key=lambda d: d.metadata.get("page", 0))
                    return documents
            except Exception:
                pass
            
            # Fallback to broad semantic search with high k if database get() fails
            return self.vectorstore.similarity_search(query, k=self.k_summary)
        else:
            # High-precision targeted semantic search for specific fact-based queries
            return self.vectorstore.similarity_search(query, k=self.k_default)

def create_vectorstore(chunks, embedding_model):
    """Creates an in-memory Chroma vector database."""
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=f"rag_{uuid.uuid4().hex}"
    )
    return vectorstore

def get_retriever(vectorstore, k=10):
    """Returns a dynamic retriever that automatically scales context for summaries."""
    return DynamicRetriever(vectorstore=vectorstore, k_default=k)
