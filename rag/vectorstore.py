import uuid
import os
from typing import List, Any
from langchain_community.vectorstores import Chroma
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

def _get_llm():
    """Dynamically initializes the Gemini LLM for query expansion."""
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="google/gemini-2.5-flash",
            openai_api_key=openrouter_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.7,
            max_tokens=512,
        )
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
            google_api_key=os.environ.get("GOOGLE_API_KEY", "missing_key")
        )

class DynamicRetriever(BaseRetriever):
    vectorstore: Any
    k_default: int = 10
    k_summary: int = 200
    retrieval_type: str = "hybrid"
    alpha: float = 0.5
    bm25_retriever: Any = None
    query_expansion_mode: str = "none"
    enable_reranking: bool = False

    _ranker: Any = None

    class Config:
        arbitrary_types_allowed = True

    def _get_ranker(self):
        if DynamicRetriever._ranker is None:
            try:
                from flashrank import Ranker
                DynamicRetriever._ranker = Ranker()
            except Exception as e:
                print(f"[DynamicRetriever] Error initializing FlashRank: {e}")
        return DynamicRetriever._ranker

    def _weighted_rrf(self, vector_docs: List[Document], bm25_docs: List[Document], alpha: float, k: int = 60) -> List[Document]:
        rrf_scores = {}
        for rank, doc in enumerate(vector_docs):
            key = doc.page_content
            if key not in rrf_scores:
                rrf_scores[key] = (0.0, doc)
            score, _ = rrf_scores[key]
            rrf_scores[key] = (score + alpha / (rank + 1 + k), doc)
            
        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content
            if key not in rrf_scores:
                rrf_scores[key] = (0.0, doc)
            score, _ = rrf_scores[key]
            rrf_scores[key] = (score + (1.0 - alpha) / (rank + 1 + k), doc)
            
        sorted_docs = sorted(rrf_scores.values(), key=lambda x: x[0], reverse=True)
        return [doc for score, doc in sorted_docs]

    def _rrf_multiple_runs(self, doc_lists: List[List[Document]], k: int = 60) -> List[Document]:
        rrf_scores = {}
        for doc_list in doc_lists:
            for rank, doc in enumerate(doc_list):
                key = doc.page_content
                if key not in rrf_scores:
                    rrf_scores[key] = (0.0, doc)
                score, _ = rrf_scores[key]
                rrf_scores[key] = (score + 1.0 / (rank + 1 + k), doc)
                
        sorted_docs = sorted(rrf_scores.values(), key=lambda x: x[0], reverse=True)
        return [doc for score, doc in sorted_docs]

    def _reconstruct_parents(self, docs: List[Document]) -> List[Document]:
        reconstructed = []
        seen_contents = set()
        for doc in docs:
            parent_content = doc.metadata.get("parent_content")
            if parent_content:
                if parent_content not in seen_contents:
                    seen_contents.add(parent_content)
                    meta = doc.metadata.copy()
                    meta.pop("parent_content", None)
                    meta.pop("parent_id", None)
                    reconstructed.append(Document(page_content=parent_content, metadata=meta))
            else:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    reconstructed.append(doc)
        return reconstructed

    def _expand_query(self, query: str) -> List[str]:
        if self.query_expansion_mode == "none":
            return [query]
            
        try:
            llm = _get_llm()
            if self.query_expansion_mode == "multiquery":
                prompt = f"""You are an AI study assistant. The user is asking a question: "{query}".
To get the best documents from a search database, generate 3 alternative search queries that use different words, synonyms, or angles.
Return ONLY the 3 alternative queries, one per line. Do not number them or add any other text.

Alternative Queries:"""
                response = llm.invoke(prompt)
                content = response.content if hasattr(response, 'content') else str(response)
                queries = [line.strip() for line in content.strip().split("\n") if line.strip()]
                combined = [query] + [q for q in queries if q != query]
                return combined[:4]
                
            elif self.query_expansion_mode == "hyde":
                prompt = f"""You are an AI study assistant. The user is asking a question: "{query}".
Write a brief, hypothetical one-paragraph answer to this question. Focus on incorporating specific technical terms, facts, and explanations that would appear in a textbook context.
Do not verify if the answer is correct; write it as if it were a direct excerpt from a reference document.

Hypothetical Answer:"""
                response = llm.invoke(prompt)
                answer = response.content if hasattr(response, 'content') else str(response)
                return [query, answer.strip()]
        except Exception as e:
            print(f"[DynamicRetriever] Query expansion failed: {e}")
            
        return [query]

    def _retrieve_for_single_query(self, q: str, limit: int) -> List[Document]:
        if self.retrieval_type == "semantic":
            return self.vectorstore.similarity_search(q, k=limit)
        elif self.retrieval_type == "keyword":
            if self.bm25_retriever is not None:
                self.bm25_retriever.k = limit
                return self.bm25_retriever.get_relevant_documents(q)
            else:
                return self.vectorstore.similarity_search(q, k=limit)
        else:
            vector_docs = self.vectorstore.similarity_search(q, k=limit)
            if self.bm25_retriever is not None:
                self.bm25_retriever.k = limit
                bm25_docs = self.bm25_retriever.get_relevant_documents(q)
                return self._weighted_rrf(vector_docs, bm25_docs, self.alpha)
            else:
                return vector_docs

    def _rerank_docs(self, query: str, docs: List[Document]) -> List[Document]:
        if not docs or not self.enable_reranking:
            return docs
            
        ranker = self._get_ranker()
        if ranker is None:
            return docs
            
        try:
            from flashrank import RerankRequest
            passages = []
            for i, doc in enumerate(docs):
                passages.append({
                    "id": i,
                    "text": doc.page_content,
                    "meta": doc.metadata
                })
                
            rerank_request = RerankRequest(query=query, passages=passages)
            results = ranker.rerank(rerank_request)
            
            reranked_docs = []
            for res in results:
                reranked_docs.append(Document(page_content=res["text"], metadata=res["meta"]))
            return reranked_docs
        except Exception as e:
            print(f"[DynamicRetriever] Reranking failed: {e}")
            return docs

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        q_lower = query.lower()
        summary_keywords = [
            "summar", "overview", "takeaway", "main point", "tldr", "tl;dr", 
            "abstract", "about the pdf", "about the doc", "executive summary", 
            "key points", "outline", "synopsis"
        ]
        is_summary = any(kw in q_lower for kw in summary_keywords)
        
        if is_summary:
            try:
                all_data = self.vectorstore.get()
                documents = []
                if all_data and "documents" in all_data and all_data["documents"]:
                    for doc_text, metadata in zip(all_data["documents"], all_data["metadatas"]):
                        documents.append(Document(page_content=doc_text, metadata=metadata))
                if documents:
                    documents = self._reconstruct_parents(documents)
                    documents.sort(key=lambda d: d.metadata.get("page", 0))
                    return documents
            except Exception:
                pass
            
            fallback_docs = self.vectorstore.similarity_search(query, k=self.k_summary)
            return self._reconstruct_parents(fallback_docs)
        
        limit = max(20, self.k_default * 2) if self.enable_reranking else self.k_default
        expanded_queries = self._expand_query(query)
        
        retrieved_runs = []
        for eq in expanded_queries:
            run_docs = self._retrieve_for_single_query(eq, limit)
            retrieved_runs.append(run_docs)
            
        if len(retrieved_runs) > 1:
            merged_docs = self._rrf_multiple_runs(retrieved_runs)
        else:
            merged_docs = retrieved_runs[0]
            
        reconstructed_docs = self._reconstruct_parents(merged_docs)
        final_docs = self._rerank_docs(query, reconstructed_docs)
        
        return final_docs[:self.k_default]

def create_vectorstore(chunks, embedding_model):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=f"rag_{uuid.uuid4().hex}"
    )
    return vectorstore

def get_retriever(vectorstore, chunks=None, retrieval_type="hybrid", k=10, alpha=0.5, query_expansion_mode="none", enable_reranking=False):
    bm25_retriever = None
    if chunks and (retrieval_type in ["keyword", "hybrid"]):
        try:
            from langchain_community.retrievers import BM25Retriever
            bm25_retriever = BM25Retriever.from_documents(chunks)
        except Exception as e:
            print(f"[DynamicRetriever] Error initializing BM25Retriever: {e}")
            retrieval_type = "semantic"
            
    return DynamicRetriever(
        vectorstore=vectorstore, 
        k_default=k, 
        retrieval_type=retrieval_type, 
        alpha=alpha, 
        bm25_retriever=bm25_retriever,
        query_expansion_mode=query_expansion_mode,
        enable_reranking=enable_reranking
    )

