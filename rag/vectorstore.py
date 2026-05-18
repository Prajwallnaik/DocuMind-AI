import uuid
from langchain_chroma import Chroma

def create_vectorstore(chunks, embedding_model):
    """Creates an in-memory Chroma vector database."""
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=f"rag_{uuid.uuid4().hex}"
    )
    return vectorstore

def get_retriever(vectorstore, k=10):
    """Returns a retriever from the vector database."""
    return vectorstore.as_retriever(search_kwargs={"k": k})
