import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def load_document(file_path: str):
    """Loads a PDF or TXT document and returns a list of Document objects."""
    _, ext = os.path.splitext(file_path)
    if ext.lower() == '.pdf':
        loader = PyPDFLoader(file_path)
    elif ext.lower() == '.txt':
        loader = TextLoader(file_path, encoding='utf-8')
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    
    return loader.load()
