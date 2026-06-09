import os
from tokenizers import Tokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize local tokenizer for token-based counting
_tokenizer = None
def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        # Resolve path to tokenizer.json in onnx_model directory
        tokenizer_path = os.path.join(os.path.dirname(__file__), "..", "onnx_model", "tokenizer.json")
        if os.path.isfile(tokenizer_path):
            _tokenizer = Tokenizer.from_file(tokenizer_path)
    return _tokenizer

def token_length_function(text: str) -> int:
    tok = get_tokenizer()
    if tok is not None:
        return len(tok.encode(text).ids)
    # Simple character length fallback if tokenizer is not available
    return len(text) // 4

def split_documents(documents, strategy="recursive", chunk_size=1000, chunk_overlap=200, parent_size=1000, parent_overlap=200):
    """Splits documents into smaller chunks using character, token, or parent-child strategy."""
    if strategy == "token":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=token_length_function
        )
        return text_splitter.split_documents(documents)
        
    elif strategy == "parent_child":
        # First split into larger parent chunks (always token-based)
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=parent_overlap,
            length_function=token_length_function
        )
        parents = parent_splitter.split_documents(documents)
        
        # Then split each parent into smaller child chunks
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=token_length_function
        )
        
        child_docs = []
        for i, parent_doc in enumerate(parents):
            # Split this specific parent
            children = child_splitter.split_documents([parent_doc])
            for child in children:
                # Add parent info to child's metadata
                child.metadata["parent_content"] = parent_doc.page_content
                child.metadata["parent_id"] = i
                child_docs.append(child)
        return child_docs
        
    else:  # "recursive" (legacy character-based)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        return text_splitter.split_documents(documents)

