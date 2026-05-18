import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from rag.loader import load_document
from rag.chunker import split_documents
from rag.embedder import get_embedding_model
from rag.vectorstore import create_vectorstore, get_retriever
from rag.chain import create_qa_chain

# Load environment variables
load_dotenv(override=True)
if "OPENAI_API_KEY" in os.environ and "your_openai" in os.environ["OPENAI_API_KEY"]:
    del os.environ["OPENAI_API_KEY"]

st.set_page_config(page_title="Enterprise Document Intelligence", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>

<div style="margin-bottom: 2rem;">
    <h1 style="
        font-weight: 800; 
        font-size: 3rem;
        margin-top: -5px;
        font-size: 3rem;
        margin-bottom: 0;
        padding-bottom: 0;
        color: white;
        text-shadow: -2px 0px 0px #ff6b6b, 2px 0px 0px #4dabf7;
    ">DocuMind AI</h1>
    <p style="
        font-size: 1.2rem;
        color: #a5d8ff;
        margin-top: 5px;
        font-weight: 400;
    ">Intelligent document analysis and retrieval system.</p>
</div>
""", unsafe_allow_html=True)

# Session state initialization
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Sidebar for file upload
with st.sidebar:
    st.header("Data Sources")
    uploaded_files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    

    if st.button("Process Documents") and uploaded_files:
        # Clear chat history when new documents are uploaded
        st.session_state.messages = []
        
        with st.status("Processing documents...", expanded=True) as status:
            all_chunks = []
            try:
                for uploaded_file in uploaded_files:
                    # Save uploaded file to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    try:
                        status.write(f"Loading document: {uploaded_file.name}...")
                        documents = load_document(tmp_file_path)
                        
                        status.write(f"Splitting {uploaded_file.name} into chunks...")
                        chunks = split_documents(documents)
                        all_chunks.extend(chunks)
                    finally:
                        try:
                            os.unlink(tmp_file_path)
                        except Exception:
                            pass
                
                if all_chunks:
                    # 3. Embed
                    status.write("Setting up embedding model...")
                    embedding_model = get_embedding_model()
                    
                    # 4. Vectorstore
                    status.write("Building vector database...")
                    vectorstore = create_vectorstore(all_chunks, embedding_model)
                    st.session_state.vectorstore = vectorstore
                    
                    # 5. Chain
                    status.write("Creating QA Chain...")
                    retriever = get_retriever(vectorstore)
                    qa_chain = create_qa_chain(retriever)
                    st.session_state.qa_chain = qa_chain
                    
                    status.update(label=f"{len(uploaded_files)} document(s) processed successfully!", state="complete", expanded=False)
                else:
                    status.update(label="No valid content found in the uploaded documents.", state="error", expanded=True)
            except Exception as e:
                status.update(label=f"An error occurred: {e}", state="error", expanded=True)

# Main area for chat
if st.session_state.qa_chain:
    st.markdown('<p style="font-size: 0.85rem; color: #6c757d; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 2rem; margin-bottom: 1rem;">Document Query Interface</p>', unsafe_allow_html=True)
    
    # Initialize chat history if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Render previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    
    # Chat input
    if user_question := st.chat_input("Enter your question:"):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
            
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                try:
                    query_with_lang = f"Please provide your answer in English. {user_question}"
                    response = st.session_state.qa_chain.invoke({"query": query_with_lang})
                    answer = response.get("result", "")
                    source_docs = response.get("source_documents", [])
                    
                    st.markdown(answer)

                            
                    # Append assistant message to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "source_docs": source_docs
                    })
                except Exception as e:
                    st.error(f"An error occurred while generating the answer: {e}")
else:
    st.info("Please upload and process documents to start asking questions.")
