import os
import tempfile
import sys

# Override standard sqlite3 with pysqlite3 to support ChromaDB on older Windows Python
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except Exception:
    pass

import streamlit as st
from dotenv import load_dotenv

from rag.loader import load_document
from rag.chunker import split_documents
from rag.onnx_embedder import get_embedding_model
from rag.vectorstore import create_vectorstore, get_retriever
from rag.chain import create_qa_chain

# Load environment variables
load_dotenv(override=True)
if "OPENAI_API_KEY" in os.environ and "your_openai" in os.environ["OPENAI_API_KEY"]:
    del os.environ["OPENAI_API_KEY"]

st.set_page_config(
    page_title="Enterprise Document Intelligence", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
    /* Core Typography & Base Styling */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    h1, h2, h3, h4, h5, h6, [data-testid="stMetricValue"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background-color: transparent !important;}

    /* Premium Theme & Backgrounds */
    .stApp {
        background-color: #090D16 !important;
        background-image: radial-gradient(circle at 10% 20%, rgba(37, 99, 235, 0.04) 0%, transparent 45%),
                          radial-gradient(circle at 90% 80%, rgba(59, 130, 246, 0.04) 0%, transparent 45%) !important;
    }

    /* Sleek Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 6px !important;
        height: 6px !important;
    }
    ::-webkit-scrollbar-track {
        background: #090D16 !important;
    }
    ::-webkit-scrollbar-thumb {
        background: #1E293B !important;
        border-radius: 3px !important;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #2E3E56 !important;
    }

    /* Sidebar Refinement */
    section[data-testid="stSidebar"] {
        background-color: #0D111A !important;
        border-right: 1px solid #1E293B !important;
        padding-top: 1.5rem;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
        font-weight: 700;
        font-size: 1.3rem;
        background: linear-gradient(135deg, #ffffff 0%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }

    /* Clean File Uploader Integration */
    section[data-testid="stFileUploader"] {
        background-color: #0D111A !important;
        border: 1px dashed #1E293B !important;
        border-radius: 8px !important;
        padding: 1.25rem !important;
        transition: border-color 0.2s ease !important;
    }
    section[data-testid="stFileUploader"]:hover {
        border-color: #3B82F6 !important;
    }

    /* Streamlit Tab Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px !important;
        background-color: #0D111A !important;
        padding: 4px !important;
        border-radius: 8px !important;
        border: 1px solid #1E293B !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 36px !important;
        border-radius: 6px !important;
        padding: 0 16px !important;
        background-color: transparent !important;
        color: #64748B !important;
        font-weight: 500 !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #F1F5F9 !important;
        background-color: rgba(255,255,255,0.02) !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E293B !important;
        color: #60A5FA !important;
        font-weight: 600 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #60A5FA !important;
    }

    /* Clean Metric Card Styling */
    div[data-testid="metric-container"] {
        background-color: #0D111A !important;
        border: 1px solid #1E293B !important;
        border-radius: 8px !important;
        padding: 1.25rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="metric-container"]:hover {
        border-color: #3B82F6 !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.08) !important;
    }
    div[data-testid="metric-container"] label {
        font-size: 0.8rem !important;
        color: #94A3B8 !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Buttons Overrides */
    button[kind="primary"] {
        background: #2563EB !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.25rem !important;
        transition: all 0.2s ease !important;
        box-shadow: none !important;
    }
    button[kind="primary"]:hover {
        background: #1D4ED8 !important;
        transform: none !important;
        box-shadow: none !important;
    }
    
    button[kind="secondary"] {
        background-color: #0D111A !important;
        color: #E2E8F0 !important;
        border: 1px solid #1E293B !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    button[kind="secondary"]:hover {
        background-color: #1E293B !important;
        color: white !important;
        border-color: #2E3E56 !important;
    }

    /* Smooth status box refinement */
    div[data-testid="stStatus"] {
        border-radius: 8px !important;
        border: 1px solid #1E293B !important;
        background-color: #0D111A !important;
        box-shadow: none !important;
    }

    /* Styled Tables / Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid #1E293B !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    /* Info and Warning Alerts styling */
    div[data-testid="stNotification"] {
        border-radius: 8px !important;
        border: 1px solid #1E293B !important;
        background-color: #0D111A !important;
    }

    /* Custom Chat Message Container */
    div[data-testid="stChatMessage"] {
        background-color: #0D111A !important;
        border: 1px solid #1E293B !important;
        border-radius: 8px !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 1rem !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stChatMessage"]:hover {
        border-color: #2E3E56 !important;
    }
    
    /* Clean chat avatars */
    div[data-testid="stChatMessageAvatar"] {
        background-color: #1E293B !important;
        border: 1px solid #2E3E56 !important;
        border-radius: 6px !important;
        width: 32px !important;
        height: 32px !important;
    }

    /* Modern Custom Chat Input */
    div[data-testid="stChatInput"] {
        background-color: #0D111A !important;
        border: 1px solid #1E293B !important;
        border-radius: 8px !important;
        padding: 0.25rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }
    div[data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: #E2E8F0 !important;
        border: none !important;
        font-size: 0.95rem !important;
    }
    div[data-testid="stChatInput"] textarea:focus {
        box-shadow: none !important;
    }
</style>

<div style="margin-bottom: 2.5rem; display: flex; align-items: center; gap: 15px;">
    <div style="
        background: #2563EB;
        padding: 12px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
    ">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/>
            <polyline points="14 2 14 8 20 8"/>
            <line x1="16" y1="13" x2="8" y2="13"/>
            <line x1="16" y1="17" x2="8" y2="17"/>
            <line x1="10" y1="9" x2="8" y2="9"/>
        </svg>
    </div>
    <div>
        <h1 style="
            font-weight: 800; 
            font-size: 2.5rem;
            margin: 0;
            padding: 0;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #ffffff 0%, #e2e8f0 70%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        ">DocuMind AI</h1>
        <p style="
            font-size: 0.95rem;
            color: #94A3B8;
            margin: 4px 0 0 0;
            font-weight: 400;
        ">Intelligent Document Intelligence & Retrieval Engine</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Session state initialization ──────────────────────────────────────────────
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# ── Sidebar — document ingestion ───────────────────────────────────────────────
with st.sidebar:
    st.header("Data Sources")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    st.write("")
    process_btn = st.button(
        "Process Documents", 
        type="primary", 
        use_container_width=True,
        disabled=not uploaded_files
    )

    if process_btn and uploaded_files:
        # Clear chat history and prior evaluation results when new documents are loaded
        st.session_state.messages = []
        st.session_state.eval_results = None
        
        with st.status("Processing documents...", expanded=True) as status:
            all_chunks = []
            try:
                for uploaded_file in uploaded_files:
                    # Save uploaded file to a temp file for processing
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
                    status.write("Setting up embedding model...")
                    embedding_model = get_embedding_model()
                    
                    status.write("Building vector database...")
                    vectorstore = create_vectorstore(all_chunks, embedding_model)
                    st.session_state.vectorstore = vectorstore
                    
                    status.write("Creating QA Chain...")
                    retriever = get_retriever(vectorstore)
                    qa_chain = create_qa_chain(retriever)
                    st.session_state.qa_chain = qa_chain
                    
                    st.session_state.processed_files = [f.name for f in uploaded_files]
                    
                    status.update(label=f"{len(uploaded_files)} document(s) processed successfully!", state="complete", expanded=False)
                    st.rerun()
                else:
                    status.update(label="No valid content found in the uploaded documents.", state="error", expanded=True)
            except Exception as e:
                status.update(label=f"An error occurred: {e}", state="error", expanded=True)

    if st.session_state.processed_files:
        st.markdown('<p style="font-size: 0.8rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 2rem; margin-bottom: 0.75rem; font-weight: 600;">Active Knowledge Base</p>', unsafe_allow_html=True)
        for fname in st.session_state.processed_files:
            st.markdown(f"""
            <div style="
                background-color: #0D111A;
                border: 1px solid #1E293B;
                border-radius: 6px;
                padding: 10px 12px;
                margin-bottom: 8px;
                font-size: 0.85rem;
                color: #E2E8F0;
                display: flex;
                align-items: center;
                gap: 8px;
            ">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#60A5FA" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink: 0;">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                    <polyline points="14 2 14 8 20 8"/>
                    <line x1="16" y1="13" x2="8" y2="13"/>
                    <line x1="16" y1="17" x2="8" y2="17"/>
                    <polyline points="10 9 9 9 8 9"/>
                </svg>
                <span style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-weight: 500;">{fname}</span>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        if st.button("Reset Knowledge Base", type="secondary", use_container_width=True):
            st.session_state.qa_chain = None
            st.session_state.vectorstore = None
            st.session_state.processed_files = []
            st.session_state.messages = []
            st.session_state.eval_results = None
            st.rerun()


# ── Main area — tabbed interface ───────────────────────────────────────────────
tab_query, tab_eval = st.tabs(["Document Query", "RAG Evaluation"])

# ── Tab 1: Document Query ──────────────────────────────────────────────────────
with tab_query:
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
            # Display user message immediately in the active frame
            with st.chat_message("user"):
                st.markdown(user_question)
                
            # Display assistant generation state and answer immediately
            with st.chat_message("assistant"):
                with st.spinner("Generating answer..."):
                    try:
                        query_with_lang = f"Please provide your answer in English. {user_question}"
                        response = st.session_state.qa_chain.invoke({"query": query_with_lang})
                        answer = response.get("result", "")
                        source_docs = response.get("source_documents", [])
                        
                        st.markdown(answer)

                        # Commit both user and assistant messages to persistent session state history
                        st.session_state.messages.append({"role": "user", "content": user_question})
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "source_docs": source_docs
                        })
                        
                        # Trigger a script rerun to force a clean historical draw with input widget strictly at the bottom
                        st.rerun()
                    except Exception as e:
                        st.error(f"An error occurred while generating the answer: {e}")
    else:
        st.info("Please upload and process documents to start asking questions.")

# ── Tab 2: RAG Evaluation (RAGAS) ─────────────────────────────────────────────
with tab_eval:
    st.markdown("### RAG Evaluation — RAGAS Metrics")
    st.caption(
        "Evaluate pipeline quality using three industry-standard RAG metrics. "
        "Process documents via the sidebar before running evaluation."
    )

    with st.expander("Metric Definitions", expanded=False):
        st.markdown("""
| Metric | What it measures | Ground Truth Required |
|---|---|---|
| **Faithfulness** | Every factual claim in the generated answer is supported by the retrieved context. Score of 1.0 = fully grounded; 0.0 = hallucinated. | No |
| **Answer Relevancy** | The generated answer directly addresses the question asked. Penalises incomplete or off-topic responses. | No |
| **Context Precision** | The most relevant retrieved chunks appear at the top of the ranked list. Computed against the expected answer. | Yes |
        """)
        st.caption("All scores are in the range [0, 1]. Higher is better.")

    st.divider()

    if not st.session_state.qa_chain:
        st.warning("No documents have been processed. Upload and process documents using the sidebar first.")
    else:
        n_cases = int(st.number_input(
            "Number of test cases", min_value=1, max_value=10, value=1, step=1
        ))

        st.markdown(
            "Enter a question and, optionally, the expected correct answer for each test case. "
            "Supplying an expected answer for **all** test cases enables the **Context Precision** metric."
        )
        st.write("")

        test_cases = []
        for i in range(n_cases):
            st.markdown(f"**Test Case {i + 1}**")
            col_q, col_gt = st.columns(2)
            
            with col_q:
                q = st.text_input(
                    "Question",
                    key=f"eval_q_{i}",
                    placeholder=f"Enter Question {i + 1}...",
                    label_visibility="collapsed",
                )
            with col_gt:
                gt = st.text_input(
                    "Expected Answer",
                    key=f"eval_gt_{i}",
                    placeholder="Enter expected answer (enables Context Precision)...",
                    label_visibility="collapsed",
                )
            test_cases.append({"question": q.strip(), "ground_truth": gt.strip()})

        st.write("")
        run_col, _ = st.columns([1, 4])
        with run_col:
            run_eval = st.button("Run Evaluation", type="primary", use_container_width=True)

        if run_eval:
            valid_cases = [tc for tc in test_cases if tc["question"]]
            if not valid_cases:
                st.warning("Enter at least one question before running evaluation.")
            else:
                with st.spinner(f"Querying pipeline and running RAGAS on {len(valid_cases)} test case(s)..."):
                    qa_pairs = []
                    pipeline_errors = []

                    for tc in valid_cases:
                        try:
                            query = f"Please provide your answer in English. {tc['question']}"
                            response = st.session_state.qa_chain.invoke({"query": query})
                            answer = response.get("result", "")
                            source_docs = response.get("source_documents", [])
                            contexts = [doc.page_content for doc in source_docs]

                            qa_pairs.append({
                                "question":     tc["question"],
                                "answer":       answer,
                                "contexts":     contexts if contexts else ["No context retrieved."],
                                "ground_truth": tc["ground_truth"],
                            })
                        except Exception as exc:
                            pipeline_errors.append(f"Error on '{tc['question']}': {exc}")

                    for err in pipeline_errors:
                        st.error(err)

                    if qa_pairs:
                        try:
                            from evaluate.ragas_eval import run_ragas_evaluation
                            df, has_gt = run_ragas_evaluation(qa_pairs)
                            st.session_state.eval_results = (df, has_gt)
                        except EnvironmentError as env_err:
                            st.error(str(env_err))
                        except ImportError:
                            st.error(
                                "RAGAS is not installed. Run: `pip install ragas datasets`"
                            )
                        except Exception as exc:
                            st.error(f"RAGAS evaluation failed: {exc}")

        # ── Results display ────────────────────────────────────────────────────
        if st.session_state.eval_results is not None:
            df, has_gt = st.session_state.eval_results

            st.divider()
            st.markdown("#### Evaluation Results")

            # Summary metric cards
            metric_definitions = [
                ("faithfulness",     "Faithfulness"),
                ("answer_relevancy", "Answer Relevancy"),
                ("context_precision","Context Precision"),
            ]
            
            # Clean up DataFrame columns: convert metric values to float for calculations
            # We preserve NaNs/Nones for accurate mathematical averages (excluding missing values)
            import pandas as pd
            df = df.copy()
            for col_key, _ in metric_definitions:
                if col_key in df.columns:
                    df[col_key] = pd.to_numeric(df[col_key], errors='coerce')

            cols = st.columns(3)
            for (col_key, label), col in zip(metric_definitions, cols):
                with col:
                    if col_key in df.columns:
                        avg = df[col_key].mean()
                        if pd.notnull(avg):
                            st.metric(label=label, value=f"{avg:.3f}")
                        else:
                            st.metric(
                                label=label,
                                value="N/A",
                                help="This metric was not computed for any test case.",
                            )
                    else:
                        st.metric(
                            label=label,
                            value="N/A",
                            help="Provide expected answers for all test cases to enable this metric.",
                        )

            st.write("")

            # Per-question breakdown table
            st.markdown("**Per-Question Breakdown**")
            display_cols = {"question": "Question", "answer": "Generated Answer"}
            for col_key, label in metric_definitions:
                if col_key in df.columns:
                    display_cols[col_key] = label

            # Create a display-friendly DataFrame with formatted scores and elegant "N/A" placeholders
            display_df = df.copy()
            for col_key, _ in metric_definitions:
                if col_key in display_df.columns:
                    display_df[col_key] = display_df[col_key].apply(
                        lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A"
                    )

            st.dataframe(
                display_df[list(display_cols.keys())].rename(columns=display_cols),
                use_container_width=True,
                hide_index=True,
            )

            if not has_gt:
                st.caption(
                    "Context Precision was not computed. "
                    "Populate the Expected Answer field for every test case to enable it."
                )
