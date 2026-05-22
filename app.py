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

# ── Session state initialization ──────────────────────────────────────────────
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

# ── Sidebar — document ingestion ───────────────────────────────────────────────
with st.sidebar:
    st.header("Data Sources")
    uploaded_files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    

    if st.button("Process Documents") and uploaded_files:
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
                    
                    status.update(label=f"{len(uploaded_files)} document(s) processed successfully!", state="complete", expanded=False)
                else:
                    status.update(label="No valid content found in the uploaded documents.", state="error", expanded=True)
            except Exception as e:
                status.update(label=f"An error occurred: {e}", state="error", expanded=True)

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
