"""
RAGAS evaluation module for DocuMind AI.

Supported metrics
-----------------
Faithfulness       : Are all claims in the generated answer supported by the
                     retrieved context? (does not require ground truth)
Answer Relevancy   : Does the generated answer address the user's question?
                     (does not require ground truth)
Context Precision  : Are the most relevant chunks ranked highest in retrieval?
                     (requires ground truth)
"""

import os
import pandas as pd
from datasets import Dataset


def run_ragas_evaluation(qa_pairs: list) -> tuple:
    """
    Evaluate a batch of RAG responses using RAGAS.

    Parameters
    ----------
    qa_pairs : list[dict]
        Each dict must contain:
            question     (str)        The user question.
            answer       (str)        The LLM-generated answer.
            contexts     (list[str])  Retrieved document chunks passed to the LLM.
            ground_truth (str)        Expected correct answer.
                                      Required for Context Precision; optional
                                      for Faithfulness and Answer Relevancy.

    Returns
    -------
    tuple[pd.DataFrame, bool]
        - DataFrame with per-question scores for every computed metric.
        - Boolean flag indicating whether Context Precision was evaluated
          (True only when every qa_pair supplies a non-empty ground_truth).

    Raises
    ------
    ImportError
        If `ragas` or `datasets` are not installed.
    EnvironmentError
        If GOOGLE_API_KEY is absent from the environment.
    """
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from ragas.llms import LangchainLLMWrapper
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Add it to your .env file and restart the app."
        )

    # Wrap Gemini so RAGAS uses the same LLM as the rest of the pipeline.
    llm_wrapper = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=api_key,
        )
    )

    # Determine whether ground truth is available for all pairs.
    has_ground_truth = all(p.get("ground_truth", "").strip() for p in qa_pairs)

    data = {
        "question":     [p["question"]               for p in qa_pairs],
        "answer":       [p["answer"]                 for p in qa_pairs],
        "contexts":     [p["contexts"]               for p in qa_pairs],
        "ground_truth": [p.get("ground_truth", "")   for p in qa_pairs],
    }
    dataset = Dataset.from_dict(data)

    metrics = [faithfulness, answer_relevancy]
    if has_ground_truth:
        metrics.append(context_precision)

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm_wrapper,
        raise_exceptions=False,
    )

    return result.to_pandas(), has_ground_truth
