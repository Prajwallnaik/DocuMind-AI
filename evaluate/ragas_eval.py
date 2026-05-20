"""
RAGAS evaluation module for DocuMind AI.

Uses RAGAS legacy metrics with a custom rate-limiting LLM wrapper to work
flawlessly within Google Gemini's free tier quota (15 RPM), along with local
HuggingFace embeddings to bypass embedding rate limits and API compatibility bugs.

Supported metrics:
- Faithfulness
- Answer Relevancy
- Context Precision
"""

import os
import time
import asyncio
import pandas as pd
from datasets import Dataset

from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import _LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.run_config import RunConfig


class RateLimitedLangchainLLMWrapper(_LangchainLLMWrapper):
    """
    Subclass of Ragas LangchainLLMWrapper that introduces a 4-second delay
    before all synchronous and asynchronous text generation calls.
    Ensures total compliance with Google Gemini's 15 requests per minute limit.
    """
    def generate_text(self, *args, **kwargs):
        time.sleep(4.0)
        return super().generate_text(*args, **kwargs)

    async def agenerate_text(self, *args, **kwargs):
        await asyncio.sleep(4.0)
        return await super().agenerate_text(*args, **kwargs)


def run_ragas_evaluation(qa_pairs: list) -> tuple:
    """
    Evaluate a batch of RAG responses using RAGAS.

    Parameters
    ----------
    qa_pairs : list[dict]
        Each dict must contain:
            question     (str)        The user question.
            answer       (str)        The LLM-generated answer.
            contexts     (list[str])  Retrieved document chunks.
            ground_truth (str)        Expected correct answer.

    Returns
    -------
    tuple[pd.DataFrame, bool]
        - DataFrame with per-question scores for every computed metric.
        - Boolean flag indicating whether Context Precision was evaluated.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Add it to your .env file and restart the app."
        )

    # Initialize Gemini Langchain LLM wrapped in our rate-limiting layer
    langchain_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=api_key
    )
    llm = RateLimitedLangchainLLMWrapper(langchain_llm)

    # Initialize local HuggingFace embeddings wrapped for RAGAS compatibility
    langchain_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)

    # Determine whether ground truth is available for all pairs
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

    # Run evaluation
    from ragas import evaluate
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(max_workers=1, timeout=300),
        raise_exceptions=False,
    )

    return result.to_pandas(), has_ground_truth
