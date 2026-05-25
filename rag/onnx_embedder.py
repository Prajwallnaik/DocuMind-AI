"""
ONNX Runtime embedding model for LangChain.

On first use, automatically exports all-MiniLM-L6-v2 to ONNX format
using the already-cached model. After that, loads directly from ONNX
for 2-3x faster inference with no PyTorch overhead.
"""

import os
from typing import List

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from langchain_core.embeddings import Embeddings


def _export_model_to_onnx(model_dir: str):
    """One-time export: convert cached all-MiniLM-L6-v2 to ONNX format."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    print(f"[ONNXEmbeddings] First-time setup: exporting {MODEL_NAME} to ONNX ...")
    os.makedirs(model_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    # Create dummy input for tracing
    dummy = tokenizer("sample text", return_tensors="pt", padding=True, truncation=True)

    onnx_path = os.path.join(model_dir, "model.onnx")

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
            onnx_path,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids":         {0: "batch", 1: "seq_len"},
                "attention_mask":    {0: "batch", 1: "seq_len"},
                "token_type_ids":    {0: "batch", 1: "seq_len"},
                "last_hidden_state": {0: "batch", 1: "seq_len"},
            },
            opset_version=14,
        )

    # Save tokenizer files alongside the model
    tokenizer.save_pretrained(model_dir)
    print(f"[ONNXEmbeddings] Export complete → {model_dir}")


class ONNXEmbeddings(Embeddings):
    """LangChain-compatible embeddings using ONNX Runtime."""

    def __init__(self, model_dir: str):
        """
        Args:
            model_dir: Path to the directory containing model.onnx and tokenizer.json.
                       If the ONNX model doesn't exist yet, it will be auto-exported.
        """
        model_path = os.path.join(model_dir, "model.onnx")
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")

        # Auto-export on first use if ONNX model doesn't exist
        if not os.path.isfile(model_path):
            _export_model_to_onnx(model_dir)

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"ONNX model not found at {model_path} even after export attempt."
            )
        if not os.path.isfile(tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer not found at {tokenizer_path}. "
                "Run `python export_onnx.py` first to generate it."
            )

        # Load ONNX model — prefer CPU to avoid GPU memory issues for embeddings
        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = os.cpu_count()
        sess_opts.intra_op_num_threads = os.cpu_count()
        self._session = ort.InferenceSession(
            model_path, sess_opts, providers=["CPUExecutionProvider"]
        )

        # Load fast Rust tokenizer (from HuggingFace tokenizers library)
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_padding()
        self._tokenizer.enable_truncation(max_length=256)

    # ── LangChain interface ────────────────────────────────────────────────

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return self._encode(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        return self._encode([text])[0]

    # ── Internal ───────────────────────────────────────────────────────────

    def _encode(self, texts: List[str]) -> List[List[float]]:
        """Tokenize, run ONNX inference, mean-pool, and L2-normalize."""
        encodings = self._tokenizer.encode_batch(texts)

        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)

        outputs = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )

        # outputs[0] shape: (batch, seq_len, hidden_dim)
        last_hidden = outputs[0]

        # Mean pooling (mask out padding tokens)
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        sum_embeddings = np.sum(last_hidden * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        embeddings = sum_embeddings / sum_mask

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        embeddings = embeddings / norms

        return embeddings.tolist()


def get_embedding_model():
    """Returns the ONNX Runtime embedding model (all-MiniLM-L6-v2)."""
    model_dir = os.path.join(os.path.dirname(__file__), "..", "onnx_model")
    return ONNXEmbeddings(model_dir=model_dir)
