"""
Download pre-exported ONNX model for all-MiniLM-L6-v2 from HuggingFace.

Run this once to set up the ONNX model files:
    python export_onnx.py

Only requires: pip install huggingface_hub tokenizers
No PyTorch needed!
"""

import os
import json

MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onnx_model")

# Files we need from the HuggingFace repo
FILES_TO_DOWNLOAD = [
    "onnx/model.onnx",       # Pre-exported ONNX model
    "tokenizer.json",         # Fast tokenizer
    "tokenizer_config.json",  # Tokenizer config
    "vocab.txt",              # Vocabulary
]


def download():
    from huggingface_hub import hf_hub_download

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Downloading ONNX model from {MODEL_REPO} ...")
    for remote_path in FILES_TO_DOWNLOAD:
        local_name = os.path.basename(remote_path)
        print(f"  Downloading {remote_path} -> {local_name} ...")

        downloaded = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=remote_path,
            local_dir=None,  # use cache
        )

        # Copy to our output dir with flat naming
        dest = os.path.join(OUTPUT_DIR, local_name)
        import shutil
        shutil.copy2(downloaded, dest)

    print(f"\nDone! Files saved to: {OUTPUT_DIR}/")
    print("Files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"  {f:40s} {size / 1024:.1f} KB")


if __name__ == "__main__":
    download()
