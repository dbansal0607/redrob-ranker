#!/usr/bin/env python3
"""
download_model.py — Run ONCE before ranking.
Downloads all-MiniLM-L6-v2 to ./model/ so ranking works offline.

Usage:
    pip install sentence-transformers
    python download_model.py
"""

import os
import sys
from pathlib import Path

MODEL_DIR = Path("./model")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    print(f"Downloading {MODEL_NAME} to {MODEL_DIR} ...")
    print("This runs ONCE — ranking works fully offline after this.\n")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed.")
        print("Run: pip install sentence-transformers")
        sys.exit(1)

    MODEL_DIR.mkdir(exist_ok=True)

    model = SentenceTransformer(MODEL_NAME)
    model.save(str(MODEL_DIR))

    # Quick sanity check
    test_embedding = model.encode(["test sentence"], convert_to_numpy=True)
    print(f"\n✅ Model downloaded successfully to {MODEL_DIR}/")
    print(f"   Embedding shape: {test_embedding.shape}")
    print(f"   Model size: {sum(f.stat().st_size for f in MODEL_DIR.rglob('*') if f.is_file()) / 1e6:.1f} MB")
    print(f"\nYou can now run rank.py — no internet needed during ranking.")

if __name__ == "__main__":
    main()