"""Precompute corpus embeddings once, offline: python3 scripts/build_embeddings.py [model]
Writes data/embeddings/<model>.npy + .json (model, corpus sha256). Commit both; the server then only encodes queries."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data import load_corpus
from core.models import DENSE_MODEL, emb_paths, corpus_sha
import numpy as np
from sentence_transformers import SentenceTransformer

model = sys.argv[1] if len(sys.argv) > 1 else DENSE_MODEL
C = load_corpus()
E = SentenceTransformer(model).encode(C.english.tolist(), normalize_embeddings=True, batch_size=32, show_progress_bar=True)
npy, meta = emb_paths(model); os.makedirs(os.path.dirname(npy), exist_ok=True)
np.save(npy, E.astype(np.float32))
json.dump(dict(model=model, corpus_sha256=corpus_sha(), n=len(C), dim=int(E.shape[1]), normalized=True), open(meta, 'w'), indent=1)
print('wrote', npy, E.shape)
