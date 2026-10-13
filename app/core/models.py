"""Optional neural components. Everything degrades gracefully: without sentence-transformers the app runs BM25-only.
Corpus embeddings are precomputed offline (scripts/build_embeddings.py) and shipped as .npy, so the server only
encodes the query."""
import os, json, hashlib, logging, numpy as np

log = logging.getLogger('ss-rag')
D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
DENSE_MODEL = os.getenv('DENSE_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
RERANK_MODEL = os.getenv('RERANK_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
slug = lambda m: m.replace('/', '__')
corpus_sha = lambda: hashlib.sha256(open(f'{D}/ss_corpus_cleaned.csv', 'rb').read()).hexdigest()


def emb_paths(model=DENSE_MODEL):
    base = f'{D}/embeddings/{slug(model)}'
    return base + '.npy', base + '.json'


def load_dense(model=DENSE_MODEL):
    """Returns (encoder, corpus_embeddings | None, status message)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None, None, 'sentence-transformers not installed (BM25 only)'
    try:
        enc = SentenceTransformer(model)
    except Exception as e:
        log.exception('dense model load failed')
        return None, None, f'dense model {model} failed to load: {e.__class__.__name__}: {e}'
    npy, meta = emb_paths(model)
    if os.path.exists(npy) and os.path.exists(meta):
        m = json.load(open(meta))
        if m.get('corpus_sha256') == corpus_sha():
            return enc, np.load(npy), f'dense: {model} (precomputed embeddings)'
        log.warning('embedding file is for a different corpus version; re-encoding at startup')
    return enc, None, f'dense: {model} (encoded at startup; run scripts/build_embeddings.py to precompute)'


def load_reranker(model=RERANK_MODEL):
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(model), f'reranker: {model}'
    except ImportError:
        return None, 'reranker unavailable (sentence-transformers not installed)'
    except Exception as e:
        log.exception('reranker load failed')
        return None, f'reranker {model} failed to load: {e.__class__.__name__}: {e}'
