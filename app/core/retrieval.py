import re, numpy as np
from rank_bm25 import BM25Okapi
from .concepts import ConceptMatcher

tok = lambda s: re.findall(r'[a-z]+', s.lower())

METHODS = {
    'bm25': 'BM25 (baseline)',
    'hybrid': 'Hybrid: BM25 + dense (RRF)',
    'proposed': 'Proposed: concept-aware expansion (glossary + lexicon)',
}


class Retriever:
    """bm25     : BM25 on the question.
    hybrid   : RRF(BM25, dense) - needs a dense encoder.
    proposed : BM25(question) + w * BM25(concept expansion terms); fused with dense via RRF when available.
    rerank   : optional cross-encoder over the top `rerank_depth` candidates of any method.
    gate     : candidate abstention scores - raw BM25 top-1, BM25 top-1 normalised by the query's IDF mass, and the
               reranker's top score when a reranker is loaded. Which one is used (and its threshold) is chosen on the
               DEV split by src/evaluation/run_benchmark.py and stored in data/gate.json."""

    def __init__(self, corpus, glossary, lexicon=None, w=0.5, dense=None, corpus_emb=None, reranker=None,
                 rerank_depth=20, rrf_k=60):
        self.C, self.G, self.L, self.w = corpus, glossary, lexicon, w
        self.ids = corpus.verse_id.tolist()
        self.display = corpus['english_display'] if 'english_display' in corpus else corpus.english
        self.bm = BM25Okapi([tok(t) for t in corpus.english])
        self.dense, self.reranker, self.depth, self.rrf_k = dense, reranker, rerank_depth, rrf_k
        self.emb = corpus_emb
        if dense is not None and self.emb is None:
            self.emb = dense.encode(corpus.english.tolist(), normalize_embeddings=True)
        self.concepts = ConceptMatcher(glossary, lexicon, dense)

    @property
    def has_dense(self): return self.dense is not None

    def available_methods(self):
        return [m for m in METHODS if m != 'hybrid' or self.has_dense]

    def _rrf(self, *scores):
        fused = np.zeros(len(self.ids))
        for s in scores: fused += 1 / (self.rrf_k + 1 + np.argsort(np.argsort(-s)))
        return fused

    def bm25_gate(self, query):
        qt = [t for t in set(tok(query)) if self.bm.idf.get(t, 0) > 0]
        top = float(np.max(self.bm.get_scores(tok(query)))) if qt else 0.0
        return dict(bm25_top1=top, bm25_norm=top / (sum(self.bm.idf[t] for t in qt) or 1.0))

    def scores(self, query, method='bm25', exclude=(), use_lexicon=True, use_glossary=True):
        s = self.bm.get_scores(tok(query)); hits, exp = [], []
        if method == 'proposed':
            hits = self.concepts.match(query, exclude, use_lexicon=use_lexicon, use_glossary=use_glossary)
            exp = list(dict.fromkeys(t for h in hits for t in h['terms']))
            if exp: s = s + self.w * self.bm.get_scores(tok(' '.join(exp)))
        if method in ('hybrid', 'proposed') and self.has_dense:
            q = self.dense.encode([query], normalize_embeddings=True)[0]
            s = self._rrf(s, self.emb @ q)
        elif method == 'hybrid':
            raise ValueError('hybrid needs a dense encoder')
        return s, hits, exp

    def search(self, query, method='bm25', k=5, exclude=(), rerank=False, **kw):
        s, hits, exp = self.scores(query, method, exclude, **kw)
        order = np.argsort(-s)
        rr = None
        if rerank and self.reranker is not None:
            cand = order[:self.depth]
            rr = np.asarray(self.reranker.predict([(query, self.C.english.iloc[i]) for i in cand]), float)
            order = np.concatenate([cand[np.argsort(-rr)], order[self.depth:]])
            rr = np.sort(rr)[::-1]
        order = order[:k] if k else order
        res = [dict(verse_id=self.ids[i], score=float(s[i]), english=self.display.iloc[i],
                    sanskrit=self.C.sanskrit_itrans.iloc[i], chapter=self.C.chapter.iloc[i]) for i in order]
        gate = self.bm25_gate(query)
        if rr is not None: gate['rerank_top1'] = float(rr[0])
        return dict(results=res, matched=hits, expansion=exp, gate=gate, method=method, reranked=rr is not None)
