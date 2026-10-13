import re, numpy as np
from rank_bm25 import BM25Okapi
from .glossary import match_entries

tok = lambda s: re.findall(r'[a-z]+', s.lower())

class Retriever:
    """mode='baseline': BM25 on the question. mode='proposed': BM25(question) + w * BM25(glossary expansion terms).
    Optional dense encoder (sentence-transformers) is fused with RRF when available."""
    def __init__(self, corpus, glossary, w=0.5, dense=None):
        self.C, self.G, self.w = corpus, glossary, w
        self.ids = corpus.verse_id.tolist()
        self.bm = BM25Okapi([tok(t) for t in corpus.english])
        self.dense = dense
        if dense is not None:
            self.emb = dense.encode(corpus.english.tolist(), normalize_embeddings=True)

    def scores(self, query, mode='baseline', exclude=()):
        s = self.bm.get_scores(tok(query)); hits = []; exp = []
        if mode == 'proposed':
            hits = match_entries(query, self.G, exclude)
            exp = [f for h in hits for f in h['forms']]
            if exp: s = s + self.w * self.bm.get_scores(tok(' '.join(exp)))
        return s, hits, exp

    def search(self, query, mode='baseline', k=5, exclude=()):
        s, hits, exp = self.scores(query, mode, exclude)
        if self.dense is not None:
            q = self.dense.encode([query], normalize_embeddings=True)[0]; d = self.emb @ q
            rb, rd = np.argsort(np.argsort(-s)), np.argsort(np.argsort(-d))
            fused = 1 / (60 + rb) + 1 / (60 + rd); order = np.argsort(-fused)[:k]
        else:
            order = np.argsort(-s)[:k]
        res = [dict(verse_id=self.ids[i], score=float(s[i]), english=self.C.english.iloc[i],
                    sanskrit=self.C.sanskrit_itrans.iloc[i], chapter=self.C.chapter.iloc[i]) for i in order]
        return dict(results=res, matched=hits, expansion=exp, top_scores=np.sort(s)[::-1][:3].tolist())
