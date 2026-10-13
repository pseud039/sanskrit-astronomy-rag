"""Concept detection: (1) frozen glossary (Sanskrit term / translator gloss), (2) frozen modern-phrasing lexicon,
(3) optional dense match of the query against lexicon concept descriptions. Each hit carries the text-side terms
added to the BM25 query."""
import re, numpy as np
from .glossary import match_entries

WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def stem(w):
    """Plural folding only (ies->y, s->''): consistent for singular/plural pairs, no over-stemming."""
    w = w.replace("'s", '')
    if len(w) > 4 and w.endswith('ies'): return w[:-3] + 'y'
    if len(w) > 3 and w.endswith('s') and not w.endswith('ss'): return w[:-1]
    return w


def toks(s): return [stem(w) for w in WORD.findall(s.lower())]


def alias_hit(alias_toks, q_toks, window=6):
    """Contiguous match, or (multi-word aliases) all alias tokens inside a short window of the query."""
    n = len(alias_toks)
    if not n: return False
    for i in range(len(q_toks) - n + 1):
        if q_toks[i:i + n] == alias_toks: return True
    if n > 1:
        need = set(alias_toks)
        for i in range(len(q_toks)):
            if need <= set(q_toks[i:i + window + n]): return True
    return False


class ConceptMatcher:
    def __init__(self, glossary, lexicon=None, dense=None, dense_threshold=0.55):
        self.G, self.L, self.dense, self.thr = glossary, lexicon, dense, dense_threshold
        self.gl = glossary.set_index('sanskrit_term')
        if lexicon is not None:
            self.alias_toks = [[(a, toks(a)) for a in r.modern_aliases.split('|') if a] for r in lexicon.itertuples()]
            if dense is not None:
                docs = [f"{r.concept_id.replace('_', ' ')}: {r.description} ({r.modern_aliases.replace('|', ', ')})"
                        for r in lexicon.itertuples()]
                self.concept_emb = dense.encode(docs, normalize_embeddings=True)

    def _lex_hit(self, r, triggered_by, source, exclude):
        terms = [t for t in r.text_terms_active.split('|') if t]
        gl_terms = [g for g in r.glossary_terms.split('|') if g and g not in exclude]
        for g in gl_terms: terms += [g] + [v for v in self.gl.loc[g, 'variants'].split('|') if v]
        return dict(source=source, id=r.concept_id, label=r.description, triggered_by=triggered_by,
                    terms=list(dict.fromkeys(terms)), glossary_terms=gl_terms, confidence='', mapping_type='')

    def match(self, query, exclude=(), use_lexicon=True, use_glossary=True, use_dense=True):
        hits = []
        if use_glossary:
            for h in match_entries(query, self.G, exclude):
                hits.append(dict(source='glossary', id=h['term'], label=h['gloss'], triggered_by=h['triggered_by'],
                                 terms=h['forms'], glossary_terms=[h['term']], confidence=h.get('confidence', ''),
                                 mapping_type=h.get('mapping_type', '')))
        if use_lexicon and self.L is not None:
            qt, found = toks(query), set()
            for r, aliases in zip(self.L.itertuples(), self.alias_toks):
                trig = [a for a, at in aliases if alias_hit(at, qt)]
                if trig: hits.append(self._lex_hit(r, trig, 'lexicon', exclude)); found.add(r.concept_id)
            if use_dense and self.dense is not None:
                q = self.dense.encode([query], normalize_embeddings=True)[0]; sims = self.concept_emb @ q
                for i in np.argsort(-sims)[:2]:
                    r = self.L.iloc[i]
                    if sims[i] >= self.thr and r.concept_id not in found:
                        hits.append(self._lex_hit(r, [f'semantic match {sims[i]:.2f}'], 'dense', exclude))
        return hits
