"""Leakage-aware glossary access. Does NOT modify the frozen glossary.
Printed-term rule (stricter mode): case-insensitive SUBSTRING of a term/variant in the gold verse text (OCR artifacts
stripped, whitespace kept). Whitespace is NOT removed: doing so creates cross-word false matches (e.g. 'east and' -> 'asta').
Sensitivity rules: 'left' (left word-boundary only) and 'wb' (full word boundary)."""
import os, re, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(REPO, 'data')
ART = re.compile('[\u0002]|_x0002_')

def load_all(root=DATA, corpus_path=None):
    G = pd.read_csv(f'{root}/glossary/ss_glossary_clean.csv', dtype=str).fillna('')
    C = pd.read_csv(corpus_path or f'{root}/processed/ss_corpus_cleaned.csv', dtype=str).fillna('').set_index('verse_id')
    Q = pd.read_csv(f'{root}/benchmark/ss_questions_all.csv', dtype=str)
    return G, C, Q[Q.type == 'answerable'].reset_index(drop=True)

def forms(row): return [row.sanskrit_term] + [v for v in row.variants.split('|') if v]

def evidence_overlap(gold_verse, G):
    return {r.sanskrit_term for r in G.itertuples() if gold_verse in r.evidence_ids.split('|')}

def printed_in_gold(gold_verse, G, C, rule='sub'):
    t = ART.sub('', C.loc[gold_verse, 'english'].lower()); out = set()
    for r in G.itertuples():
        for f in map(str.lower, forms(r)):
            if (rule == 'sub' and f in t) or (rule == 'left' and re.search(r'(?<![a-z])' + re.escape(f), t)) \
               or (rule == 'wb' and re.search(r'(?<![a-z])' + re.escape(f) + r'(?![a-z])', t)):
                out.add(r.sanskrit_term); break
    return out

def exclude_entries(gold_verse, G, C, mode, rule='sub'):
    if mode == 'all': return set()
    if mode == 'strict': return evidence_overlap(gold_verse, G)
    if mode == 'stricter': return evidence_overlap(gold_verse, G) | printed_in_gold(gold_verse, G, C, rule)
    raise ValueError(mode)

def allowed_glossary(qid, G, Q, mode='strict', C=None, rule='sub'):
    gv = Q.loc[Q.qid == qid, 'gold_verse'].iloc[0]
    return G[~G.sanskrit_term.isin(exclude_entries(gv, G, C, mode, rule))]

def match_entries(query, Gsub):
    ql = ' ' + re.sub(r'[^a-z0-9\- ]', ' ', query.lower()) + ' '; hit = []
    for r in Gsub.itertuples():
        keys = {r.sanskrit_term, r.english_gloss, *[x for x in r.other_glosses.split('|') if x], *[x for x in r.variants.split('|') if x]}
        if any(re.search(r'(?<![a-z])' + re.escape(k.lower()) + r'(s|es)?(?![a-z])', ql) for k in keys if k): hit.append(r.sanskrit_term)
    return hit

def subset_masks(Q, G, C, sensitivity=False):
    out = {'ALL': pd.Series(True, index=Q.index)}
    specs = [('strict', 'strict', 'sub'), ('stricter', 'stricter', 'sub')]
    if sensitivity: specs += [('stricter', 'stricter_left', 'left'), ('stricter', 'stricter_wb', 'wb')]
    for mode, tag, rule in specs:
        ov = Q.gold_verse.map(lambda v: bool(exclude_entries(v, G, C, mode, rule)))
        out[f'CLEAN_{tag}'] = ~ov; out[f'OVERLAP_{tag}'] = ov
    return out
