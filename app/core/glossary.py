import re

def forms(row):
    return [row.sanskrit_term] + [v for v in row.variants.split('|') if v]

def match_entries(query, G, exclude=()):
    """Glossary entries triggered by the query (term, variants, primary/other glosses)."""
    ql = ' ' + re.sub(r'[^a-z0-9\- ]', ' ', query.lower()) + ' '
    hits = []
    for r in G.itertuples():
        if r.sanskrit_term in exclude: continue
        keys = {r.sanskrit_term, r.english_gloss, *[x for x in r.other_glosses.split('|') if x],
                *[x for x in r.variants.split('|') if x]}
        trig = [k for k in keys if k and re.search(r'(?<![a-z])' + re.escape(k.lower()) + r'(s|es)?(?![a-z])', ql)]
        if trig: hits.append(dict(term=r.sanskrit_term, gloss=r.english_gloss, forms=forms(r), triggered_by=sorted(trig),
                              confidence=getattr(r, 'confidence', ''), mapping_type=getattr(r, 'mapping_type', '')))
    return hits
