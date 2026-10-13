import os, re, pandas as pd
D = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def clean_text(t):
    t = re.sub(r'[\u0002]|_x0002_', '', str(t))
    return t.replace('â€”', '—').replace('â€™', "'").strip()

def load_corpus():
    C = pd.read_csv(f'{D}/ss_corpus_cleaned.csv', dtype=str).fillna('')
    C['english'] = C.english.map(clean_text)
    return C.reset_index(drop=True)

def load_glossary():
    G = pd.read_csv(f'{D}/ss_glossary_clean.csv', dtype=str).fillna('')
    fz = f'{D}/glossary_v1.0_FROZEN.csv'  # Phase 1 freeze: adds confidence / mapping type for display
    if os.path.exists(fz):
        F = pd.read_csv(fz, dtype=str).fillna('')[['sanskrit_term', 'confidence', 'mapping_type', 'concept_category']]
        G = G.merge(F, on='sanskrit_term', how='left').fillna('')
    return G

def load_questions():
    return pd.read_csv(f'{D}/ss_questions_all.csv', dtype=str)
