import os, pandas as pd
from .text import clean_english, ocr_map

D = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def load_glossary():
    G = pd.read_csv(f'{D}/ss_glossary_clean.csv', dtype=str).fillna('')
    fz = f'{D}/glossary_v1.0_FROZEN.csv'  # Phase 1 freeze: adds confidence / mapping type for display
    if os.path.exists(fz):
        F = pd.read_csv(fz, dtype=str).fillna('')[['sanskrit_term', 'confidence', 'mapping_type', 'concept_category']]
        G = G.merge(F, on='sanskrit_term', how='left').fillna('')
    return G


def load_corpus():
    """Corpus as indexed for retrieval (frozen text, artifacts stripped) + `english_display` for users / the LLM,
    where OCR'd Sanskrit in parentheses (e.g. 'mdna') is shown in its glossary spelling ('mana')."""
    C = pd.read_csv(f'{D}/ss_corpus_cleaned.csv', dtype=str).fillna('')
    C['english'] = C.english.map(clean_english)
    m = ocr_map(load_glossary())
    C['english_display'] = C.english.map(lambda t: clean_english(t, m))
    return C.reset_index(drop=True)


def load_lexicon():
    p = f'{D}/concept_lexicon_v1_FROZEN.csv'
    return pd.read_csv(p, dtype=str).fillna('') if os.path.exists(p) else None


def load_questions():
    return pd.read_csv(f'{D}/ss_questions_all.csv', dtype=str)
