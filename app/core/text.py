"""Display/LLM-side text cleanup. Retrieval indexes the corpus as frozen; this only fixes what users and the LLM see."""
import re

ART = re.compile('[\u0002]|_x0002_')
MOJIBAKE = {'â€”': '—', 'â€™': "'", 'â€œ': '"', 'â€\x9d': '"'}


def ocr_map(G):
    """OCR'd Sanskrit forms in parentheses (e.g. 'mdna', 'krdnti') -> the glossary spelling, from glossary rows
    whose note records an OCR fix."""
    m = {}
    for r in G.itertuples():
        if 'OCR' in r.note:
            for v in [x for x in r.variants.split('|') if x]: m[v] = r.sanskrit_term
    return m


def clean_english(t, ocr=None):
    t = ART.sub('', str(t))
    for a, b in MOJIBAKE.items(): t = t.replace(a, b)
    if ocr:
        t = re.sub(r'([\(\{])\s*([a-z\-]+)\s*([\)\}])',
                   lambda m: f'({ocr.get(m.group(2), m.group(2))})', t)
    return re.sub(r'\s+', ' ', t).strip()


def sanskrit_display(itrans, script='IAST'):
    """ITRANS -> IAST / Devanagari via indic_transliteration when installed; raw ITRANS otherwise."""
    if script == 'ITRANS': return itrans
    try:
        from indic_transliteration import sanscript
        target = sanscript.DEVANAGARI if script == 'Devanagari' else sanscript.IAST
        return sanscript.transliterate(itrans, sanscript.ITRANS, target)
    except Exception:
        return itrans
