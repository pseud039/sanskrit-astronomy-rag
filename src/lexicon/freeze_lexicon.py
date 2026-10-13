"""Freeze the modern-phrasing concept lexicon (v1).
Input : data/lexicon/concept_lexicon_v1.csv (hand-authored from the frozen glossary, corpus "English (sanskrit)"
        parentheticals and standard astronomy vocabulary; NOT from benchmark questions or their failures).
Output: data/lexicon/concept_lexicon_v1_FROZEN.csv + manifest. A text term is kept only if it occurs (word-bounded)
        in the cleaned corpus, because an expansion term the corpus never prints cannot help retrieval.
        This filter uses the corpus only, never the benchmark.
Usage : python3 src/lexicon/freeze_lexicon.py"""
import os, re, json, hashlib, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LX = os.path.join(REPO, 'data', 'lexicon')
SRC, OUT = f'{LX}/concept_lexicon_v1.csv', f'{LX}/concept_lexicon_v1_FROZEN.csv'
CORPUS = os.path.join(REPO, 'data', 'processed', 'ss_corpus_cleaned.csv')
GLOSS = os.path.join(REPO, 'data', 'glossary', 'ss_glossary_clean.csv')
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
split = lambda s: [x for x in str(s).split('|') if x]


def main():
    L = pd.read_csv(SRC, dtype=str).fillna('')
    G = pd.read_csv(GLOSS, dtype=str).fillna('')
    txt = ' ' + ' '.join(pd.read_csv(CORPUS, dtype=str).fillna('').english.str.lower()) + ' '
    occurs = lambda t: bool(re.search(r'(?<![a-z])' + re.escape(t.lower()) + r'(?![a-z])', txt))
    errors = []
    if L.concept_id.duplicated().any(): errors.append(f'duplicate concept ids: {L.concept_id[L.concept_id.duplicated()].tolist()}')
    bad_g = {g for x in L.glossary_terms for g in split(x)} - set(G.sanskrit_term)
    if bad_g: errors.append(f'unknown glossary terms: {sorted(bad_g)}')
    if errors: raise SystemExit('FREEZE REFUSED:\n' + '\n'.join(errors))
    L['text_terms_active'] = L.text_terms.map(lambda s: '|'.join(t for t in split(s) if occurs(t)))
    L['text_terms_dropped'] = L.text_terms.map(lambda s: '|'.join(t for t in split(s) if not occurs(t)))
    L['lexicon_version'] = 'v1.0'
    L.to_csv(OUT, index=False, lineterminator='\n', encoding='utf-8')
    man = dict(version='v1.0', frozen_file=os.path.basename(OUT), sha256=sha(OUT), concepts=len(L),
               aliases=int(L.modern_aliases.map(lambda s: len(split(s))).sum()),
               text_terms_active=int(L.text_terms_active.map(lambda s: len(split(s))).sum()),
               text_terms_dropped=int(L.text_terms_dropped.map(lambda s: len(split(s))).sum()),
               inputs={os.path.basename(p): sha(p) for p in (SRC, CORPUS, GLOSS)},
               protocol='Authored without reference to benchmark failures. Question texts the author had seen before '
                        'freezing are listed in data/benchmark/seen_by_lexicon_author.txt and are excluded from the '
                        'headline test metrics.')
    json.dump(man, open(f'{LX}/concept_lexicon_v1_manifest.json', 'w', encoding='utf-8', newline='\n'), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == '__main__':
    main()
