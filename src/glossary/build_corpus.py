"""Builds the retrieval corpus in two explicit steps and writes a manifest with hashes.
 ss_corpus_filtered.csv : rows only (role == gold_eligible). Text UNTOUCHED.
 ss_corpus_cleaned.csv  : filtered + conservative text cleaning + flags. Notes are flagged, NOT removed.
Usage: python3 build_corpus.py <path-to-glossary_freeze>"""
import sys, re, json, hashlib, collections, pandas as pd
R = sys.argv[1]
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
raw = pd.read_csv(f'{R}/data/ss_corpus_all.csv', dtype=str).fillna('')
al = pd.read_csv(f'{R}/data/ss_aligned.csv', dtype=str)
al['verse_id'] = al.chapter + '.' + al.verse
flt = raw[raw.role == 'gold_eligible'].reset_index(drop=True); flt.to_csv('ss_corpus_filtered.csv', index=False)

ART = re.compile('[\u0002]|_x0002_')
strip_art = lambda t: ART.sub('', t)
vocab = collections.Counter(w for t in raw.english for w in re.findall(r'[a-z]+', strip_art(t).lower()))
joins = []
def fix_hyphen(t):
    def rep(m):
        a, b = m.group(1), m.group(2)
        if vocab[(a + b).lower()] > 0: joins.append(a + b); return a + b
        return m.group(0)
    return re.sub(r'([A-Za-z]{2,})- ([a-z]{2,})', rep, t)
NOTE = re.compile(r'\b(The usual|These verses|This verse|The translator|Whitney|Colebrooke|Bentley|commentator|The text reads|MS\.)', re.I)

c = flt.copy(); c['english_raw_len'] = c.english.str.len()
c['english'] = c.english.map(lambda t: re.sub(r'\s+', ' ', fix_hyphen(strip_art(t))).strip())
lb = al.set_index('verse_id').long_block
c['long_block'] = c.verse_id.map(lb).fillna('')
c['truncated_cap'] = (c.english_raw_len >= 600)                    # source capped long blocks at 600 chars
c['note_suspect'] = c.english.map(lambda t: bool(NOTE.search(t)))   # flagged only; not removed
c.to_csv('ss_corpus_cleaned.csv', index=False)

# integrity: every glossary evidence verse still contains a term form after cleaning
G = pd.read_csv(f'{R}/glossary/ss_glossary_clean.csv', dtype=str).fillna(''); ci = c.set_index('verse_id').english.str.lower()
bad = [(r.sanskrit_term, e) for r in G.itertuples() for e in r.evidence_ids.split('|')
       if not any(f.lower() in ci[e] for f in [r.sanskrit_term] + [v for v in r.variants.split('|') if v])]
man = dict(source=dict(file='ss_corpus_all.csv', sha256=sha(f'{R}/data/ss_corpus_all.csv'), rows=len(raw)),
    filtered=dict(file='ss_corpus_filtered.csv', sha256=sha('ss_corpus_filtered.csv'), rows=len(flt), removed=len(raw) - len(flt)),
    cleaned=dict(file='ss_corpus_cleaned.csv', sha256=sha('ss_corpus_cleaned.csv'), rows=len(c)),
    glossary_sha256=sha(f'{R}/glossary/ss_glossary_clean.csv'),
    cleaning=dict(rules=['strip \\x02 / _x0002_ artifacts', 'join "xxx- yyy" only if the joined word occurs elsewhere in corpus', 'collapse whitespace'],
        hyphen_joins=len(joins), rows_changed=int((c.english != flt.english).sum()), truncated_cap_rows=int(c.truncated_cap.sum()), note_suspect_rows=int(c.note_suspect.sum()),
        known_limits=['English capped at 600 chars for long blocks (truncated_cap flag)', 'translator notes are flagged, not removed']),
    evidence_check_failures=bad)
json.dump(man, open('corpus_manifest.json', 'w'), indent=1); print(json.dumps(man, indent=1)[:1800])
print('remaining artifacts:', c.english.str.contains('\u0002|_x0002_').sum(), '| leftover "x- y" breaks:', c.english.str.contains(r'[A-Za-z]- [a-z]').sum())
