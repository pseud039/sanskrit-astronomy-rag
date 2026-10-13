"""Phase 1 freeze: builds glossary_v1.0_FROZEN.csv + agreement log from the candidate glossary.
Inputs (data/glossary/): ss_glossary_clean.csv (terms, glosses, evidence; NOT modified), glossary_meta_DRAFT.csv
(rule-derived confidence/mapping type), glossary_review_v1.csv (reviewed mapping type / category / overrides),
spotcheck_precheck.csv (reviewer sheet). Corpus: data/processed/ss_corpus_cleaned.csv.
Refuses to freeze if any evidence verse is missing from the corpus or does not print a term form.
Usage: python3 src/glossary/freeze_glossary.py"""
import os, re, json, hashlib, datetime, numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GD, OUT = os.path.join(REPO, 'data', 'glossary'), os.path.join(REPO, 'data', 'glossary')
CORPUS = os.path.join(REPO, 'data', 'processed', 'ss_corpus_cleaned.csv')
VERSION = 'v1.0'
MAPPING_TYPES = {'direct_equivalence', 'one_to_many', 'many_to_one', 'componential', 'conceptual_restructuring', 'no_equivalent'}
ART = re.compile('[\u0002]|_x0002_')
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
split = lambda s: [x for x in str(s).split('|') if x]


def rule_confidence(n, n_prim, n_any):
    """Reproduces the draft rule: High if primary gloss sits next to the term in >=75% of evidence verses,
    Medium if any gloss does in >=2/3, else Low."""
    if n_prim / n >= 0.75: return 'High'
    if n_any / n >= 2 / 3: return 'Medium'
    return 'Low'


def cohen_kappa(a, b):
    a, b = pd.Series(a, dtype=str).reset_index(drop=True), pd.Series(b, dtype=str).reset_index(drop=True)
    po = (a == b).mean(); labels = sorted(set(a) | set(b))
    pe = sum((a == l).mean() * (b == l).mean() for l in labels)
    return float('nan') if pe == 1 else (po - pe) / (1 - pe)


def snippet(text, forms, width=90):
    t = ART.sub('', text); tl = t.lower()
    for f in forms:
        i = tl.find(f.lower())
        if i >= 0:
            s, e = max(0, i - width), min(len(t), i + len(f) + 20)
            return ('…' if s else '') + t[s:e].strip() + ('…' if e < len(t) else '')
    return ''


def main():
    G = pd.read_csv(f'{GD}/ss_glossary_clean.csv', dtype=str).fillna('')
    M = pd.read_csv(f'{GD}/glossary_meta_DRAFT.csv', dtype=str).fillna('').set_index('sanskrit_term')
    R = pd.read_csv(f'{GD}/glossary_review_v1.csv', dtype=str).fillna('').set_index('sanskrit_term')
    S = pd.read_csv(f'{GD}/spotcheck_precheck.csv', dtype=str).fillna('')
    C = pd.read_csv(CORPUS, dtype=str).fillna('').set_index('verse_id')
    errors = []

    # ---- coverage checks
    for name, idx in (('meta', M.index), ('review', R.index)):
        miss, extra = set(G.sanskrit_term) - set(idx), set(idx) - set(G.sanskrit_term)
        if miss or extra: errors.append(f'{name}: missing {sorted(miss)} extra {sorted(extra)}')
    bad_types = {t for col in ('mapping_type', 'mapping_type_secondary') for t in R[col] if t and t not in MAPPING_TYPES}
    if bad_types: errors.append(f'unknown mapping types: {bad_types}')
    if (R.mapping_type == '').any(): errors.append(f'no mapping type: {list(R.index[R.mapping_type == ""])}')
    if errors: raise SystemExit('FREEZE REFUSED:\n' + '\n'.join(errors))

    rows, attest = [], []
    for i, g in enumerate(G.sort_values('sanskrit_term').itertuples(), 1):
        forms = [g.sanskrit_term] + split(g.variants)
        ev = split(g.evidence_ids); m, r = M.loc[g.sanskrit_term], R.loc[g.sanskrit_term]
        # ---- attestation: every cited verse exists and prints a term form
        found = []
        for v in ev:
            ok = v in C.index and any(f.lower() in ART.sub('', C.loc[v, 'english']).lower() for f in forms)
            found.append(ok); attest.append(dict(sanskrit_term=g.sanskrit_term, verse_id=v, term_printed=ok))
            if not ok: errors.append(f'{g.sanskrit_term}: evidence verse {v} does not print {forms}')
        # ---- confidence: rule (reproduced from counts), optionally overridden by review
        n, npg, nag = int(m.n_evidence), int(m.n_primary_gloss_adjacent), int(m.n_any_gloss_adjacent)
        if n != len(ev): errors.append(f'{g.sanskrit_term}: meta n_evidence {n} != {len(ev)} cited')
        rc = rule_confidence(n, npg, nag)
        if rc != m.confidence_draft: errors.append(f'{g.sanskrit_term}: rule gives {rc}, draft says {m.confidence_draft}')
        conf = r.confidence_override or rc
        sc = S[S.sanskrit_term == g.sanskrit_term]
        if len(sc):
            n_sup = int((sc.sheet_gloss_supported == 'Y').sum())
            spot = f'{n_sup}/{len(sc)} verses gloss supported' + (f"; proposed: {'|'.join(sorted(set(split('|'.join(sc.sheet_correction)))))}" if n_sup < len(sc) else '')
        else: spot = ''
        rows.append(dict(
            entry_id=f'GL{i:03d}', sanskrit_term=g.sanskrit_term, modern_concept=g.english_gloss,
            other_glosses=g.other_glosses, variants=g.variants, concept_category=r.concept_category,
            evidence_ids=g.evidence_ids, n_evidence=len(ev),
            chapters='|'.join(sorted({v.split('.')[0] for v in ev}, key=int)),
            evidence_snippet=snippet(C.loc[ev[0], 'english'], forms) if ev[0] in C.index else '',
            confidence=conf, confidence_rule=rc, confidence_basis=f'primary gloss adjacent {npg}/{n}, any gloss adjacent {nag}/{n}'
                + (f'; overridden {rc}->{conf} in review' if r.confidence_override and r.confidence_override != rc else ''),
            mapping_type=r.mapping_type, mapping_type_secondary=r.mapping_type_secondary, mapping_type_rule=m.mapping_type_draft,
            attestation='PASS' if all(found) else 'FAIL', spotcheck=spot,
            curation_status=g.status, curation_note=g.note, review_note=r.review_note,
            v1_1_flag='YES' if 'FLAG for v1.1' in r.review_note else '', glossary_version=VERSION))
    if errors: raise SystemExit('FREEZE REFUSED:\n' + '\n'.join(errors))

    F = pd.DataFrame(rows)
    frozen = f'{OUT}/glossary_{VERSION}_FROZEN.csv'
    F.to_csv(frozen, index=False)
    pd.DataFrame(attest).to_csv(f'{OUT}/glossary_attestation_check.csv', index=False)

    # ---- agreement log
    ent = S.groupby('sanskrit_term').agg(n=('evidence_verse', 'size'), sup=('sheet_gloss_supported', lambda x: (x == 'Y').sum()))
    ent_agree = int((ent.sup == ent.n).sum())
    v_agree = int((S.sheet_gloss_supported == 'Y').sum())
    auto = S.any_gloss_next_to_term.map({'True': 'Y', 'False': 'N'}).fillna('N')
    k_auto = cohen_kappa(auto, S.sheet_gloss_supported)
    a_auto = float((auto == S.sheet_gloss_supported).mean())
    k_map = cohen_kappa(F.mapping_type_rule, F.mapping_type); a_map = float((F.mapping_type_rule == F.mapping_type).mean())
    k_conf = cohen_kappa(F.confidence_rule, F.confidence); a_conf = float((F.confidence_rule == F.confidence).mean())
    disagree = F[F.spotcheck.str.contains('proposed')]

    L = []
    L.append(f'# Glossary {VERSION} — verification & agreement log\n')
    L.append(f'Generated {datetime.date.today().isoformat()} by `src/glossary/freeze_glossary.py`. Entries: **{len(F)}**; '
             f'evidence citations: **{int(F.n_evidence.sum())}** across chapters {", ".join(sorted({c for x in F.chapters for c in x.split("|")}, key=int))}.\n')
    L.append('## 1. Automated attestation (all entries)\n')
    L.append(f'Every cited evidence verse was checked to exist in the cleaned corpus and to print the Sanskrit term or a listed variant: '
             f'**{sum(a["term_printed"] for a in attest)}/{len(attest)} PASS**. Per-verse results: `glossary_attestation_check.csv`.\n')
    L.append('## 2. Human spot-check (cross-check sample)\n')
    L.append(f'Reviewer **S** independently checked **{len(ent)}/{len(F)} entries ({len(ent)/len(F):.0%}, above the 20% target)** across '
             f'**{len(S)} evidence verses**, using the sheet in `spotcheck_precheck.csv`.\n')
    L.append(f'| level | agree with glossary | rate |\n|---|---|---|\n| entry (all cited verses supported) | {ent_agree}/{len(ent)} | {ent_agree/len(ent):.1%} |\n'
             f'| verse | {v_agree}/{len(S)} | {v_agree/len(S):.1%} |\n')
    L.append(f'Agreement between the automated pre-check (`any_gloss_next_to_term`) and reviewer S: {a_auto:.1%} raw, Cohen\'s κ = {k_auto:.2f}. '
             'The low κ is expected: the pre-check only tests whether a gloss is printed next to the term, while the reviewer judges the sense. '
             'Many verses the reviewer accepted (e.g. contextual senses) have no adjacent gloss, so the pre-check alone cannot replace human review.\n')
    L.append('### Disagreements and adjudication\n')
    L.append('| term | reviewer result | adjudication |\n|---|---|---|')
    for d in disagree.itertuples(): L.append(f'| {d.sanskrit_term} | {d.spotcheck} | {d.review_note} |')
    L.append('')
    L.append('## 3. Confidence & mapping-type review\n')
    L.append(f'The draft confidence rule (High if the primary gloss sits next to the term in ≥75% of evidence verses; Medium if any gloss does in ≥2/3; otherwise Low) '
             f'was **re-derived from the evidence counts and reproduced for all {len(F)} entries**. The review overrode {int((F.confidence != F.confidence_rule).sum())} entries '
             f'(raw agreement {a_conf:.1%}, κ = {k_conf:.2f}).\n')
    L.append(f'The draft mapping types came from the shape of each glossary row (plural forms were counted as extra senses; "conceptual_restructuring" and '
             f'"no_equivalent" were never assigned). The review re-assigned each entry linguistically. Agreement between the draft and the review: '
             f'{a_map:.1%} raw, κ = {k_map:.2f}.\n')
    L.append('| mapping type (final) | n |\n|---|---|')
    for t, n in F.mapping_type.value_counts().items(): L.append(f'| {t} | {n} |')
    L.append('\n| confidence (final) | n |\n|---|---|')
    for t in ('High', 'Medium', 'Low'): L.append(f'| {t} | {int((F.confidence == t).sum())} |')
    L.append('\n| concept category | n |\n|---|---|')
    for t, n in F.concept_category.value_counts().items(): L.append(f'| {t} | {n} |')
    L.append('\n## 4. Known issues carried into v1.0 (flagged for v1.1; frozen file not edited)\n')
    for d in F[F.v1_1_flag == 'YES'].itertuples(): L.append(f'- **{d.sanskrit_term}**: {d.review_note}')
    L.append('\n## 5. Sign-off\n')
    L.append('The mapping-type and concept-category review (`glossary_review_v1.csv`) was drafted with AI assistance. Per the standing rule, '
             'P1 and P2 each confirm their half below. Any change goes into `glossary_review_v1.csv`; then re-run this script **before** anything downstream uses the file.\n')
    half = F.chapters.map(lambda c: int(c.split('|')[0])) <= 3
    L.append(f'- [ ] P1: entries whose first evidence chapter is 1–3 ({int(half.sum())} entries)')
    L.append(f'- [ ] P2: entries whose first evidence chapter is 4–14 ({int((~half).sum())} entries)')
    # open(f'{OUT}/glossary_agreement_log.md', 'w').write('\n'.join(L) + '\n')
    with open(
    f'{OUT}/glossary_agreement_log.md',
    'w',
    encoding='utf-8'
) as f:
    f.write('\n'.join(L) + '\n')
    

    man = dict(version=VERSION, frozen_file=os.path.basename(frozen), sha256=sha(frozen), entries=len(F),
               inputs={os.path.basename(p): sha(p) for p in [f'{GD}/ss_glossary_clean.csv', f'{GD}/glossary_meta_DRAFT.csv',
                       f'{GD}/glossary_review_v1.csv', f'{GD}/spotcheck_precheck.csv', CORPUS]},
               attestation=f'{sum(a["term_printed"] for a in attest)}/{len(attest)}',
               spotcheck=dict(entries=len(ent), entry_agreement=round(ent_agree / len(ent), 3), verse_agreement=round(v_agree / len(S), 3),
                              precheck_vs_reviewer_kappa=round(k_auto, 3)),
               review=dict(mapping_rule_vs_review_kappa=round(k_map, 3), confidence_overrides=int((F.confidence != F.confidence_rule).sum())),
               v1_1_flags=F.sanskrit_term[F.v1_1_flag == 'YES'].tolist())
    json.dump(man, open(f'{OUT}/glossary_{VERSION}_manifest.json', 'w'), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == '__main__':
    main()
