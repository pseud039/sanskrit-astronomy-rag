# Glossary v1.0 — verification & agreement log

Generated 2026-10-13 by `src/glossary/freeze_glossary.py`. Entries: **57**; evidence citations: **170** across chapters 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14.

## 1. Automated attestation (all entries)

Every cited evidence verse was checked to exist in the cleaned corpus and to print the Sanskrit term or a listed variant: **170/170 PASS**. Per-verse results: `glossary_attestation_check.csv`.

## 2. Human spot-check (cross-check sample)

Reviewer **S** independently checked **18/57 entries (32%, above the 20% target)** across **62 evidence verses**, using the sheet in `spotcheck_precheck.csv`.

| level | agree with glossary | rate |
|---|---|---|
| entry (all cited verses supported) | 16/18 | 88.9% |
| verse | 54/62 | 87.1% |

Agreement between the automated pre-check (`any_gloss_next_to_term`) and reviewer S: 79.0% raw, Cohen's κ = 0.11. The low κ is expected: the pre-check only tests whether a gloss is printed next to the term, while the reviewer judges the sense. Many verses the reviewer accepted (e.g. contextual senses) have no adjacent gloss, so the pre-check alone cannot replace human review.

### Disagreements and adjudication

| term | reviewer result | adjudication |
|---|---|---|
| ayana | 0/6 verses gloss supported; proposed: precession | Solstice, the sun's northward/southward progress, and precession (ayanamsa) are all cited senses. Reviewer S proposed 'precession' as the primary gloss; adjudicated KEEP 'solstice' as primary (4 of 6 verses print solstice), with precession kept in other_glosses. |
| yuga | 0/2 verses gloss supported; proposed: epoch | Cosmological age (13.18 prints 'successive age (yuga)'). Reviewer S proposed 'epoch' and is CORRECT for 1.56, which prints 'any epoch (yuga)'. The glossary note 'epoch unsupported' is wrong for that verse. Glossary frozen, so no edit: FLAG for v1.1 (add 'epoch' to other_glosses). Confidence stays Low. |

## 3. Confidence & mapping-type review

The draft confidence rule (High if the primary gloss sits next to the term in ≥75% of evidence verses; Medium if any gloss does in ≥2/3; otherwise Low) was **re-derived from the evidence counts and reproduced for all 57 entries**. The review overrode 3 entries (raw agreement 94.7%, κ = 0.89).

The draft mapping types came from the shape of each glossary row (plural forms were counted as extra senses; "conceptual_restructuring" and "no_equivalent" were never assigned). The review re-assigned each entry linguistically. Agreement between the draft and the review: 52.6% raw, κ = 0.40.

| mapping type (final) | n |
|---|---|
| direct_equivalence | 15 |
| one_to_many | 14 |
| conceptual_restructuring | 11 |
| no_equivalent | 8 |
| componential | 5 |
| many_to_one | 4 |

| confidence (final) | n |
|---|---|
| High | 36 |
| Medium | 16 |
| Low | 5 |

| concept category | n |
|---|---|
| planetary_theory | 12 |
| cosmology_philosophy | 11 |
| spherical_astronomy | 11 |
| calendrics | 7 |
| eclipses_parallax | 7 |
| units_measurement | 4 |
| mathematics_geometry | 3 |
| phenomena_visibility | 1 |
| instruments | 1 |

## 4. Known issues carried into v1.0 (flagged for v1.1; frozen file not edited)

- **aksha**: aksha = terrestrial latitude. Draft 'componential' came only from the two-word gloss. FLAG for v1.1: variants lamba/lambaka are CO-latitude, as the corpus itself shows at 3.14 'co-latitude (lamba) and of latitude (aksha)' and 12.44 'co-latitude (lambaka)'. They are kept only because the glossary is frozen.
- **yuga**: Cosmological age (13.18 prints 'successive age (yuga)'). Reviewer S proposed 'epoch' and is CORRECT for 1.56, which prints 'any epoch (yuga)'. The glossary note 'epoch unsupported' is wrong for that verse. Glossary frozen, so no edit: FLAG for v1.1 (add 'epoch' to other_glosses). Confidence stays Low.

## 5. Sign-off

The mapping-type and concept-category review (`glossary_review_v1.csv`) was drafted with AI assistance. Per the standing rule, P1 and P2 each confirm their half below. Any change goes into `glossary_review_v1.csv`; then re-run this script **before** anything downstream uses the file.

- [ ] P1: entries whose first evidence chapter is 1–3 (28 entries)
- [ ] P2: entries whose first evidence chapter is 4–14 (29 entries)
