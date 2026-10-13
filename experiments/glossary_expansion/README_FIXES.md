# Glossary fixes package (does NOT modify the frozen glossary)

| file | what it is |
|---|---|
| ss_corpus_clean.csv | 427 gold-eligible verses (English present, not garbled). Index THIS, not ss_corpus_all.csv |
| ss_corpus_removed.csv | 73 removed verses (55 no English, 18 garbled) + reason. None is a gold/evidence verse |
| ss_questions_leakage.csv | per-question leakage flags + exclude lists (strict / stricter) + how many glossary entries can fire |
| leakage_utils.py | allowed_glossary(qid,...), match_entries(), subset_masks() -> ALL / CLEAN / OVERLAP |
| eval_utils.py | Recall@k, MRR, nDCG, bootstrap CIs, paired bootstrap, chapter-grouped folds (trained models only) |
| smoke_test.py | BM25 +/- glossary expansion across leakage modes (pipeline check, NOT a result) |
| spotcheck_precheck.csv | independent check of the 18 spot-check entries across ALL their evidence verses |
| glossary_meta_DRAFT.csv | confidence + mapping_type sidecar; rule-derived DRAFT, verified_by_human=NO |
| score_verification.py | scores the team's filled verification sheets against the organizer key |

Leakage modes: strict = drop entries citing the gold verse in evidence_ids; stricter = also drop entries whose
term/variant is printed in the gold verse (OCR artifacts stripped before matching).
Run: python3 build_leakage_table.py <path-to-glossary_freeze>
