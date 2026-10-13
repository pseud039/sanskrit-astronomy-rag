---
title: Surya Siddhanta Concept-Aware QA
sdk: streamlit
app_file: app.py
pinned: false
---
# Surya Siddhanta concept-aware QA (P4 app)
Run: `pip install -r requirements.txt && ANTHROPIC_API_KEY=... streamlit run app.py`
Without a key the app still runs and falls back to the top retrieved verse.
Hugging Face Spaces: push this folder, add `ANTHROPIC_API_KEY` as a Space secret.
Env: `CLAUDE_MODEL` (default claude-sonnet-5-5), `DENSE_MODEL` (default paraphrase-multilingual-MiniLM-L12-v2).
Data: `data/ss_corpus_cleaned.csv` (427 verses), `data/ss_glossary_clean.csv` (57 entries) + `data/glossary_v1.0_FROZEN.csv` (Phase 1 freeze: confidence / mapping type shown in the concept table).
