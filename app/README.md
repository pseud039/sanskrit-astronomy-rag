---
title: Surya Siddhanta Concept-Aware QA
sdk: streamlit
app_file: app.py
pinned: false
---
# Surya Siddhanta concept-aware QA (P4 app)

```bash
pip install -r requirements.txt          # light: BM25 + concept lexicon
pip install -r requirements-full.txt     # + dense retrieval and cross-encoder reranking
streamlit run app.py
python3 tests.py                         # offline tests (fake LLM server, no network)
```

## Answer generation: any LLM, no vendor SDK

Set these environment variables on the server. If none are set, the app says so in red and shows only the retrieved verses.

| provider | `LLM_PROVIDER` | `LLM_BASE_URL` | `LLM_MODEL` (example) | `LLM_API_KEY` |
|---|---|---|---|---|
| Ollama (local) | `ollama` | `http://localhost:11434` | `qwen2.5:7b-instruct` | – |
| Ollama (hosted / remote box) | `ollama` | `https://your-ollama-host` | `llama3.1:8b` | if your proxy needs one |
| Groq (free tier, fast) | `openai` | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` | Groq key |
| OpenRouter | `openai` | `https://openrouter.ai/api/v1` | e.g. `meta-llama/llama-3.3-70b-instruct` | OpenRouter key |
| Hugging Face router | `openai` | `https://router.huggingface.co/v1` | e.g. `Qwen/Qwen2.5-72B-Instruct` | HF token |
| any OpenAI-compatible server (vLLM, LM Studio, Ollama `/v1`) | `openai` | its `/v1` URL | – | if needed |

Optional: `LLM_TIMEOUT` (seconds, default 60). Generation errors (bad key, unreachable host, non-JSON output) are shown in the UI and logged. Every cited verse and quote is checked against the retrieved verses.

## Retrieval

- **Methods**: BM25 baseline; Hybrid (BM25 + dense, reciprocal rank fusion); Proposed (concept expansion from the frozen glossary + frozen modern-phrasing lexicon, fused with dense when available).
- **Dense / rerank**: enable with `ENABLE_DENSE=1` / `ENABLE_RERANK=1` (needs `requirements-full.txt`). `DENSE_MODEL` (default `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) and `RERANK_MODEL` (default `cross-encoder/ms-marco-MiniLM-L-6-v2`) can be overridden. Precompute corpus embeddings once with `python3 scripts/build_embeddings.py` and commit `data/embeddings/`.
- **Abstention gate**: `data/gate.json`, chosen by `src/evaluation/run_benchmark.py` on the dev split among raw BM25 top-1, IDF-normalised BM25 and reranker top score, and deployed only if dev balanced accuracy is ≥ 0.7. Current choice: raw BM25 top-1 ≥ 17.25 (dev 0.88; test refuses 6/6 unanswerable and 8/61 answerable). If the gate passes but the model finds no support, it answers "The text does not answer this".

## Deploying

- **Render (512 MB free tier)**: build `pip install -r requirements.txt`, start `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`, root directory `app`. Use a hosted LLM (Groq / OpenRouter / HF / your own Ollama host). Dense and reranking don't fit in 512 MB.
- **Hugging Face Spaces (≈16 GB CPU RAM)**: push this folder, replace `requirements.txt` with the contents of `requirements-full.txt`, set `ENABLE_DENSE=1`, `ENABLE_RERANK=1` and the `LLM_*` variables as Space secrets.

Data: `data/ss_corpus_cleaned.csv` (427 verses), `data/ss_glossary_clean.csv` + `data/glossary_v1.0_FROZEN.csv` (57 entries),
`data/concept_lexicon_v1_FROZEN.csv` (58 modern-phrasing concepts), `data/splits_v1.csv` (dev/test).
