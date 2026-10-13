# Sanskrit Astronomy RAG

### Concept-Aware Query Expansion for Terminology-Gap Retrieval in Historical Sanskrit Astronomical Texts

A research-oriented Retrieval-Augmented Generation (RAG) system for **Sūrya Siddhānta** that investigates whether explicitly mapping modern astronomical concepts to historical Sanskrit terminology can improve retrieval when conventional search and semantic retrieval struggle with vocabulary mismatch.

> **Research question:**
> Does concept-aware query expansion improve retrieval performance as the terminology gap between a modern query and a historical Sanskrit astronomical text increases?

---

## Overview

Historical scientific texts present a unique information-retrieval challenge.

A modern user may ask:

> *"How does the text describe the apparent motion of a planet?"*

However, the historical text may describe the same underlying concept using terminology, expressions, and linguistic structures that differ substantially from modern scientific language.

Conventional keyword search and even semantic retrieval may therefore fail to retrieve the passage that actually contains the relevant information.

This project explores whether a **concept-aware query expansion layer** can bridge this gap before retrieval.

The system:

```text
User Query
    ↓
Query Understanding
    ↓
Astronomical Concept Identification
    ↓
Concept-Aware Terminology Expansion
    ↓
┌───────────────────────────────┐
│ BM25 / Dense / Hybrid Search  │
└───────────────────────────────┘
    ↓
Candidate Passages
    ↓
Reranking
    ↓
Relevant Evidence
    ↓
Grounded RAG
    ↓
GenAI Answer + Citation
```

The generated answer is grounded in the retrieved passages and linked back to the corresponding Sanskrit source.

---

## Research Motivation

Most RAG systems assume that the vocabulary used by a user is reasonably close to the vocabulary present in the underlying corpus.

That assumption becomes problematic for historical scientific texts.

This project therefore studies a more specific question:

> **Can concept-level terminology mapping rescue retrieval when modern and historical terminology diverge?**

Rather than simply building another RAG chatbot, we evaluate **why and when** terminology-aware retrieval helps.

The core hypothesis is:

> **The benefit of concept-aware query expansion should increase as the terminology gap between the query and the relevant historical passage becomes larger.**

This gives the project a measurable research contribution rather than making the system merely an application.

---

## Research Contributions

The project focuses on four main contributions:

### 1. Terminology-aware retrieval

A concept mapping layer connects modern astronomical concepts with terminology used in the historical Sanskrit corpus.

### 2. Terminology-gap benchmark

We construct and validate a benchmark containing questions grouped according to the degree of terminology mismatch between modern queries and the source text.

### 3. Controlled retrieval comparison

Concept-aware retrieval is compared against multiple baselines, including:

* BM25
* TF-IDF-based retrieval
* Dense retrieval
* Hybrid retrieval
* Machine-translated query retrieval
* Generic LLM query expansion
* Concept-aware query expansion

This allows us to determine whether improvements actually come from the proposed concept-mapping approach rather than from query expansion in general.

### 4. Grounded Generative AI

A Generative AI component produces answers from retrieved evidence while requiring source citations.

The system can also abstain when sufficient supporting evidence cannot be retrieved.

---

# Dataset

## Primary Corpus

The primary corpus is **Sūrya Siddhānta**, a classical Sanskrit astronomical text.

The Sanskrit source is maintained alongside an aligned English translation so that retrieved evidence can be connected to the original Sanskrit verses.

The project uses verse-level or verse-cluster-level retrieval units.

### Benchmark

A custom evaluation benchmark will be constructed containing approximately:

**150–300 research questions**

Each question will contain information such as:

```text
question
gold_passage
chapter
verse_id
concepts
terminology_gap_category
mapping_type
difficulty
```

Questions will be divided into terminology-gap categories such as:

```text
Low Gap
Medium Gap
High Gap
```

The categories will not rely solely on subjective labeling. Lexical/semantic overlap measurements will also be used to validate the difficulty stratification.

---

# Phase 1 — Glossary (complete)

| file | what it is |
|---|---|
| `data/glossary/glossary_v1.0_FROZEN.csv` | **The frozen glossary.** 57 entries: concept, Sanskrit term, variants, evidence verses + snippet, concept category, confidence (High/Medium/Low) with its basis, mapping type (+ secondary), attestation result, spot-check result |
| `data/glossary/glossary_agreement_log.md` | Verification & agreement log: 170/170 evidence citations attested; spot-check covers 18/57 entries (32%), 88.9% entry-level / 87.1% verse-level agreement; adjudicated disagreements; mapping-type review; v1.1 flags; P1/P2 sign-off |
| `data/glossary/glossary_v1.0_manifest.json` | SHA-256 of the frozen file and of every input |
| `data/glossary/ss_glossary_clean.csv` | Curated candidate glossary (terms, glosses, evidence). Never edited |
| `data/glossary/glossary_review_v1.csv` | Reviewed concept category / mapping type / confidence overrides, with rationale |
| `data/glossary/glossary_meta_DRAFT.csv`, `spotcheck_precheck.csv` | Rule-derived draft metadata and the reviewer's spot-check sheet |

Rebuild (deterministic; refuses to freeze if any evidence verse does not print its term):

```bash
pip install pandas numpy rank-bm25
python3 src/glossary/freeze_glossary.py              # -> glossary_v1.0_FROZEN.csv + agreement log + manifest
python3 src/glossary/build_leakage_table.py          # -> data/benchmark/ss_questions_leakage.csv
python3 experiments/glossary_expansion/smoke_test.py # BM25 ± glossary expansion under leakage control
```

Mapping-type typology: `direct_equivalence`, `one_to_many`, `many_to_one`, `componential`, `conceptual_restructuring`, `no_equivalent`.
Known issues are flagged for v1.1 rather than edited: the `aksha` variants `lamba`/`lambaka` mean co-latitude, and `yuga` = "epoch" at 1.56.

### App (portal)

```bash
cd app && pip install -r requirements.txt && ANTHROPIC_API_KEY=... streamlit run app.py
python3 tests.py   # generation/citation validation + abstention gate checks
```

Without a key, the app falls back to showing the top retrieved verse. To deploy to Hugging Face Spaces, push `app/` and add `ANTHROPIC_API_KEY` as a Space secret.
`retrieval-service/` is the earlier Phase 0 FastAPI search skeleton (40-verse initial corpus).

---

# System Architecture

## 1. Corpus Processing

The Sanskrit text and its English translation are:

* collected
* cleaned
* segmented
* aligned by verse
* assigned stable identifiers
* indexed for retrieval

Example:

```text
verse_id: SS_03_17

sanskrit:
...

translation:
...

chapter:
3

concepts:
[planetary_motion, longitude, ...]
```

---

## 2. Query Understanding

A user's modern English query is analyzed to identify the underlying astronomical concepts.

For example:

```text
Input:
"How is the apparent position of a planet determined?"

↓
Concepts:

planet
apparent position
planetary motion
position calculation
```

---

## 3. Concept-Aware Query Expansion

The identified concepts are mapped to terminology associated with the historical corpus.

The mapping layer contains information such as:

```text
Modern Concept
        ↓
Historical Concept
        ↓
Sanskrit Term
        ↓
Related Terms / Variants
        ↓
Evidence / Confidence
```

The purpose is not simply to translate the entire query.

Instead, the system attempts to identify **what the user means** and connect those concepts with terminology that is actually useful for retrieving historical passages.

---

## 4. Retrieval

Multiple retrieval strategies are evaluated.

### Lexical Retrieval

BM25 / TF-IDF based retrieval.

### Dense Retrieval

Sentence-transformer based semantic retrieval.

### Hybrid Retrieval

Combines lexical and dense retrieval.

### Concept-Aware Retrieval

The proposed approach expands the query using the terminology mapping layer before retrieval.

---

## 5. Reranking

Retrieved passages can optionally be reranked using a cross-encoder to determine which candidate passages are most relevant to the query.

---

## 6. Retrieval-Augmented Generation

The highest-ranked passages are provided as context to the Generative AI model.

The model is instructed to:

* answer using retrieved evidence
* avoid unsupported claims
* provide citations
* distinguish information from the source from modern interpretation
* abstain when sufficient evidence is unavailable

---

# Experimental Design

The central experiment compares retrieval performance across progressively larger terminology gaps.

### Example

| Condition         | Query Processing                        |
| ----------------- | --------------------------------------- |
| Baseline          | Original query                          |
| BM25              | Keyword retrieval                       |
| Dense             | Semantic retrieval                      |
| Hybrid            | BM25 + dense                            |
| MT Expansion      | Machine-translated terminology          |
| LLM Expansion     | Generic LLM expansion                   |
| **Concept-Aware** | **Domain-specific terminology mapping** |

The key analysis is not simply:

> "Did our method get the highest score?"

Instead, we investigate:

> **How does the advantage of concept-aware retrieval change as terminology mismatch increases?**

---

# Evaluation

## Retrieval Metrics

The retrieval systems will be evaluated using:

* Recall@K
* Mean Reciprocal Rank (MRR)
* nDCG@K
* Precision@K where appropriate

The main comparison will examine whether the proposed method retrieves the correct gold passage more frequently.

---

## Terminology-Gap Analysis

Results will be reported separately for:

```text
Low terminology gap
        ↓
Medium terminology gap
        ↓
High terminology gap
```

The expected research result is that concept-aware expansion provides its largest improvement in the high-gap category.

---

## Ablation Studies

To understand *why* the system works, different components will be removed or modified.

Examples:

```text
Full system
    ↓
Remove concept expansion
    ↓
Remove specific mapping type
    ↓
Use only high-confidence mappings
    ↓
Use generic LLM expansion
    ↓
Use machine translation
```

This helps determine whether the improvement comes from the proposed terminology knowledge itself or merely from adding more text to the query.

---

# Generative AI Component

Generative AI is used in multiple parts of the system.

### Research / Data Construction

A GenAI model assists with:

* generating candidate research questions
* identifying candidate astronomical concepts
* suggesting terminology mappings
* generating query variations

All generated material is manually verified against the source corpus before being included in the evaluation dataset.

### Final Answer Generation

The GenAI component receives:

```text
User Query
+
Retrieved Evidence
+
Source Metadata
```

and produces:

```text
Answer
+
Supporting Citation
+
Source Verse
```

The model is explicitly constrained to the retrieved evidence.

---

# Technology Stack

### NLP / Retrieval

* Python
* NLTK / spaCy
* BM25
* TF-IDF
* Sentence Transformers
* FAISS

### ML / DL

* Scikit-learn
* PyTorch
* Transformer-based encoders
* Cross-encoder reranking

### GenAI

* Claude API

### RAG

* Hybrid retrieval
* Reranking
* Evidence-grounded generation

### Application

* FastAPI / Streamlit
* Optional web interface

### Development

* Git
* GitHub
* Jupyter / Google Colab
* Hugging Face

---

# Project Structure

```text
sanskrit-astronomy-rag/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── benchmark/
│   └── glossary/
│
├── notebooks/
│   ├── 01_corpus_eda.ipynb
│   ├── 02_benchmark_analysis.ipynb
│   ├── 03_retrieval_baselines.ipynb
│   ├── 04_concept_aware_retrieval.ipynb
│   └── 05_evaluation.ipynb
│
├── src/
│   ├── preprocessing/
│   ├── glossary/
│   ├── retrieval/
│   ├── reranking/
│   ├── generation/
│   └── evaluation/
│
├── experiments/
│   ├── baselines/
│   ├── ablations/
│   └── results/
│
├── app/
│   └── ...
│
├── paper/
│   ├── figures/
│   ├── tables/
│   └── manuscript/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# Research Paper

The project is designed alongside a research paper rather than treating the paper as documentation written after implementation.

The paper will investigate:

### Research Question

> Does concept-aware query expansion improve retrieval for historical Sanskrit astronomical texts, particularly when modern queries exhibit a high terminology gap?

### Hypothesis

> Concept-aware query expansion will provide increasingly larger retrieval improvements as terminology mismatch between modern queries and historical source passages increases.

### Planned Paper Structure

```text
1. Abstract

2. Introduction
   ├── Historical scientific text retrieval
   ├── Vocabulary mismatch
   ├── Problem statement
   └── Contributions

3. Related Work
   ├── Sanskrit NLP
   ├── Historical text retrieval
   ├── Dense and hybrid retrieval
   ├── Query expansion
   └── RAG / grounded generation

4. Dataset and Benchmark
   ├── Sūrya Siddhānta corpus
   ├── Verse alignment
   ├── Question construction
   └── Terminology-gap validation

5. Methodology
   ├── Query understanding
   ├── Concept mapping
   ├── Query expansion
   ├── Retrieval
   └── Reranking

6. Experimental Setup
   ├── Baselines
   ├── Metrics
   └── Ablations

7. Results
   ├── Overall retrieval
   ├── Low-gap queries
   ├── Medium-gap queries
   ├── High-gap queries
   └── Ablation results

8. Analysis
   ├── When concept expansion helps
   ├── When it fails
   ├── Mapping-type analysis
   └── Error analysis

9. GenAI / RAG Evaluation
   ├── Groundedness
   ├── Citation correctness
   └── Abstention

10. Limitations

11. Conclusion

12. References
```

---

# Why This Is Not Just Another RAG Project

The goal is **not** to build:

> PDF → embeddings → vector database → Claude → chatbot

Instead, RAG is used as the experimental environment for investigating a specific retrieval problem.

The central contribution is the relationship between:

```text
Terminology Gap
       ↓
Retrieval Difficulty
       ↓
Effect of Concept-Aware Expansion
```

This allows the project to make a research claim about **when and why** terminology-aware retrieval works rather than simply reporting that one implementation achieved a higher score.

---

# Current Research Scope

### Core

* Sūrya Siddhānta
* English + Sanskrit aligned corpus
* terminology-gap benchmark
* concept-aware query expansion
* BM25 / dense / hybrid baselines
* retrieval evaluation
* ablation studies
* grounded GenAI generation

### Optional Extensions

* Sanskrit-side cross-lingual retrieval
* additional Sanskrit astronomical texts
* advanced reranking
* out-of-domain evaluation
* citation verification
* interactive web application

Optional extensions will not be allowed to compromise the core research experiments.

---

# Team

This project is being developed as a four-member research team.

| Role                 | Responsibility                                           |
| -------------------- | -------------------------------------------------------- |
| NLP / Research Lead  | Concept mapping, terminology analysis, literature review |
| Data Lead            | Corpus preparation, benchmark construction, EDA          |
| ML / DL Lead         | Retrieval models, reranking, experiments                 |
| GenAI / Systems Lead | RAG pipeline, generation, application                    |

All team members contribute to evaluation, analysis, documentation, and the research paper.

---

# Project Status

🚧 **Active Research Project**

Current stage:

* [ ] Literature review
* [x] Corpus collection
* [x] Corpus preprocessing (427 gold-eligible verses, `data/processed/`)
* [x] Terminology glossary — **Phase 1 complete: `glossary_v1.0_FROZEN.csv`** (pending P1/P2 sign-off, see agreement log)
* [ ] Benchmark construction (103 answerable + 12 abstain questions drafted; leakage table built)
* [ ] Baseline retrieval
* [ ] Concept-aware retrieval
* [ ] Ablation experiments
* [x] RAG integration (Streamlit app in `app/`)
* [ ] GenAI evaluation
* [ ] Final experiments
* [ ] Research paper
* [ ] Deployment

---

# Research Philosophy

This project prioritizes **measurable research findings over architectural complexity**.

A more complicated system is not automatically a better research contribution.

Every major component is therefore evaluated against the question:

> **Does this component help us understand or solve the terminology-gap retrieval problem?**

---

## Citation

If you use this work, please cite:

```bibtex
@misc{sanskrit_astronomy_rag,
  title  = {Concept-Aware Query Expansion for Terminology-Gap Retrieval in Historical Sanskrit Astronomical Texts},
  author = {Research Team},
  year   = {2026}
}
```

---

## License

This project is intended for academic and research purposes.
