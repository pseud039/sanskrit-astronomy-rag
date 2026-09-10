
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util


# =========================
# LOAD DATA
# =========================

df = pd.read_csv("data/surya_siddhanta_initial_corpus.csv")

# =========================
# BM25 SETUP
# =========================

tokenized_documents = [
    text.lower().split()
    for text in df["English"].astype(str)
]

bm25 = BM25Okapi(tokenized_documents)


# =========================
# LOAD EMBEDDING MODEL
# =========================

model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# =========================
# CREATE ENGLISH EMBEDDINGS
# =========================

english_texts = df["English"].astype(str).tolist()

english_embeddings = model.encode(
    english_texts,
    convert_to_tensor=True
)


# =========================
# BM25 SEARCH
# =========================

def bm25_search(query, top_k=5):

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(
        tokenized_query
    )

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    results = df.iloc[top_indices][
        ["ID", "Chapter", "Verse", "English", "Sanskrit"]
    ].copy()

    results["BM25_Score"] = [
        scores[i]
        for i in top_indices
    ]

    return results


# =========================
# SEMANTIC SEARCH
# =========================

def semantic_search(query, top_k=5):

    query_embedding = model.encode(
        query,
        convert_to_tensor=True
    )

    scores = util.cos_sim(
        query_embedding,
        english_embeddings
    )[0]

    top_indices = torch.argsort(
        scores,
        descending=True
    )[:top_k]

    results = df.iloc[
        top_indices.cpu().numpy()
    ][
        ["ID", "Chapter", "Verse", "English", "Sanskrit"]
    ].copy()

    results["Semantic_Score"] = (
        scores[top_indices]
        .cpu()
        .numpy()
    )

    return results
