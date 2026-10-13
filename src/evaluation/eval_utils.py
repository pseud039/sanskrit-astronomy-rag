"""Metrics + uncertainty for a single-gold-verse benchmark (n≈103). Pure numpy/pandas."""
import numpy as np, pandas as pd

def rank_of_gold(ranked_ids, gold):
    try: return ranked_ids.index(gold) + 1
    except ValueError: return np.inf

def per_query_metrics(ranks, ks=(5, 10)):
    """ranks: array of 1-based gold ranks (inf if not retrieved). One gold per query => nDCG = 1/log2(rank+1)."""
    r = np.asarray(ranks, float); out = {}
    for k in ks:
        out[f'R@{k}'] = (r <= k).astype(float)
        out[f'nDCG@{k}'] = np.where(r <= k, 1 / np.log2(r + 1), 0.0)
    out['MRR'] = np.where(np.isfinite(r), 1 / r, 0.0)
    return pd.DataFrame(out)

def bootstrap_ci(x, B=10000, seed=0, alpha=0.05):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return x.mean(), np.quantile(means, alpha / 2), np.quantile(means, 1 - alpha / 2)

def paired_bootstrap(a, b, B=10000, seed=0):
    """Mean difference (a-b) with 95% CI and two-sided bootstrap p (sign-flip permutation)."""
    d = np.asarray(a, float) - np.asarray(b, float); rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    flips = (d * rng.choice([-1, 1], (B, len(d)))).mean(1)
    p = (np.sum(np.abs(flips) >= abs(d.mean())) + 1) / (B + 1)
    return d.mean(), np.quantile(boots, .025), np.quantile(boots, .975), p

def report(metrics_df, masks, cols=('R@5', 'R@10', 'MRR', 'nDCG@10')):
    """Table of mean [95% CI] per metric for every subset mask (ALL / CLEAN_* / OVERLAP_*)."""
    rows = []
    for name, m in masks.items():
        m = np.asarray(m, bool)
        row = {'subset': name, 'n': int(m.sum())}
        for c in cols:
            mu, lo, hi = bootstrap_ci(metrics_df.loc[m, c].values)
            row[c] = f'{mu:.3f} [{lo:.3f},{hi:.3f}]'
        rows.append(row)
    return pd.DataFrame(rows)

def grouped_folds(gold_verses, k=5, seed=0):
    """Chapter-grouped k-fold (no chapter appears in both train and test) for TRAINED models only.
    Training-free retrievers (BM25/dense/expansion) are evaluated on all queries; no split needed."""
    ch = pd.Series([v.split('.')[0] for v in gold_verses]); rng = np.random.default_rng(seed)
    chapters = rng.permutation(ch.unique()); fold_of = {c: i % k for i, c in enumerate(chapters)}
    f = ch.map(fold_of).values
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]
