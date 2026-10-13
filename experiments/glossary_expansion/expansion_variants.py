"""BM25 query-expansion variants under leakage control. EXPLORATORY: 3 variants x 3 weights = 9 comparisons per mode; report all, tune none.
Usage: python3 experiments/glossary_expansion/expansion_variants.py [data-root]"""
import os, sys
_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(_R, 'src', 'glossary'), os.path.join(_R, 'src', 'evaluation')]
import sys, re, numpy as np, pandas as pd
from rank_bm25 import BM25Okapi
from leakage_utils import *; from eval_utils import *
G, C, Q = load_all(*sys.argv[1:2]); L = pd.read_csv(os.path.join(DATA, 'benchmark', 'ss_questions_leakage.csv'), dtype=str)
tok = lambda s: re.findall(r'[a-z]+', s.lower()); ids = list(C.index); bm = BM25Okapi([tok(t) for t in C.english])
gl = G.set_index('sanskrit_term')
def exp_terms(entries, kind):
    out = []
    for e in entries:
        r = gl.loc[e]
        if kind in ('sanskrit', 'both'): out += [e] + [v for v in r.variants.split('|') if v]
        if kind in ('english', 'both'): out += [r.english_gloss] + [x for x in r.other_glosses.split('|') if x]
    return ' '.join(out)
def run(mode, kind, w):
    ranks = []
    for r in Q.itertuples():
        s = bm.get_scores(tok(r.question))
        if kind:
            allowed = allowed_glossary(r.qid, G, Q, mode, C); hit = match_entries(r.question, allowed)
            if hit: s = s + w * bm.get_scores(tok(exp_terms(hit, kind)))
        ranks.append(rank_of_gold([ids[i] for i in np.argsort(-s)], r.gold_verse))
    return per_query_metrics(ranks)
base = run('all', None, 0)
print(f'BASELINE BM25  MRR {base.MRR.mean():.3f}  R@5 {base["R@5"].mean():.3f}  R@10 {base["R@10"].mean():.3f}\n')
rows = []
for mode in ('all', 'strict', 'stricter'):
    app = (L['n_applicable_' + mode if mode != 'all' else 'n_applicable_all'].astype(int) > 0).values
    for kind in ('sanskrit', 'english', 'both'):
        for w in (0.25, 0.5, 1.0):
            m = run(mode, kind, w)
            d_all = paired_bootstrap(m.MRR, base.MRR); d_app = paired_bootstrap(m.MRR[app], base.MRR[app])
            rows.append(dict(leak_control=mode, variant=kind, w=w, dMRR_ALL=f'{d_all[0]:+.3f} [{d_all[1]:+.3f},{d_all[2]:+.3f}]',
                n_app=int(app.sum()), dMRR_applicable=f'{d_app[0]:+.3f} [{d_app[1]:+.3f},{d_app[2]:+.3f}] p={d_app[3]:.2f}',
                dR5_applicable=f'{(m["R@5"][app] - base["R@5"][app]).mean():+.3f}'))
T = pd.DataFrame(rows); T.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'expansion_variants_results.csv'), index=False); print(T.to_string(index=False))
