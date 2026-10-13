import os, sys
_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(_R, 'src', 'glossary'), os.path.join(_R, 'src', 'evaluation')]
import sys, numpy as np, pandas as pd, re
from rank_bm25 import BM25Okapi
from leakage_utils import *; from eval_utils import *
G, C, Q = load_all(*sys.argv[1:2])
Cc = C[C.role == 'gold_eligible']; ids = list(Cc.index)
tok = lambda s: re.findall(r'[a-z]+', re.sub(r'[\u0002]|_x0002_', '', s.lower()))
bm = BM25Okapi([tok(t) for t in Cc.english])
def run(expand_mode):
    ranks = []
    for r in Q.itertuples():
        qt = tok(r.question)
        if expand_mode:
            Ga = allowed_glossary(r.qid, G, Q, expand_mode, C)
            for e in match_entries(r.question, Ga):
                row = G[G.sanskrit_term == e].iloc[0]; qt += tok(' '.join(forms(row)))
        s = bm.get_scores(qt); order = [ids[i] for i in np.argsort(-s)]; ranks.append(rank_of_gold(order, r.gold_verse))
    return per_query_metrics(ranks)
masks = subset_masks(Q, G, C)
base = run(None)
print('--- BM25 no expansion'); print(report(base, masks).to_string(index=False))
for m in ('all', 'strict', 'stricter'):
    ex = run(m); print(f'--- BM25 + glossary expansion (leakage mode={m})'); print(report(ex, masks).to_string(index=False))
    print('paired MRR diff vs baseline on ALL: mean %.3f CI [%.3f,%.3f] p=%.3f' % paired_bootstrap(ex.MRR, base.MRR))
