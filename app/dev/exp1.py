import sys, numpy as np, pandas as pd
import os
P4 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [P4, os.path.join(os.path.dirname(P4), 'src', 'evaluation')]
from core.data import *; from core.retrieval import Retriever
from eval_utils import *
C=load_corpus(); G=load_glossary(); Q=load_questions(); L=pd.read_csv(f'{D}/ss_questions_leakage.csv',dtype=str).fillna('')
A=Q[Q.type=='answerable'].merge(L[['qid','exclude_entries_strict']],on='qid')
for w in [0,0.1,0.25,0.5,1.0]:
    R=Retriever(C,G,w=w); out={}
    for mode,exk in [('baseline',False),('LOO',True)]:
        rk=[]
        for r in A.itertuples():
            ex=set(r.exclude_entries_strict.split('|'))-{''} if exk else ()
            o=R.search(r.question,'baseline' if mode=='baseline' else 'proposed',k=len(C),exclude=ex)
            ids=[x['verse_id'] for x in o['results']]; rk.append(rank_of_gold(ids,r.gold_verse))
        out[mode]=per_query_metrics(rk)
    d=paired_bootstrap(out['LOO'].MRR,out['baseline'].MRR)
    print(f"w={w}: baseline MRR {out['baseline'].MRR.mean():.3f} | proposed(LOO) MRR {out['LOO'].MRR.mean():.3f} diff {d[0]:+.3f} CI[{d[1]:+.3f},{d[2]:+.3f}] p={d[3]:.3f}  R@5 {out['LOO']['R@5'].mean():.3f}")
