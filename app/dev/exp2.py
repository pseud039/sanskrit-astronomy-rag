import sys, numpy as np, pandas as pd
import os
P4 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [P4, os.path.join(os.path.dirname(P4), 'src', 'evaluation')]
from core.data import *; from core.retrieval import Retriever, tok
C=load_corpus(); G=load_glossary(); Q=load_questions(); R=Retriever(C,G)
def feats(q):
    s=R.bm.get_scores(tok(q)); o=np.sort(s)[::-1]
    qt=[t for t in set(tok(q)) if t in R.bm.idf and R.bm.idf[t]>0]
    maxs=sum(R.bm.idf[t] for t in qt) or 1
    return dict(top1=o[0],margin=o[0]-o[1],norm=o[0]/maxs)
F=pd.DataFrame([feats(q) for q in Q.question]); F['y']=(Q.type=='answerable').astype(int).values
def auroc(x,y):
    pos,neg=x[y==1],x[y==0]; return np.mean([(p>n)+0.5*(p==n) for p in pos for n in neg])
for f in ['top1','margin','norm']: print(f,'AUROC',round(auroc(F[f].values,F.y.values),3), '| abstain mean',round(F[F.y==0][f].mean(),2),'answerable mean',round(F[F.y==1][f].mean(),2))
print(Q[Q.type=='abstain'].question.tolist())
F.to_csv(os.path.join(P4, 'dev', 'feats.csv'),index=False)
# threshold sweep on top1
for f in ['top1','norm']:
    print(f)
    for t in np.quantile(F[f],[.05,.1,.15,.2,.25]):
        pred=(F[f]>=t); tp=(pred&(F.y==1)).sum(); fp=(pred&(F.y==0)).sum(); 
        print(f'  thr {t:.2f}: answers {pred.sum()}/115; abstain-recall {((~pred)&(F.y==0)).sum()}/12; wrongly-abstained {((~pred)&(F.y==1)).sum()}/103')
