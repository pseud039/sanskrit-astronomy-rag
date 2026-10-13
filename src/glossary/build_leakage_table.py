import sys, pandas as pd
from leakage_utils import *
G, C, Q = load_all(*sys.argv[1:2]); rows = []
for r in Q.itertuples():
    ex_s = exclude_entries(r.gold_verse, G, C, 'strict'); ex_x = exclude_entries(r.gold_verse, G, C, 'stricter')
    ex_l = exclude_entries(r.gold_verse, G, C, 'stricter', 'left'); ex_w = exclude_entries(r.gold_verse, G, C, 'stricter', 'wb')
    m = match_entries(r.question, G)
    rows.append(dict(qid=r.qid, gold_verse=r.gold_verse, chapter=r.gold_verse.split('.')[0],
        split_strict='OVERLAP' if ex_s else 'CLEAN', split_stricter='OVERLAP' if ex_x else 'CLEAN',
        split_stricter_left='OVERLAP' if ex_l else 'CLEAN', split_stricter_wb='OVERLAP' if ex_w else 'CLEAN',
        exclude_entries_strict='|'.join(sorted(ex_s)), exclude_entries_stricter='|'.join(sorted(ex_x)),
        entries_matched_all='|'.join(m), n_applicable_all=len(m),
        n_applicable_strict=len([e for e in m if e not in ex_s]), n_applicable_stricter=len([e for e in m if e not in ex_x])))
df = pd.DataFrame(rows); df.to_csv(os.path.join(DATA, 'benchmark', 'ss_questions_leakage.csv'), index=False)
for c in ['split_strict', 'split_stricter', 'split_stricter_left', 'split_stricter_wb']: print(c, df[c].value_counts().to_dict())
print('Q011:', df[df.qid == 'Q011'].exclude_entries_stricter.iloc[0], '| Q028:', df[df.qid == 'Q028'].exclude_entries_stricter.iloc[0] or '(none)')
print('>=1 applicable entry: ALL', (df.n_applicable_all > 0).sum(), '| strict', (df.n_applicable_strict > 0).sum(), '| stricter', (df.n_applicable_stricter > 0).sum())
for t in ('strict', 'stricter'):
    cl = df[df['split_' + t] == 'CLEAN']; print(f'CLEAN_{t}: n={len(cl)}, applicable={(cl["n_applicable_" + t] > 0).sum()}')
