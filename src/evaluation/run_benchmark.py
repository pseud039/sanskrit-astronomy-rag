"""Retrieval benchmark: BM25 vs concept-aware expansion (+ dense / rerank when sentence-transformers is installed).
Protocol (frozen before running):
  * split   : data/benchmark/splits_v1.csv (chapter-grouped dev/test). Tuning uses DEV only; TEST is reported.
  * headline: TEST answerable questions NOT seen by the lexicon author (seen_by_lexicon_author == False).
  * leakage : glossary entries citing the gold verse are excluded per question ('strict' mode, leakage_utils),
              for both the glossary and the lexicon's glossary links.
  * tuned on dev: expansion weight w in {0.25, 0.5, 1.0}; abstention threshold on the gate score.
Writes experiments/results/benchmark_v1.{csv,md} and app/data/gate.json.
Usage: python3 src/evaluation/run_benchmark.py [--dense] [--rerank]"""
import os, sys, json, argparse, numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(REPO, 'app'), os.path.join(REPO, 'src', 'evaluation')]
from core.data import load_corpus, load_glossary, load_lexicon
from core.retrieval import Retriever
from eval_utils import rank_of_gold, per_query_metrics, bootstrap_ci, paired_bootstrap

OUT = os.path.join(REPO, 'experiments', 'results')
B = os.path.join(REPO, 'data', 'benchmark')
MIN_GATE_BAL_ACC = 0.7   # a gate weaker than this on DEV does more harm (refusing answerable questions) than good


def ranks(R, Q, method, rerank=False, **kw):
    out, hit = [], []
    for r in Q.itertuples():
        ex = set(r.exclude_entries_strict.split('|')) - {''}
        if rerank:
            o = R.search(r.question, method, k=0, exclude=ex, rerank=True, **kw)
            order, nh = [x['verse_id'] for x in o['results']], len(o['matched'])
        else:
            s, h, _ = R.scores(r.question, method, ex, **kw)
            order, nh = [R.ids[i] for i in np.argsort(-s)], len(h)
        out.append(rank_of_gold(order, r.gold_verse)); hit.append(nh > 0)
    m = per_query_metrics(out); m['concept_hit'] = np.array(hit, float)
    return m


def fmt(x):
    mu, lo, hi = bootstrap_ci(x); return f'{mu:.3f} [{lo:.3f},{hi:.3f}]'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dense', action='store_true'); ap.add_argument('--rerank', action='store_true')
    a = ap.parse_args()
    C, G, L = load_corpus(), load_glossary(), load_lexicon()
    Qall = pd.read_csv(f'{B}/ss_questions_all.csv', dtype=str)
    S = pd.read_csv(f'{B}/splits_v1.csv', dtype=str)
    LK = pd.read_csv(f'{B}/ss_questions_leakage.csv', dtype=str).fillna('')
    Q = Qall.merge(S[['qid', 'split', 'seen_by_lexicon_author']], on='qid')
    A = Q[Q.type == 'answerable'].merge(LK[['qid', 'exclude_entries_strict']], on='qid').reset_index(drop=True)
    dense = corpus_emb = reranker = None
    if a.dense or a.rerank:
        from core.models import load_dense, load_reranker
        dense, corpus_emb, msg = load_dense(); print(msg)
        if a.rerank: reranker, msg = load_reranker(); print(msg)
    dev = (A.split == 'dev').values; test = (A.split == 'test').values
    unseen = test & (A.seen_by_lexicon_author != 'True').values

    # ---- tune w on DEV (proposed, BM25 side only so the choice does not depend on optional models)
    tune = []
    for w in (0.25, 0.5, 1.0):
        R = Retriever(C, G, L, w=w)
        tune.append((w, ranks(R, A[dev], 'proposed').MRR.mean()))
    w_best = max(tune, key=lambda t: t[1])[0]
    print('dev tuning (w, MRR):', [(w, round(m, 3)) for w, m in tune], '-> w =', w_best)
    R = Retriever(C, G, L, w=w_best, dense=dense, corpus_emb=corpus_emb, reranker=reranker)

    conds = [('BM25', 'bm25', {}), ('Glossary expansion', 'proposed', dict(use_lexicon=False)),
             ('Lexicon expansion', 'proposed', dict(use_glossary=False)), ('Proposed (glossary + lexicon)', 'proposed', {})]
    if R.has_dense: conds += [('Hybrid BM25+dense', 'hybrid', {}), ('Proposed + dense', 'proposed', {})]
    res = {}
    for name, method, kw in conds:
        if name == 'Proposed + dense': res[name] = ranks(R, A, 'proposed')
        elif R.has_dense and method == 'proposed':  # BM25-side expansion rows stay lexical-only for comparability
            Rl = Retriever(C, G, L, w=w_best); res[name] = ranks(Rl, A, method, **kw)
        else: res[name] = ranks(R, A, method, **kw)
        if reranker is not None and name in ('BM25', 'Proposed (glossary + lexicon)', 'Proposed + dense', 'Hybrid BM25+dense'):
            res[name + ' + rerank'] = ranks(R if name != 'Proposed (glossary + lexicon)' else Retriever(C, G, L, w=w_best, reranker=reranker),
                                            A, method, rerank=True, **kw)

    rows = []
    for subset, mask in (('TEST unseen (headline)', unseen), ('TEST all', test), ('DEV', dev), ('ALL', np.ones(len(A), bool))):
        for name, m in res.items():
            mm = m[mask]
            rows.append(dict(subset=subset, n=int(mask.sum()), method=name, **{c: fmt(mm[c]) for c in ('R@5', 'R@10', 'MRR', 'nDCG@10')},
                             concept_hit_rate=f"{mm.concept_hit.mean():.0%}"))
    T = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True); T.to_csv(f'{OUT}/benchmark_v1.csv', index=False)

    # ---- paired comparison vs BM25 on the headline subset
    cmp = []
    for name, m in res.items():
        if name == 'BM25': continue
        d = paired_bootstrap(m.MRR[unseen], res['BM25'].MRR[unseen])
        dh = paired_bootstrap(m.MRR[unseen & (m.concept_hit.values > 0)], res['BM25'].MRR[unseen & (m.concept_hit.values > 0)]) \
            if (unseen & (m.concept_hit.values > 0)).sum() > 1 else (np.nan,) * 4
        cmp.append(dict(method=name, dMRR_vs_BM25=f'{d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] p={d[3]:.2f}',
                        n_with_concept=int((unseen & (m.concept_hit.values > 0)).sum()),
                        dMRR_on_concept_hit=f'{dh[0]:+.3f} [{dh[1]:+.3f},{dh[2]:+.3f}] p={dh[3]:.2f}'))
    CMP = pd.DataFrame(cmp)

    # ---- abstention gate: pick score + threshold on DEV (answerable vs abstain), report on TEST
    G_ = {r.qid: R.search(r.question, 'proposed', k=1, rerank=reranker is not None)['gate'] for r in Q.itertuples()}
    Q['y'] = (Q.type == 'answerable').astype(int)
    def bal(sc, t, D): p = D[sc] >= t; return 0.5 * ((p & (D.y == 1)).sum() / (D.y == 1).sum() + (~p & (D.y == 0)).sum() / (D.y == 0).sum())
    grows = []
    for sc in G_[Q.qid.iloc[0]]:
        Q[sc] = Q.qid.map(lambda q: G_[q][sc]); d, t_ = Q[Q.split == 'dev'], Q[Q.split == 'test']
        thr = float(max(np.unique(d[sc]), key=lambda t: (bal(sc, t, d), -t)))
        p = t_[sc] >= thr
        grows.append(dict(score=sc, dev_threshold=round(thr, 3), dev_bal_acc=round(bal(sc, thr, d), 3), test_bal_acc=round(bal(sc, thr, t_), 3),
                          test_abstain_refused=f"{int((~p & (t_.y == 0)).sum())}/{int((t_.y == 0).sum())}",
                          test_answerable_refused=f"{int((~p & (t_.y == 1)).sum())}/{int((t_.y == 1).sum())}"))
    GT = pd.DataFrame(grows); best = GT.sort_values('dev_bal_acc', ascending=False).iloc[0]
    deploy = best.dev_bal_acc >= MIN_GATE_BAL_ACC
    gate = dict(score=best.score, threshold=float(best.dev_threshold)) if deploy else dict(score=None)
    gate_txt = (GT.to_markdown(index=False) + '\n\n' +
                (f'**Deployed:** `{best.score}` ≥ {best.dev_threshold} (selected by DEV balanced accuracy; TEST shown for reporting only).'
                 if deploy else f'**No gate deployed** (best DEV balanced accuracy {best.dev_bal_acc} < {MIN_GATE_BAL_ACC}); '
                 'abstention is left to the model ("The text does not answer this").'))
    json.dump(gate, open(os.path.join(REPO, 'app', 'data', 'gate.json'), 'w'), indent=1)

    md = ['# Retrieval benchmark v1\n',
          f'Generated by `src/evaluation/run_benchmark.py`. Corpus: {len(C)} verses. Answerable questions: {len(A)} '
          f'(dev {int(dev.sum())}, test {int(test.sum())}, test unseen {int(unseen.sum())}). '
          f'Expansion weight tuned on DEV: w = {w_best} (dev MRR by w: {", ".join(f"{w}: {m:.3f}" for w, m in tune)}). '
          'Leakage control: glossary entries citing the gold verse are excluded per question. 95% bootstrap CIs.\n',
          f'Dense / rerank: {"on" if R.has_dense else "off (sentence-transformers or model not available)"} / '
          f'{"on" if reranker is not None else "off"}.\n',
          '## Results\n', T.to_markdown(index=False), '\n## Paired difference vs BM25 (TEST unseen)\n', CMP.to_markdown(index=False),
          '\n`dMRR_on_concept_hit` restricts to questions where the method detected at least one concept; on the rest '
          'concept expansion adds nothing, so it equals BM25.\n', '## Abstention gate\n', gate_txt, '']
    open(f'{OUT}/benchmark_v1.md', 'w').write('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
