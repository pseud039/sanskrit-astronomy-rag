from .generate import generate

DEFAULT_THRESHOLD = 13.75   # BM25 top-1 gate; tuned on 103 answerable + 12 abstain questions (indicative only)

def answer(question, retriever, mode='proposed', k=5, threshold=DEFAULT_THRESHOLD, include_modern=False, model=None, client=None, exclude=()):
    r = retriever.search(question, mode=mode, k=k, exclude=exclude)
    top1 = r['top_scores'][0] if r['top_scores'] else 0.0
    if top1 < threshold:
        return dict(retrieval=r, decision='abstain', reason=f'retrieval gate: top BM25 score {top1:.1f} < {threshold}', answer=None)
    out = generate(question, r['results'], r['matched'], include_modern, model, client)
    if not out['evidence_sufficient']:
        return dict(retrieval=r, decision='abstain', reason='model judged the retrieved verses insufficient', answer=out)
    return dict(retrieval=r, decision='answer', reason='', answer=out)
