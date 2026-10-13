import os, json, logging, time
from .generate import generate
from .llm import LLMError, config, describe

log = logging.getLogger('ss-rag')
GATE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'gate.json')


def load_gate():
    """{"score": <gate score name> | null, "threshold": float}, chosen on the DEV split (src/evaluation/run_benchmark.py).
    A null score means no retrieval gate: abstention is left to the model."""
    try:
        return json.load(open(GATE_FILE))
    except (OSError, ValueError):
        return {'score': None}


def answer(question, retriever, method='proposed', k=5, include_modern=False, rerank=False, gate=None,
           cfg=None, model=None, exclude=()):
    """decision: answer | abstain (retrieval gate or model) | no_llm (generation not configured) | error (LLM failed)."""
    r = retriever.search(question, method=method, k=k, exclude=exclude, rerank=rerank)
    g = gate or load_gate(); name = g.get('score')
    base = dict(retrieval=r, answer=None)
    if name in r['gate'] and r['gate'][name] < g['threshold']:  # a gate needing an unloaded model (reranker) is skipped
        return dict(base, decision='abstain',
                    reason=f"retrieval evidence too weak ({name} {r['gate'][name]:.2f} < {g['threshold']:.2f})")
    cfg = cfg or config()
    if cfg['provider'] == 'none':
        return dict(base, decision='no_llm', reason=f'Answer generation is off: {describe(cfg)}.')
    t0 = time.time()
    try:
        out = generate(question, r['results'], r['matched'], include_modern, cfg, model)
    except LLMError as e:
        log.error('generation failed (%s): %s', describe(cfg), e)
        return dict(base, decision='error', reason=f'Generation failed ({describe(cfg)}): {e}')
    log.info('generated in %.1fs via %s; answerable=%s warnings=%s', time.time() - t0, describe(cfg),
             out['answerable'], out['warnings'])
    if not out['answerable']:
        return dict(base, answer=out, decision='abstain', reason='the model judged the retrieved verses insufficient')
    return dict(base, answer=out, decision='answer', reason='')
