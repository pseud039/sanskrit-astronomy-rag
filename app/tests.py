"""Offline tests: generation/citation validation and error surfacing against a local fake LLM server that speaks both
the Ollama (/api/chat) and OpenAI-compatible (/chat/completions) protocols. Run: python3 tests.py"""
import json, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from core.data import load_corpus, load_glossary, load_lexicon
from core.retrieval import Retriever
from core.pipeline import answer

R = Retriever(load_corpus(), load_glossary(), load_lexicon())
Q = 'Where is the city of Lanka located?'
TOP = R.search(Q, 'proposed')['results']
V0 = TOP[0]['verse_id']; QUOTE0 = ' '.join(TOP[0]['english'].split()[:6])
STATE = dict(reply='', status=200, last=None)


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        STATE['last'] = (self.path, body, self.headers.get('Authorization'))
        if STATE['status'] != 200:
            self.send_response(STATE['status']); self.end_headers(); self.wfile.write(b'{"error":"invalid api key"}'); return
        out = {'message': {'role': 'assistant', 'content': STATE['reply']}} if self.path == '/api/chat' else \
            {'choices': [{'message': {'role': 'assistant', 'content': STATE['reply']}}]}
        self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers()
        self.wfile.write(json.dumps(out).encode())


srv = HTTPServer(('127.0.0.1', 0), Fake); threading.Thread(target=srv.serve_forever, daemon=True).start()
BASE = f'http://127.0.0.1:{srv.server_port}'
cfg = lambda p, base=BASE: dict(provider=p, base_url=base, model='test-model', api_key='k' if p == 'openai' else '', timeout=5, error='')
reply = lambda **kw: json.dumps({**dict(answerable=True, answer=f'Lanka is on the equator [{V0}].',
                                        evidence=[dict(verse_id=V0, quote=QUOTE0)], confidence='high',
                                        closest_verses=[], modern_context=''), **kw})
fails = 0


def check(name, got, want):
    global fails
    ok = got == want; fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {got}" + ('' if ok else f'  (expected {want})'))


def run(text, provider='openai', status=200, c=None):
    STATE.update(reply=text, status=status)
    return answer(Q, R, 'proposed', cfg=c or cfg(provider))


o = run(reply()); check('valid answer (openai)', (o['decision'], o['answer']['citations'], o['answer']['evidence'][0]['verified']), ('answer', [V0], True))
check('openai request shape', (STATE['last'][0], STATE['last'][1]['response_format'], STATE['last'][2]), ('/chat/completions', {'type': 'json_object'}, 'Bearer k'))
o = run(reply(), 'ollama'); check('valid answer (ollama)', o['decision'], 'answer')
check('ollama sends JSON schema', ('format' in STATE['last'][1], STATE['last'][1]['stream']), (True, False))
o = run('```json\n' + reply() + '\n```'); check('fenced JSON', o['decision'], 'answer')
o = run(reply(answer='Lanka is far away [99.99].', evidence=[])); check('fake citation -> not answered', o['decision'], 'abstain')
o = run(reply(answer=f'Lanka [{V0}] and more [99.99].')); check('mixed citations keep the valid one', (o['decision'], o['answer']['citations']), ('answer', [V0]))
o = run(reply(evidence=[dict(verse_id=V0, quote='words that are not in the verse')])); check('unverified quote flagged', o['answer']['evidence'][0]['verified'], False)
o = run(reply(answerable=False, answer='The text does not answer this.', evidence=[], closest_verses=[V0])); check('model abstains', (o['decision'], o['answer']['closest_verses']), ('abstain', [V0]))
o = run('Sure! Lanka is somewhere.'); check('non-JSON -> visible error', o['decision'], 'error')
o = run(reply(), status=401); check('HTTP 401 -> visible error with status', (o['decision'], 'HTTP 401' in o['reason']), ('error', True))
o = run(reply(), c=cfg('openai', 'http://127.0.0.1:9')); check('unreachable server -> visible error', (o['decision'], 'cannot reach' in o['reason']), ('error', True))
o = answer(Q, R, 'proposed', cfg=dict(provider='none', error='')); check('no provider -> explicit no_llm', o['decision'], 'no_llm')

# dense / rerank wiring with stand-in models (real models need sentence-transformers + Hugging Face access)
import numpy as np
class FakeDense:
    def encode(self, texts, normalize_embeddings=True):
        v = np.array([[hash(w) % 7 for w in (t.lower().split() + ['x'] * 8)[:8]] for t in texts], float) + 1
        return v / np.linalg.norm(v, axis=1, keepdims=True)
class FakeRerank:
    def predict(self, pairs): return np.array([len(set(q.lower().split()) & set(d.lower().split())) for q, d in pairs], float)
Rd = Retriever(load_corpus(), load_glossary(), load_lexicon(), dense=FakeDense(), reranker=FakeRerank())
check('hybrid available with dense', Rd.available_methods(), ['bm25', 'hybrid', 'proposed'])
o = Rd.search(Q, 'hybrid', k=5, rerank=True)
check('rerank gate score present', (o['reranked'], sorted(o['gate']), len(o['results'])), (True, ['bm25_norm', 'bm25_top1', 'rerank_top1'], 5))
o = answer('How many moons does Jupiter have?', R, 'proposed', gate=dict(score='bm25_top1', threshold=17.25), cfg=dict(provider='none', error=''))
check('retrieval gate refuses off-text question', o['decision'], 'abstain')
o = answer(Q, R, 'proposed', gate=dict(score='rerank_top1', threshold=99), cfg=dict(provider='none', error=''))
check('gate needing an unloaded reranker is skipped', o['decision'], 'no_llm')
try: R.search(Q, 'hybrid'); check('hybrid without dense raises', False, True)
except ValueError: check('hybrid without dense raises', True, True)
srv.shutdown()
print('\nall passed' if not fails else f'\n{fails} FAILED'); raise SystemExit(1 if fails else 0)
