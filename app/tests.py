import json, types, pandas as pd
from core.data import *; from core.retrieval import Retriever; from core.pipeline import answer
from core.generate import parse_json, validate
R = Retriever(load_corpus(), load_glossary()); Q = load_questions()
def fake(payload):
    class M:  # mimics anthropic client
        class messages:
            @staticmethod
            def create(**kw): return types.SimpleNamespace(content=[types.SimpleNamespace(type='text', text=payload)])
    return M
def run(payload, q='Where is the city of Lanka located?'):
    o = answer(q, R, 'proposed', client=fake(payload)); return o['decision'], o['answer'] and o['answer'].get('citation_warnings')
good = lambda cites, ok=True: json.dumps(dict(evidence_sufficient=ok, what_text_states='x', translation_note='', modern_context='', citations=cites))
top = R.search('Where is the city of Lanka located?', 'proposed')['results'][0]['verse_id']
print('valid cite     ->', run(good([top])))
print('fenced json    ->', run('```json\n' + good([top]) + '\n```'))
print('fake citation  ->', run(good(['99.99'])))
print('mixed cites    ->', run(good([top, '99.99'])))
print('model abstains ->', run(good([], ok=False)))
print('non-JSON       ->', run('Sure! The answer is...'))
# end-to-end over the benchmark with an "oracle-ish" fake that cites the top verse (tests the gate only)
dec = []
for r in Q.itertuples():
    o = answer(r.question, R, 'proposed', client=fake(good([R.search(r.question)['results'][0]['verse_id']])))
    dec.append((r.type, o['decision']))
print(pd.crosstab(pd.Series([d[0] for d in dec], name='truth'), pd.Series([d[1] for d in dec], name='gate+fake-LLM')))
