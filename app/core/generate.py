"""Grounded answer generation. The model may use ONLY the retrieved verses; the output is JSON, and every citation and
quoted snippet is checked in code against the verses that were actually supplied."""
import json, re
from .llm import chat_json, LLMError

SYSTEM = """You answer questions about the Surya Siddhanta using ONLY the numbered verses provided (Burgess's 1860 English translation; Sanskrit terms appear in parentheses; there is some OCR noise).
1. Answer the question directly in 1-4 plain sentences.
2. Cite verse numbers like [5.8] after every claim. Cite only verse numbers that appear in the provided verses.
3. If the verses do not answer the question, set "answerable" to false, set "answer" to "The text does not answer this.", and list the closest verses in "closest_verses".
4. Use no outside knowledge in "answer". Only if modern context is enabled, add at most two sentences of modern astronomy in "modern_context", clearly separate from the answer; otherwise leave it empty.
5. "evidence": short verbatim quotes (at most 20 words each) copied exactly from the verses that support the answer.
6. "confidence": high, medium or low - how directly the verses answer the question.
Return ONLY one JSON object with the keys: answerable, answer, evidence, confidence, closest_verses, modern_context."""

SCHEMA = {
    'type': 'object',
    'properties': {
        'answerable': {'type': 'boolean'},
        'answer': {'type': 'string'},
        'evidence': {'type': 'array', 'items': {'type': 'object', 'properties': {
            'verse_id': {'type': 'string'}, 'quote': {'type': 'string'}}, 'required': ['verse_id', 'quote']}},
        'confidence': {'type': 'string', 'enum': ['high', 'medium', 'low']},
        'closest_verses': {'type': 'array', 'items': {'type': 'string'}},
        'modern_context': {'type': 'string'},
    },
    'required': ['answerable', 'answer', 'evidence', 'confidence', 'closest_verses', 'modern_context'],
}
CITE = re.compile(r'\[(\d{1,2}\.\d{1,3})\]')
norm = lambda s: re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()


def build_user(question, passages, concepts, include_modern):
    ctx = '\n\n'.join(f"[{p['verse_id']}] {p['english']}" for p in passages)
    cm = ''
    if concepts:
        cm = '\nConcepts detected in the question (modern -> the text\'s terms): ' + '; '.join(
            f"{c['label'][:60]} -> {', '.join(c['terms'][:6])}" for c in concepts) + '\n'
    return (f"Question: {question}\n{cm}\nVerses:\n{ctx}\n\n"
            f"Modern context enabled: {'yes' if include_modern else 'no'}")


def parse_json(text):
    text = re.sub(r'^```(?:json)?|```$', '', text.strip(), flags=re.M).strip()
    m = re.search(r'\{.*\}', text, re.S)
    return json.loads(m.group(0) if m else text)


def validate(out, passages, include_modern=False):
    """Keep only citations/evidence that point at supplied verses and quotes that really occur in them."""
    ids = {p['verse_id']: norm(p['english']) for p in passages}
    warn = []
    answer = str(out.get('answer', '') or '').strip()
    cited = list(dict.fromkeys(CITE.findall(answer)))
    bad = [c for c in cited if c not in ids]
    if bad: warn.append(f"answer cites verses that were not retrieved: {', '.join(bad)}")
    ev = []
    for e in out.get('evidence') or []:
        if not isinstance(e, dict): continue
        v, q = str(e.get('verse_id', '')).strip('[] '), str(e.get('quote', '')).strip()
        if v not in ids: warn.append(f'evidence from non-retrieved verse {v} dropped'); continue
        words = q.split()
        if len(words) > 20: warn.append(f'quote from {v} trimmed to 20 words'); q = ' '.join(words[:20])
        ev.append(dict(verse_id=v, quote=q, verified=bool(norm(q)) and norm(q) in ids[v]))
    if any(not e['verified'] for e in ev): warn.append('some quotes do not appear verbatim in the cited verse')
    good = [c for c in cited if c in ids]
    answerable = bool(out.get('answerable')) and bool(good or ev)
    if out.get('answerable') and not answerable:
        warn.append('answer had no valid citation -> treated as "not answered"')
    conf = str(out.get('confidence', '')).lower()
    return dict(answerable=answerable, answer=answer if answerable else 'The text does not answer this.',
                citations=good or list(dict.fromkeys(e['verse_id'] for e in ev)), evidence=ev,
                confidence=conf if conf in ('high', 'medium', 'low') else 'low',
                closest_verses=[str(c).strip('[] ') for c in out.get('closest_verses') or [] if str(c).strip('[] ') in ids],
                modern_context=str(out.get('modern_context', '') or '') if include_modern else '',
                warnings=warn)


def generate(question, passages, concepts=(), include_modern=False, cfg=None, model=None):
    """Raises LLMError (connection / HTTP / empty output) so the caller can show the real reason."""
    text = chat_json(SYSTEM, build_user(question, passages, list(concepts), include_modern), SCHEMA, cfg, model)
    try:
        out = parse_json(text)
    except ValueError as e:
        raise LLMError(f'model did not return JSON: {text[:300]}') from e
    return validate(out, passages, include_modern)
