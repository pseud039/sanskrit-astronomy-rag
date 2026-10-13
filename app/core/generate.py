"""Grounded answer generation. Claude may use ONLY the retrieved verses; output is validated JSON with citations."""
import os, json, re

SYSTEM = """You answer questions about the Surya Siddhanta using ONLY the numbered verse passages supplied (Burgess's English translation, with some OCR noise).
Rules:
- Every factual statement must come from the passages. Do not use outside knowledge for what_text_states.
- Cite verse ids exactly as given, e.g. "12.39". Never cite an id that was not supplied.
- If the passages do not answer the question, set evidence_sufficient to false and leave what_text_states empty. Do not guess.
- translation_note: flag where the English translation (Burgess's wording or editorial notes) may differ from the Sanskrit, or the passage is garbled. Empty string if none.
- modern_context: at most two sentences linking to modern astronomy, clearly framed as outside the text. Empty string if not requested.
Return ONLY a JSON object: {"evidence_sufficient": bool, "what_text_states": str, "translation_note": str, "modern_context": str, "citations": [str]}"""

def build_user(question, passages, concepts, include_modern):
    ctx = '\n\n'.join(f"[{p['verse_id']}] {p['english']}" for p in passages)
    cm = ''
    if concepts:
        cm = '\nGlossary concepts detected in the question: ' + '; '.join(f"{c['gloss']} = {c['term']}" for c in concepts) + '\n'
    return (f"Question: {question}\n{cm}\nPassages:\n{ctx}\n\n"
            f"include_modern_context: {'yes' if include_modern else 'no'}")

def parse_json(text):
    text = re.sub(r'^```(?:json)?|```$', '', text.strip(), flags=re.M).strip()
    m = re.search(r'\{.*\}', text, re.S)
    return json.loads(m.group(0) if m else text)

def validate(out, passages):
    ids = {p['verse_id'] for p in passages}
    cites = [str(c) for c in out.get('citations', [])]
    bad = [c for c in cites if c not in ids]
    out['citations'] = [c for c in cites if c in ids]
    out['citation_warnings'] = bad
    if out.get('evidence_sufficient') and not out['citations']:
        out['evidence_sufficient'] = False; out['citation_warnings'].append('answer had no valid citation -> treated as abstain')
    for k in ('what_text_states', 'translation_note', 'modern_context'): out[k] = str(out.get(k, '') or '')
    return out

def generate(question, passages, concepts=(), include_modern=False, model=None, client=None):
    if client is None:
        key = os.getenv('ANTHROPIC_API_KEY')
        if not key:  # extractive fallback so the demo still runs
            p = passages[0]
            return dict(evidence_sufficient=True, what_text_states=p['english'], translation_note='',
                        modern_context='', citations=[p['verse_id']], citation_warnings=[], fallback=True)
        import anthropic; client = anthropic.Anthropic(api_key=key)
    model = model or os.getenv('CLAUDE_MODEL', 'claude-sonnet-5-5')
    msg = client.messages.create(model=model, max_tokens=900, temperature=0, system=SYSTEM,
        messages=[{'role': 'user', 'content': build_user(question, passages, list(concepts), include_modern)}])
    text = ''.join(b.text for b in msg.content if getattr(b, 'type', '') == 'text')
    try: out = parse_json(text)
    except Exception: return dict(evidence_sufficient=False, what_text_states='', translation_note='', modern_context='',
                                  citations=[], citation_warnings=['model returned non-JSON; treated as abstain'], raw=text)
    return validate(out, passages)
