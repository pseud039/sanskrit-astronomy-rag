"""Provider-neutral chat client over plain HTTP (no vendor SDK).
LLM_PROVIDER = ollama  -> POST {LLM_BASE_URL}/api/chat  (local or hosted Ollama; JSON-schema constrained output)
             = openai  -> POST {LLM_BASE_URL}/chat/completions  (any OpenAI-compatible server: Groq, OpenRouter,
                          Together, Hugging Face router, vLLM, LM Studio, Ollama's /v1 ...)
             = none    -> generation disabled (the app says so explicitly)
LLM_MODEL, LLM_API_KEY (Bearer; optional for local Ollama), LLM_TIMEOUT (seconds, default 60)."""
import os, json, logging, requests

log = logging.getLogger('ss-rag')
DEFAULTS = {'ollama': ('http://localhost:11434', 'qwen2.5:7b-instruct'),
            'openai': ('https://api.groq.com/openai/v1', 'llama-3.3-70b-versatile')}


class LLMError(RuntimeError):
    pass


def config():
    p = os.getenv('LLM_PROVIDER', '').strip().lower()
    if not p:  # infer: an explicit base URL or key means "configured"
        p = 'openai' if os.getenv('LLM_API_KEY') else ('ollama' if os.getenv('LLM_BASE_URL') else 'none')
    if p not in ('ollama', 'openai', 'none'):
        return dict(provider='none', error=f'unknown LLM_PROVIDER={p!r} (use ollama, openai or none)')
    if p == 'none': return dict(provider='none', error='')
    base, model = DEFAULTS[p]
    return dict(provider=p, base_url=os.getenv('LLM_BASE_URL', base).rstrip('/'), model=os.getenv('LLM_MODEL', model),
                api_key=os.getenv('LLM_API_KEY', ''), timeout=float(os.getenv('LLM_TIMEOUT', '60')), error='')


def describe(cfg=None):
    cfg = cfg or config()
    if cfg['provider'] == 'none': return cfg['error'] or 'not configured (set LLM_PROVIDER / LLM_BASE_URL / LLM_MODEL)'
    return f"{cfg['provider']} · {cfg['model']} @ {cfg['base_url']}"


def _post(url, payload, cfg):
    headers = {'Content-Type': 'application/json'}
    if cfg.get('api_key'): headers['Authorization'] = f"Bearer {cfg['api_key']}"
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=cfg['timeout'])
    except requests.RequestException as e:
        raise LLMError(f'cannot reach {url}: {e.__class__.__name__}: {e}') from e
    if r.status_code >= 400:
        raise LLMError(f'{url} returned HTTP {r.status_code}: {r.text[:400]}')
    try:
        return r.json()
    except ValueError as e:
        raise LLMError(f'{url} returned non-JSON body: {r.text[:200]}') from e


def chat_json(system, user, schema, cfg=None, model=None):
    """Returns the model's raw text (expected to be a JSON object). Raises LLMError with a readable message."""
    cfg = dict(cfg or config())
    if cfg['provider'] == 'none': raise LLMError(describe(cfg))
    if model: cfg['model'] = model
    msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
    if cfg['provider'] == 'ollama':
        out = _post(f"{cfg['base_url']}/api/chat", dict(model=cfg['model'], messages=msgs, stream=False, format=schema,
                                                       options=dict(temperature=0)), cfg)
        text = (out.get('message') or {}).get('content', '')
    else:
        payload = dict(model=cfg['model'], messages=msgs, temperature=0, response_format={'type': 'json_object'})
        try:
            out = _post(f"{cfg['base_url']}/chat/completions", payload, cfg)
        except LLMError as e:  # some servers reject response_format; the prompt still asks for JSON
            if 'response_format' not in str(e): raise
            payload.pop('response_format'); out = _post(f"{cfg['base_url']}/chat/completions", payload, cfg)
        try:
            text = out['choices'][0]['message']['content'] or ''
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f'unexpected response shape: {json.dumps(out)[:300]}') from e
    if not text.strip(): raise LLMError('model returned an empty response')
    return text
