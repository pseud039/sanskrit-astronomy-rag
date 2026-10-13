import os, logging, streamlit as st, pandas as pd
from core.data import load_corpus, load_glossary, load_lexicon, load_questions
from core.retrieval import Retriever, METHODS
from core.models import load_dense, load_reranker
from core.pipeline import answer
from core.llm import config, describe
from core.text import sanskrit_display

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
st.set_page_config(page_title='Surya Siddhanta concept-aware QA', layout='wide')


@st.cache_resource
def get_models(use_dense, use_rerank):
    dense = emb = rr = None; notes = []
    if use_dense:
        dense, emb, msg = load_dense(); notes.append(msg)
    if use_rerank:
        rr, msg = load_reranker(); notes.append(msg)
    return dense, emb, rr, notes


@st.cache_resource
def get_retriever(w, use_dense, use_rerank):
    dense, emb, rr, notes = get_models(use_dense, use_rerank)
    return Retriever(load_corpus(), load_glossary(), load_lexicon(), w=w, dense=dense, corpus_emb=emb, reranker=rr), notes


Q = load_questions()
cfg = config()
st.title('Surya Siddhanta: concept-aware question answering')
st.caption("Answers are grounded in Burgess's 1860 English translation of the Surya Siddhanta. Sanskrit text: GRETIL e-text "
           "(input by Michio Yano), shown for research and teaching use.")

with st.sidebar:
    st.header('Settings')
    view = st.radio('Mode', ['Answer a question', 'Compare two retrieval methods'])
    k = st.slider('Verses retrieved', 3, 10, 5)
    w = st.slider('Concept expansion weight', 0.0, 1.0, 0.25, 0.05, help='0.25 was selected on the dev split.')
    use_dense = st.checkbox('Dense retrieval (needs sentence-transformers)', value=os.getenv('ENABLE_DENSE', '0') == '1')
    use_rerank = st.checkbox('Cross-encoder reranking (needs sentence-transformers)', value=os.getenv('ENABLE_RERANK', '0') == '1')
    modern = st.checkbox('Include modern-astronomy context (outside the text)', value=False)
    script = st.selectbox('Sanskrit script', ['IAST', 'Devanagari', 'ITRANS'])
    st.divider(); st.subheader('Status')
    (st.success if cfg['provider'] != 'none' else st.error)(f"LLM: {describe(cfg)}")

R, notes = get_retriever(w, use_dense, use_rerank)
with st.sidebar:
    for n in notes: st.caption(n)
methods = R.available_methods()
label = lambda m: METHODS[m]

demo = st.selectbox('Pick a benchmark question (optional)', [''] + Q.question.tolist())
question = st.text_input('Ask a question', value=demo)
if view.startswith('Compare'):
    s1, s2 = st.columns(2)
    pair = [s1.selectbox('Left method', methods, index=methods.index('bm25'), format_func=label),
            s2.selectbox('Right method', methods, index=methods.index('proposed'), format_func=label)]
else:
    method = st.radio('Retrieval method', methods, index=methods.index('proposed'), format_func=label, horizontal=True)


def show_concepts(res):
    if res['matched']:
        st.dataframe(pd.DataFrame([dict(source=m['source'], you_said=', '.join(m['triggered_by']), concept=m['label'][:80],
                                        added_to_search=', '.join(m['terms'][:8]), confidence=m.get('confidence', ''),
                                        mapping_type=m.get('mapping_type', '')) for m in res['matched']]),
                     hide_index=True, width='stretch')
    else:
        st.caption('No concept detected in this question: the proposed method falls back to plain retrieval.')


def show_passages(res):
    for i, p in enumerate(res['results'], 1):
        with st.expander(f"{i}. Verse {p['verse_id']}  (score {p['score']:.3f})", expanded=(i == 1)):
            a, b = st.columns(2)
            a.markdown(f'**Sanskrit ({script})**'); a.code(sanskrit_display(p['sanskrit'], script), language=None, wrap_lines=True)
            b.markdown('**English (Burgess)**'); b.write(p['english'])


def gold_note(res, gold):
    if gold: st.caption(f"Benchmark gold verse: {gold} | in top-{k}: {gold in [p['verse_id'] for p in res['results']]}")


if st.button('Ask', type='primary') and question.strip():
    gold = Q.loc[Q.question == question, 'gold_verse']
    gold = gold.iloc[0] if len(gold) and pd.notna(gold.iloc[0]) else None
    if view.startswith('Compare'):
        for col, m in zip(st.columns(2), pair):
            with col:
                res = R.search(question, m, k, rerank=use_rerank and R.reranker is not None)
                st.subheader(label(m))
                if m == 'proposed': show_concepts(res)
                show_passages(res); gold_note(res, gold)
    else:
        m = method
        with st.spinner('Retrieving and generating...'):
            out = answer(question, R, m, k, modern, rerank=use_rerank and R.reranker is not None, cfg=cfg)
        if m == 'proposed': st.subheader('Concepts detected'); show_concepts(out['retrieval'])
        d = out['decision']
        if d == 'error':
            st.error(out['reason'])
        elif d == 'no_llm':
            st.error(out['reason'] + ' Showing the retrieved verses only. Set LLM_PROVIDER, LLM_BASE_URL, LLM_MODEL '
                     '(and LLM_API_KEY for hosted APIs) in the server environment.')
        elif d == 'abstain':
            st.warning(f"The text does not answer this. ({out['reason']})")
            if out['answer'] and out['answer']['closest_verses']:
                st.caption('Closest verses: ' + ', '.join(out['answer']['closest_verses']))
        else:
            a = out['answer']
            st.subheader('Answer'); st.write(a['answer'])
            if a['evidence']:
                st.markdown('**Evidence**')
                for e in a['evidence']:
                    st.markdown(f"- [{e['verse_id']}] “{e['quote']}” " + ('✓' if e['verified'] else '⚠ not verbatim in the verse'))
            st.caption(f"Confidence: {a['confidence']} · Cited verses: {', '.join(a['citations'])}")
            if a['modern_context']: st.markdown('**Modern context (outside the text)**'); st.write(a['modern_context'])
            if a['warnings']: st.warning('Citation check: ' + '; '.join(a['warnings']))
        st.subheader('Retrieved verses'); show_passages(out['retrieval']); gold_note(out['retrieval'], gold)
