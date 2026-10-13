import os, streamlit as st, pandas as pd
from core.data import load_corpus, load_glossary, load_questions
from core.retrieval import Retriever
from core.pipeline import answer, DEFAULT_THRESHOLD

st.set_page_config(page_title='Surya Siddhanta concept-aware QA', layout='wide')

@st.cache_resource
def get_retriever(w, use_dense):
    dense = None
    if use_dense:
        try:
            from sentence_transformers import SentenceTransformer
            dense = SentenceTransformer(os.getenv('DENSE_MODEL', 'paraphrase-multilingual-MiniLM-L12-v2'))
        except Exception as e:
            st.sidebar.warning(f'Dense model unavailable ({e.__class__.__name__}); using BM25 only.')
    return Retriever(load_corpus(), load_glossary(), w=w, dense=dense)

Q = load_questions()
st.title('Surya Siddhanta: concept-aware question answering')
st.caption('Answers are grounded in Burgess\'s English translation of the Surya Siddhanta. The glossary has 57 entries mined from this one text.')

with st.sidebar:
    st.header('Settings')
    view = st.radio('Retrieval', ['Proposed (glossary-aware)', 'Baseline (BM25)', 'Compare side by side'])
    k = st.slider('Verses retrieved', 3, 10, 5)
    thr = st.slider('Abstention gate (BM25 top-1)', 0.0, 30.0, float(DEFAULT_THRESHOLD), 0.25)
    w = st.slider('Glossary expansion weight', 0.0, 1.0, 0.5, 0.05)
    modern = st.checkbox('Include modern-astronomy context (outside the text)', value=False)
    use_dense = st.checkbox('Fuse with dense retrieval (needs sentence-transformers)', value=False)
    st.text_input('Claude model', value=os.getenv('CLAUDE_MODEL', 'claude-sonnet-5-5'), key='model')
    if not os.getenv('ANTHROPIC_API_KEY'): st.info('No ANTHROPIC_API_KEY set: answers fall back to the top retrieved verse.')

R = get_retriever(w, use_dense)
demo = st.selectbox('Pick a benchmark question (optional)', [''] + Q.question.tolist())
question = st.text_input('Ask a question', value=demo)

def show_passages(res):
    for i, p in enumerate(res['results'], 1):
        with st.expander(f"{i}. Verse {p['verse_id']}  (score {p['score']:.1f})", expanded=(i == 1)):
            a, b = st.columns(2); a.markdown('**Sanskrit (ITRANS)**'); a.code(p['sanskrit'], language=None)
            b.markdown('**English (Burgess)**'); b.write(p['english'])

def show_concepts(res):
    if res['matched']:
        st.dataframe(pd.DataFrame([dict(you_said=', '.join(m['triggered_by']), concept=m['gloss'], sanskrit_term=m['term'],
                      added_to_search=', '.join(m['forms']), confidence=m.get('confidence', ''), mapping_type=m.get('mapping_type', '')) for m in res['matched']]), hide_index=True, width='stretch')
    else: st.caption('No glossary concept detected in this question.')

if st.button('Ask', type='primary') and question.strip():
    gold = Q.loc[Q.question == question, 'gold_verse']
    gold = gold.iloc[0] if len(gold) and pd.notna(gold.iloc[0]) else None
    if view.startswith('Compare'):
        base = R.search(question, 'baseline', k); prop = R.search(question, 'proposed', k)
        c1, c2 = st.columns(2)
        for col, name, res in ((c1, 'Baseline (BM25)', base), (c2, 'Proposed (glossary-aware)', prop)):
            with col:
                st.subheader(name); 
                if name.startswith('Prop'): show_concepts(res)
                show_passages(res)
        if gold: st.caption(f"Benchmark gold verse: {gold}. In baseline top-{k}: {gold in [p['verse_id'] for p in base['results']]} | proposed top-{k}: {gold in [p['verse_id'] for p in prop['results']]}")
    else:
        mode = 'proposed' if view.startswith('Proposed') else 'baseline'
        with st.spinner('Retrieving and generating...'):
            out = answer(question, R, mode, k, thr, modern, st.session_state.get('model'))
        if mode == 'proposed': st.subheader('Concepts detected'); show_concepts(out['retrieval'])
        if out['decision'] == 'abstain':
            st.warning(f"The text does not give enough evidence to answer this. ({out['reason']})")
        else:
            a = out['answer']
            if a.get('fallback'): st.caption('Fallback mode: showing the top retrieved verse, no model-written answer.')
            st.subheader('Answer'); st.markdown('**What the text states**'); st.write(a['what_text_states'])
            if a['translation_note']: st.markdown('**Translation caveat**'); st.write(a['translation_note'])
            if a['modern_context']: st.markdown('**Modern context (outside the text)**'); st.write(a['modern_context'])
            st.caption('Cited verses: ' + ', '.join(a['citations']))
            if a['citation_warnings']: st.warning('Citation check: ' + '; '.join(a['citation_warnings']))
        st.subheader('Retrieved verses'); show_passages(out['retrieval'])
        if gold: st.caption(f"Benchmark gold verse: {gold} | retrieved in top-{k}: {gold in [p['verse_id'] for p in out['retrieval']['results']]}")
