import scattertext as st

movie_df = st.SampleCorpora.RottenTomatoes.get_data()

corpus = st.CorpusFromPandas(
    movie_df,
    category_col='category',
    text_col='text',
    nlp=st.whitespace_nlp_with_sentences
).build().get_unigram_corpus().remove_categories(['plot'])

term_scorer = st.CredTFIDF(corpus).set_categories('fresh', ['rotten'])

print(term_scorer.get_score_df().sort_values(by='delta_cred_tf_idf', ascending=False).head())

html = st.produce_frequency_explorer(
    corpus,
    category='fresh',
    not_category_name='rotten',
    term_scorer=term_scorer,
    metadata=corpus.get_df()['movie_name'],
    grey_threshold=0
)
file_name = 'demo_cred_tfidf.html'
open(file_name, 'wb').write(html.encode('utf-8'))
print('Open %s in Chrome or Firefox.' % file_name)
