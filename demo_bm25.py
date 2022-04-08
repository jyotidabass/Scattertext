import scattertext as st

movie_df = st.SampleCorpora.RottenTomatoes.get_data()
movie_df.category = movie_df.category\
	.apply(lambda x: {'rotten': 'Negative', 'fresh': 'Positive', 'plot': 'Plot'}[x])

corpus = st.CorpusFromPandas(
	movie_df,
	category_col='category',
	text_col='text',
	nlp=st.whitespace_nlp_with_sentences
).build().get_unigram_corpus()

term_scorer = (st.BM25Difference(corpus, k1=1.2, b=0.9)
               .set_categories('Positive', ['Negative'], ['Plot']))

html = st.produce_frequency_explorer(
	corpus,
	category='Positive',
	not_categories=['Negative'],
	neutral_categories=['Plot'],
	term_scorer=term_scorer,
	metadata=movie_df['movie_name'],
	grey_threshold=0,
	show_neutral=True
)
file_name = 'demo_bm25.html'
open(file_name, 'wb').write(html.encode('utf-8'))
print('./' + file_name)
