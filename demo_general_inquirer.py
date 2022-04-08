from __future__ import print_function

from scattertext.WhitespaceNLP import whitespace_nlp_with_sentences

from scattertext import FeatsFromGeneralInquirer, produce_scattertext_explorer
from scattertext import SampleCorpora
from scattertext.CorpusFromPandas import CorpusFromPandas


def main():
	convention_df = SampleCorpora.ConventionData2012.get_data()
	feat_builder = FeatsFromGeneralInquirer()
	corpus = CorpusFromPandas(convention_df,
	                          category_col='party',
	                          text_col='text',
	                          nlp=whitespace_nlp_with_sentences,
	                          feats_from_spacy_doc=feat_builder).build()
	html = produce_scattertext_explorer(corpus,
	                                    category='democrat',
	                                    category_name='Democratic',
	                                    not_category_name='Republican',
	                                    width_in_pixels=1000,
	                                    metadata=convention_df['speaker'],
	                                    use_non_text_features=True,
	                                    use_full_doc=True,
	                                    topic_model_term_lists=feat_builder.get_top_model_term_lists(),
										metadata_descriptions=feat_builder.get_definitions()
										)
	open('./demo_general_inquirer.html', 'wb').write(html.encode('utf-8'))
	print('Open ./demo_general_inquirer.html in Chrome or Firefox.')


if __name__ == '__main__':
	main()
