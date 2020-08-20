from lda_classification.preprocess.base import BasePreprocessor
from gensim.parsing.preprocessing import (stem_text, strip_tags,
                                          strip_short, strip_punctuation2,
                                          strip_multiple_whitespaces,
                                          remove_stopwords, strip_numeric,
                                          preprocess_string)


class GensimCleaner(BasePreprocessor):
    FILTERS = [strip_multiple_whitespaces, strip_tags, strip_punctuation2,
               strip_numeric, remove_stopwords, stem_text, strip_short]

    def _process_token(self, w):
        raise NotImplementedError

    def process_text(self, input_text):
        return preprocess_string(input_text, GensimCleaner.FILTERS)
