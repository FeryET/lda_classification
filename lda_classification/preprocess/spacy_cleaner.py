from typing import List

import spacy
from tqdm import tqdm

from lda_classification.preprocess.base import BasePreprocessor

nlp = spacy.load('en', disable=['tagger', 'parser', 'ner'])
nlp.add_pipe(nlp.create_pipe('sentencizer'))

MIN_TOKEN_LENGTH = 3


class SpacyCleaner(BasePreprocessor):
    def _process_token(self, token):
        return token.lemma_.lower()

    def process_text(self, text):

        return [self._process_token(token) for token in nlp(text) if
                _is_valid(token)]

    def process_documents(self, docs: List):
        for doc in tqdm(nlp.pipe(docs, batch_size=30)):
            yield [self._process_token(token) for token in doc if
                   _is_valid(token)]


def _is_valid(token):
    return token.is_alpha and not token.is_stop and len(
            token) > MIN_TOKEN_LENGTH
