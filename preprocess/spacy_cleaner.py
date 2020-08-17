from typing import List
import multiprocessing as mp
import spacy

from preprocess.base import BasePreprocessor

nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner'])
nlp.add_pipe(nlp.create_pipe('sentencizer'))

workers = mp.cpu_count() // 2


class SpacyCleaner(BasePreprocessor):
    def _process_token(self, token):
        def is_valid(input_token):
            return token.is_alpha and not token.is_stop

        return token.lemma_.lower() if is_valid(token) else ""

    def process_text(self, text):
        return " ".join([self._process_token(token) for token in nlp(text)])

    def process_documents(self, docs: List):
        for doc in nlp.pipe(docs, batch_size=30):
            yield " ".join([self._process_token(token) for token in doc])

    def multi_thread_process_documents(self, docs: List):
        with mp.Pool(processes=workers) as pool:
            return pool.map(self.process_text, docs)
