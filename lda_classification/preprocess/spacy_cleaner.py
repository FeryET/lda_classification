import multiprocessing
from typing import List

import spacy
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import numpy as np
from lda_classification.preprocess.base import BasePreprocessor
from lda_classification.utils.progress_parallel import ProgressParallel
from joblib import delayed

nlp = spacy.load('en', disable=['tagger', 'parser', 'ner'])
nlp.add_pipe(nlp.create_pipe('sentencizer'))

MIN_TOKEN_LENGTH = 3


def _chunker(iterable, chunksize, total_length=None):
    if total_length is None:
        total_length = len(iterable)
    return (iterable[pos: pos + chunksize] for pos in
            range(0, total_length, chunksize))


def _process(token):
    return token.lemma_.lower()


def _flatten(l):
    return list(np.array(l, dtype=object).flat)


def _is_valid(token):
    return token.is_alpha and not token.is_stop and len(
            token) > MIN_TOKEN_LENGTH


def _process_chunk(docs_chunk):
    result = []
    for doc in nlp.pipe(docs_chunk, batch_size=20):
        result.append([_process(token) for token in doc if _is_valid(token)])
    return result


class SpacyCleaner(BasePreprocessor):
    def _process_token(self, w):
        return _process(w)

    def process_text(self, text):
        return [_process(token) for token in nlp(text) if _is_valid(token)]

    def process_documents(self, docs: List):
        do = delayed(_process_chunk)
        tasks = (do(chunk) for chunk in
                 _chunker(docs, chunksize=self.chunksize))
        total = len(docs)/self.chunksize
        executor = ProgressParallel(n_jobs=self.workers, total=total,
                                    chunk_size=self.chunksize,
                                    use_tqdm=self.enable_tqdm,
                                    backend='multiprocessing',
                                    prefer="processes")
        result = executor(tasks)
        return _flatten(result)
