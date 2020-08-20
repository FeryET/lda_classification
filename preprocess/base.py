from abc import ABC, abstractmethod
from typing import List
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import multiprocessing as mp
from sklearn.base import TransformerMixin


class BasePreprocessor(ABC, TransformerMixin):
    def __init__(self, workers=2, chunksize=100, enable_tqdm=True):
        self.workers = workers
        self.chunksize = chunksize
        self.enable_tqdm = enable_tqdm

    @abstractmethod
    def _process_token(self, w):
        pass

    @abstractmethod
    def process_text(self, text):
        pass

    def transform(self, docs):
        if self.enable_tqdm is True:
            return process_map(self.process_text, docs,
                               max_workers=self.workers,
                               desc=self.__class__.__name__,
                               chunksize=self.chunksize)
        else:
            with mp.Pool(self.workers) as pool:
                results = pool.map(self.process_text, docs,
                                   chunksize=self.chunksize)
            return results

    def fit_transform(self, docs, y=None, **fit_params):
        return self.transform(docs)
