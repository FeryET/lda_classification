from abc import ABC, abstractmethod
from typing import List
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import multiprocessing as mp

workers = mp.cpu_count() // 2


class BasePreprocessor(ABC):
    @abstractmethod
    def _process_token(self, w):
        pass

    @abstractmethod
    def process_text(self, w):
        pass

    def process_documents(self, docs: List):
        for d in tqdm(docs):
            yield self.process_text(d)

    def process_documents_multithread(self, docs: List):
        return process_map(self.process_text, docs, max_workers=workers,
                           desc=self.__class__.__name__)
