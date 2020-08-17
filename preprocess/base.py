from abc import ABC, abstractmethod
from typing import List


class BasePreprocessor(ABC):
    @abstractmethod
    def _process_token(self, w):
        pass

    @abstractmethod
    def process_text(self, w):
        pass

    def process_documents(self, docs: List):
        for d in docs:
            yield self.process_text(d)
