import os
from abc import ABC, abstractmethod
from typing import List, Iterable, Generic, TypeVar
import pandas as pd


class BaseData(ABC):
    def __init__(self, file_path):
        self.file_path = file_path
        self.text = self._text

    @property
    @abstractmethod
    def label(self):
        pass

    @property
    def _text(self):
        with open(self.file_path) as readfile:
            t = readfile.read()
            return t

    def __str__(self):
        return self.text

    def update(self, text):
        self.text = text
        return self

    def __len__(self):
        return len(self.text)

    def __repr__(self):
        return "LABEL: {}\t TEXT: {}".format(self.label, self.text)

    def __getitem__(self, sliced):
        return self.text[sliced]

    def as_dict(self):
        return {"text": self.text, "label": self.label}


class DataReader:
    def __init__(self, location: str, data_type: BaseData,
                 excluded_extensions=None):
        """
        :param location: location of the file :param data_type: type of the
        data this loader wants to load (should be a subclass of BaseData)
        :param excluded_extensions: which extensions should be avoided in the
        path
        """
        if excluded_extensions is None:
            excluded_extensions = []
        self.location = location
        self.object_type = data_type
        self.excluded_extensions = excluded_extensions

    def __iter__(self):
        for root, _, files in os.walk(self.location):
            for f in files:
                ext = os.path.splitext(f)[-1]
                if self.excluded_extensions is not None and ext not in \
                        self.excluded_extensions:
                    file_path = os.path.join(root, f)
                    yield self.object_type(file_path)

    def to_pandas(self):
        elements = [row.as_dict() for row in self]
        return pd.DataFrame(elements)
