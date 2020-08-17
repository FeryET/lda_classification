import os
from abc import ABC, abstractmethod
from typing import List, Iterable, Generic, TypeVar


class BaseData(ABC):
    def __init__(self, file_path):
        self.file_path = file_path
        self.text = self._text
        self.label = self._label  # print("item_created", self.label)

    @property
    def _label(self):
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


class DataReader:
    def __init__(self, location: str, data_type: BaseData = BaseData,
                 excluded_extensions: List[str] = None):
        self.location = location
        self.object_type = data_type
        self.excluded_extensions = excluded_extensions

    def __iter__(self):
        for root, _, files in os.walk(self.location):
            for f in files:
                file_path = os.path.join(root, f)
                yield self.object_type(file_path)
