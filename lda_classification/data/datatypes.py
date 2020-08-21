import pathlib
from abc import ABC, abstractmethod


class BaseData(ABC):
    """This is the base class for any type of data that is going to be
    handled by
    data.reader.DataReader"""

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


class CogSciData(BaseData):
    def __init__(self, file_path):
        super().__init__(file_path)

    @property
    def label(self):
        return pathlib.Path(self.file_path).parent.name
