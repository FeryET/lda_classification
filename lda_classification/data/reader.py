import os
import pandas as pd

from lda_classification.data.datatypes import BaseData


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

    def to_dataframe(self):
        elements = [row.as_dict() for row in self]
        return pd.DataFrame(elements)
