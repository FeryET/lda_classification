import pathlib
from abc import ABC

from dataloader.reader import DataReader, BaseData, List


class CogSciData(BaseData):
    def __init__(self, file_path):
        super().__init__(file_path)

    @property
    def label(self):
        return pathlib.Path(self.file_path).parent.name
