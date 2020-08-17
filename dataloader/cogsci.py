import pathlib
from abc import ABC

from dataloader.base import DataReader, BaseData, List


class CogSciData(BaseData, ABC):
    def __init__(self, file_path):
        super(CogSciData, self).__init__(file_path)

    def _label(self):
        return pathlib.Path(self.file_path).parent.name


class CogSciDataReader(DataReader):
    def __int__(self, location: str, excluded_extensions: List[str] = None):
        super().__init__(location=location, data_type=CogSciData,
                         excluded_extensions=excluded_extensions)
