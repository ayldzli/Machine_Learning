from abc import ABC, abstractmethod
import pandas as pd
import os

class BaseDataLoader(ABC):
    """Abstract Base Class for all data loaders."""
    @abstractmethod
    def load_data(self, file_path: str) -> pd.DataFrame:
        pass


class CSVDataLoader(BaseDataLoader):
    """Concrete loader specialized for CSV files."""
    def load_data(self, file_path: str) -> pd.DataFrame:
        print(f"[CSVDataLoader] {file_path} yükleniyor...")
        return pd.read_csv(file_path)


class JSONDataLoader(BaseDataLoader):
    """Concrete loader specialized for JSON files."""
    def load_data(self, file_path: str) -> pd.DataFrame:
        print(f"[JSONDataLoader] {file_path} yükleniyor...")
        return pd.read_json(file_path)


class DataLoaderFactory:
    """
    Automatically detects the file extension 
    and instantiates the correct loader polymorphically.
    """
    @staticmethod
    def get_loader(file_path: str) -> BaseDataLoader:
        _, extension = os.path.splitext(file_path)
        extension = extension.lower()

        if extension == '.csv':
            return CSVDataLoader()
        elif extension == '.json':
            return JSONDataLoader()
        else:
            raise ValueError(f"Desteklenmeyen dosya formatı: {extension}")

class MLDataManager:
    """
    Unified utility class 
    that serves as the single entry point for all data operations.
    """
    
    @staticmethod
    def load(file_path: str) -> pd.DataFrame:
        """Automatically handles factory lookup and loads data in one step."""
        loader = DataLoaderFactory.get_loader(file_path)
        return loader.load_data(file_path)
