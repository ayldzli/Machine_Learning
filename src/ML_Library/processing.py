from abc import ABC, abstractmethod
import pandas as pd
from typing import Callable, Optional

class ImputationStrategy(ABC):
    """Abstract interface for handling missing values."""
    @abstractmethod
    def compute_value(self, series: pd.Series) -> float:
        pass


class MeanImputer(ImputationStrategy):
    """Calculates the statistical mean."""
    def compute_value(self, series: pd.Series) -> float:
        return float(series.mean())


class MedianImputer(ImputationStrategy):
    """Calculates the statistical median."""
    def compute_value(self, series: pd.Series) -> float:
        return float(series.median())


class DataProcessor:
    """Orchestrates data cleaning and feature transformations."""
    def __init__(self, strategy: ImputationStrategy):
        self.strategy = strategy

    def fill_missing(self, df: pd.DataFrame, column: Optional[str] = None) -> pd.DataFrame:
        """
        Scans and imputes missing values. If no column name is supplied,
        it automatically cleans all numeric columns across the entire DataFrame.
        """
        df_copy = df.copy()
        
        if column is not None:
            
            if df_copy[column].isnull().any():
                fill_value = self.strategy.compute_value(df_copy[column].dropna())
                df_copy[column] = df_copy[column].fillna(fill_value)
        else:
            
            print("[DataProcessor] Scanning...")
            for col in df_copy.columns:
    
                if pd.api.types.is_numeric_dtype(df_copy[col]):
                    if df_copy[col].isnull().any():
                        fill_value = self.strategy.compute_value(df_copy[col].dropna())
                        print(f"  --> '{col}' column null values filled with: {fill_value:.4f}")
                        df_copy[col] = df_copy[col].fillna(fill_value)
                        
        return df_copy

    def create_custom_transformer(self, func: Callable[[pd.Series], pd.Series]) -> Callable[[pd.Series], pd.Series]:
        """Generates custom stateful feature mappings."""
        def transformer(series: pd.Series) -> pd.Series:
            return func(series)
        return transformer
