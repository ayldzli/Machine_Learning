from .data_loader import MLDataManager  # <- Expose the static class
from .processing import DataProcessor, MeanImputer, MedianImputer
from .models import SimpleDecisionTree, LinearRegressionFromScratch
from .evaluation import ClassificationEvaluator, RegressionEvaluator

__all__ = [
    "MLDataManager",  # <- Expose it here
    "DataProcessor",
    "MeanImputer",
    "MedianImputer",
    "SimpleDecisionTree",
    "LinearRegressionFromScratch",
    "ClassificationEvaluator",
    "RegressionEvaluator"
]