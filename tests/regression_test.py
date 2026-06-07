import pytest
import numpy as np
from ML_Library import RegressionEvaluator

def test_regression_metrics_calculation():
    """MAE, MSE ve RMSE hesaplamalarının matematiksel doğruluğunu test eder."""
    reg_evaluator = RegressionEvaluator()
    
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([12.0, 19.0]) 
    # Hatalar: (10-12) = -2  ve  (20-19) = 1
    # Mutlak Hatalar: 2 ve 1 -> MAE = (2 + 1) / 2 = 1.5
    # Kare Hatalar: 4 ve 1 -> MSE = (4 + 1) / 2 = 2.5
    # RMSE = sqrt(2.5) ~= 1.5811
    
    metrics = reg_evaluator.calculate_metrics(y_true, y_pred)
    
    assert metrics["mae"] == 1.5
    assert metrics["mse"] == 2.5
    assert np.isclose(metrics["rmse"], np.sqrt(2.5))