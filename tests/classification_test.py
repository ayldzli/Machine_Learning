import pytest
import numpy as np
import pandas as pd
from ML_Library import (
    MLDataManager, 
    DataProcessor, 
    MeanImputer, 
    SimpleDecisionTree, 
    ClassificationEvaluator
)

def test_data_loader_factory_and_strategy():
    """Factory ve Strategy desenlerinin entegrasyonunu test eder."""
    # CSV verisi simüle et
    df_raw = pd.DataFrame({'Feature1': [2.0, np.nan, 4.0]})
    df_raw.to_csv("tests/data/test_clf_temp.csv", index=False)
    
    # DataManager Test
    df = MLDataManager.load("tests/data/test_clf_temp.csv")
    assert isinstance(df, pd.DataFrame)
    
    # Strategy (MeanImputer) test
    processor = DataProcessor(strategy=MeanImputer())
    df_clean = processor.fill_missing(df, "Feature1")
    
    # 2.0 ve 4.0'ın ortalaması 3.0 olmalı, nan yerine 3.0 gelmeli
    assert df_clean["Feature1"].iloc[1] == 3.0

def test_functional_transformer():
    """Functional Programming (Closure) yapısını test eder."""
    processor = DataProcessor(strategy=MeanImputer())
    # Değeri 3 ile çarpan bir closure fonksiyonu üretelim
    triple_transformer = processor.create_custom_transformer(lambda x: x * 3)
    
    series = pd.Series([1, 2, 3])
    result = triple_transformer(series)
    
    assert result.iloc[0] == 3
    assert result.iloc[2] == 9

def test_decision_tree_fit_and_predict():
    """Karar ağacının rekürsif fit ve predict süreçlerini test eder."""
    X = np.array([[1.0], [2.0], [10.0], [11.0]])
    y = np.array([0, 0, 1, 1])
    
    tree = SimpleDecisionTree(max_depth=2)
    tree.fit(X, y)
    
    # Ağacın kökünün başarıyla kurulduğunu doğrula
    assert tree.root is not None
    
    # Yeni tahminler üret ve kontrol et
    preds = tree.predict(np.array([[1.5], [10.5]]))
    assert preds[0] == 0
    assert preds[1] == 1

def test_classification_metrics():
    """Sınıflandırma metrik motorunun matematiksel doğruluğunu test eder."""
    evaluator = ClassificationEvaluator()
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0]) # 4 tahminden 3'ü doğru (Accuracy = 0.75)
    
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.75
    assert "f1_score" in metrics
