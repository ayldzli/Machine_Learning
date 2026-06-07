import numpy as np
import pandas as pd
from ML_Library import (
    MLDataManager, 
    DataProcessor, 
    MeanImputer, 
    SimpleDecisionTree, 
    ClassificationEvaluator
)

def main():
    print("=== SENARYO 1: SINIFLANDIRMA (CLASSIFICATION) DEMOSU ===\n")

    # 1. Test verisi oluşturma
    df_raw = pd.DataFrame({
        'Feature1': [1.2, 2.3, np.nan, 4.5, 5.1, 1.1, 2.2, 3.3, 4.2],
        'Label': [0, 1, 0, 1, 1, 0, 0, 1, 1]
    })
    df_raw.to_csv("examples/data/classification_data.csv", index=False)

    # 2. Factory Pattern ile yükleme
    df = MLDataManager.load("examples/data/classification_data.csv")

    # 3. Strategy Pattern ile eksik veri doldurma
    # NEW SMART WAY
    processor = DataProcessor(strategy=MeanImputer())

    # Cleans all missing values across the entire dataset automatically!
    df_clean = processor.fill_missing(df)

    # 4. Functional Programming (Closure)
    double_transformer = processor.create_custom_transformer(lambda x: x * 2)
    df_clean["Feature1_Doubled"] = double_transformer(df_clean["Feature1"])

    print("\n[DataProcessor] Hazırlanan Sınıflandırma Verisi:")
    print(df_clean)

    # 5. Veriyi NumPy matrisine çevirme
    X = df_clean[["Feature1", "Feature1_Doubled"]].to_numpy()
    y = df_clean["Label"].to_numpy()
    
    # Model fabrikası (Thread'ler için)
    tree_factory = lambda: SimpleDecisionTree(max_depth=3)

    # 6. Concurrency ile Paralel Cross-Validation ve Sınıflandırma Metrikleri
    evaluator = ClassificationEvaluator()
    print("\n--- Paralel Cross-Validation Başlatılıyor ---")
    results = evaluator.parallel_cross_validation(tree_factory, X, y, n_folds=3)

    print("\n--- Sınıflandırma Katman Bazlı Detaylı Sonuçlar ---")
    for idx, fold_metrics in enumerate(results):
        print(
            f"Fold {idx} -> "
            f"Accuracy: {fold_metrics['accuracy']:.4f} | "
            f"Precision: {fold_metrics['precision']:.4f} | "
            f"Recall: {fold_metrics['recall']:.4f} | "
            f"F1-Score: {fold_metrics['f1_score']:.4f}"
        )

    # Ortalamaları hesaplama
    mean_accuracy = np.mean([fold['accuracy'] for fold in results])
    mean_f1 = np.mean([fold['f1_score'] for fold in results])

    print("\n--- Sınıflandırma Genel Performans Özeti ---")
    print(f"Ortalama Doğruluk (Mean Accuracy): {mean_accuracy:.4f}")
    print(f"Ortalama F1-Skoru (Mean F1-Score) : {mean_f1:.4f}")

if __name__ == "__main__":
    main()