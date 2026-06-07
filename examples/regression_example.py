import numpy as np
from ML_Library import RegressionEvaluator

def main():
    print("=== SENARYO 2: REGRESYON (REGRESSION) METRİK DEMOSU ===\n")

    # Gerçek fiyatlar: 100k, 150k, 200k, 250k, 300k
    y_true = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
    
    # Modelin ürettiği tahmin fiyatları
    y_pred = np.array([105.0, 142.0, 210.0, 245.0, 290.0])

    print("Regresyon Veri Seti:")
    print(f"  Gerçek Ev Fiyatları (y_true) : {y_true}")
    print(f"  Tahmin Edilen Fiyatlar (y_pred): {y_pred}\n")

    reg_evaluator = RegressionEvaluator()

    print("--- Regresyon Metrik Motoru Çalıştırılıyor ---")
    metrics = reg_evaluator.calculate_metrics(y_true, y_pred)

    print("\n[RegressionEvaluator] Hesaplanan Hata Payları:")
    print(f"  MAE  (Mean Absolute Error)      : {metrics['mae']:.4f}")
    print(f"  --> Açıklama: Modelimiz tahminlerinde ortalama {metrics['mae']:.2f} birim sapma yapıyor.")
    
    print(f"  MSE  (Mean Squared Error)       : {metrics['mse']:.4f}")
    print(f"  --> Açıklama: Hataların karesi alındı. Büyük hataları cezalandırmak için kullanıldı.")
    
    print(f"  RMSE (Root Mean Squared Error)  : {metrics['rmse']:.4f}")
    print(f"  --> Açıklama: Standart sapma cinsinden ortalama hata payımız {metrics['rmse']:.2f} birimdir.")

if __name__ == "__main__":
    main()
