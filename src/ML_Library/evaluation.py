from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import List, Dict, Any

class BaseEvaluator(ABC):
    """Manages the common evaluation workflow."""
    
    def evaluate_pipeline(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Runs predictions and calculates metrics."""
        preds = model.predict(X)
        return self.calculate_metrics(y, preds)

    @abstractmethod
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Must be implemented by subclasses to calculate task-specific scores."""
        pass

    def parallel_cross_validation(self, model_factory: Any, X: np.ndarray, y: np.ndarray, n_folds: int = 3) -> List[Dict[str, float]]:
        """
        Splits data and trains models concurrently using threads.
        Works universally for both classification and regression models.
        """
        results = []
        fold_size = len(X) // n_folds

        with ThreadPoolExecutor(max_workers=n_folds) as executor:
            futures = []
            for i in range(n_folds):
                # Calculate index bounds for the validation fold
                val_start, val_end = i * fold_size, (i + 1) * fold_size
                X_val, y_val = X[val_start:val_end], y[val_start:val_end]
                
                # Combine remaining folds for the training set
                X_train = np.concatenate([X[:val_start], X[val_end:]], axis=0)
                y_train = np.concatenate([y[:val_start], y[val_end:]], axis=0)
                
                # Generate a clean model instance from the factory
                fresh_model = model_factory()
                
                # Submit the single fold job asynchronously to the background threads
                future = executor.submit(
                    self._evaluate_single_fold, i, fresh_model, X_train, y_train, X_val, y_val
                )
                futures.append(future)

            # Gather results as threads complete their operations
            for future in futures:
                results.append(future.result())
                
        return results

    def _evaluate_single_fold(self, fold_idx: int, model: Any, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Worker function run by background threads."""
        # Dynamically grabs the class name (ClassificationEvaluator or RegressionEvaluator) for clean logs
        evaluator_type = self.__class__.__name__
        print(f"[Thread] {evaluator_type} -> Fold {fold_idx} eğitimi başladı...")
        
        model.fit(X_train, y_train)
        return self.evaluate_pipeline(model, X_val, y_val)


class ClassificationEvaluator(BaseEvaluator):
    """Handles evaluation specific to categorical classification."""
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "accuracy": float(accuracy), "precision": float(precision),
            "recall": float(recall), "f1_score": float(f1_score)
        }


class RegressionEvaluator(BaseEvaluator):
    """Handles evaluation specific to continuous number regression."""

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        errors = y_true - y_pred
        mae = np.mean(np.abs(errors))
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        return {"mae": float(mae), "mse": float(mse), "rmse": float(rmse)}
