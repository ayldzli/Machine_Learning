from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """
    Base model for all machine learning models in library.
    """
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Abstract method to train the model from scratch."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Abstract method to generate predictions from learned parameters."""
        pass


class DecisionNode:
    """Helper data container representing a single node/leaf in the tree structure."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class SimpleDecisionTree(BaseModel):
    """Recursive Classification Model"""
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Recursive tree training."""
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> DecisionNode:
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            most_common_value = np.bincount(y).argmax() if len(y) > 0 else 0
            return DecisionNode(value=most_common_value)

        feature_idx = 0
        threshold = np.median(X[:, feature_idx])

        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return DecisionNode(feature=feature_idx, threshold=threshold, left=left_child, right=right_child)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Recursive row traversal."""
        return np.array([self._predict_row(self.root, row) for row in X])

    def _predict_row(self, node: DecisionNode, row: np.ndarray) -> float:
        if node.value is not None:
            return node.value
        if row[node.feature] <= node.threshold:
            return self._predict_row(node.left, row)
        return self._predict_row(node.right, row)


class LinearRegressionFromScratch(BaseModel):
    """Linear Regression Model."""
    def __init__(self, learning_rate=0.01, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Solves linear coefficients."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generates matrix dot-product predictions."""
        return np.dot(X, self.weights) + self.bias
