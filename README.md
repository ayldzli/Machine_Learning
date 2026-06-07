# 🤖 Simple Machine Learning Library

This repository contains a modular, production-grade **Machine Learning and Data Science Library** built entirely from scratch using plain Python and NumPy. It was developed as a term project for the **Advanced Programming (YZM1022)** course at Yıldız Technical University (YTÜ).

The core objective of this library is to demonstrate clean software engineering practices by implementing end-to-end data pipelines, recursive model fitting, and multi-threaded cross-validation without relying on high-level frameworks like `scikit-learn`.

## Repository Directory Structure

```text
Machine_Learning/
│
├── pyproject.toml              # Modern package build configuration
├── README.md                   # Documentation frontpage
│
├── src/
│   └── my_ml_library/         # Main library package
│       ├── __init__.py         # Public API exposure mappings
│       ├── data_loader.py      # Facade and Factory ingestion engines
│       ├── processing.py       # Strategy-based data preprocessing
│       ├── models.py           # Polymorphic classification & regression models
│       └── evaluation.py       # Template-based concurrent K-Fold evaluator
│
├── examples/                   # Executable entry points for evaluation
│   ├── data/
│   │   └── classification_data.csv # Tabular dataset for classification testing
│   ├── demo_classification.py  # End-to-end classification pipeline
│   └── demo_regression.py      # End-to-end regression evaluation
│
└── tests/                      # Core automated test suite
    ├── data/
    │   └── test_clf_temp.csv   # Temporary mockup dataset for classification testing
    ├── test_classification.py
    └── test_regression.py

## Architectural Highlights & Design Patterns

The entire library is strictly designed around **SOLID Principles** to ensure loose coupling, high testability, and clean separation of concerns:

* **Polymorphic Model Interface (`BaseModel`):** All models inherit from an abstract structural contract, enforcing the implementation of `.fit()` and `.predict()`.
    * *Classification:* Features a custom **Recursive (Özyinelemeli)** `SimpleDecisionTree` built from scratch.
    * *Regression:* Features a vector-optimized `LinearRegressionFromScratch` solved via Gradient Descent loops.
* **The Facade Pattern (`MLDataManager`):** Hides the complexity of data ingestion. Users do not need to specify file extensions or call factories manually. A single static call (`MLDataManager.load("file.csv")`) automatically parses the extension and maps it to the appropriate polymorphic loader (`CSVDataLoader` or `JSONDataLoader`).
* **The Strategy Pattern (`DataProcessor`):** Isolates data imputation algorithms. Users can pass a `MeanImputer` or `MedianImputer` strategy. The processor dynamically scans the entire DataFrame, automatically locating and cleaning missing items across all numerical columns in one operation.
* **Template Method Pattern with Concurrency (`BaseEvaluator`):** To eliminate duplication and satisfy the **DRY (Don't Repeat Yourself)** principle, the K-Fold data splitting and threading engine is abstracted into the parent class. It leverages a `ThreadPoolExecutor` to train and grade independent validation folds **concurrently (asynchronously)** across multiple background threads.
* **Functional Programming (Closure):** Uses higher-order functions to dynamically generate stateful custom data transformers during the feature engineering stage.

---

## Installation Guide

The library utilizes modern Python packaging standards specified in `pyproject.toml`. Follow these steps to install the package cleanly on your local machine:

### 1. Clone the Repository
```bash
git clone [https://github.com/ayldzli/Machine_Learning.git](https://github.com/ayldzli/Machine_Learning.git)
cd Machine_Learning
```

### 2. Install the Library in Editable Mode

Install the package locally along with all its required dependencies (numpy, pandas, pytest) using the developer development flag (-e):
```bash
pip install -e .
```

## Examples

### 1. Classification Pipeline
Loads tabular classification data, automatically cleans all missing fields, handles feature transformations via closures, and triggers parallel evaluation threads to evaluate the recursive Decision Tree model:

```bash
python3 examples/classification_example.py
```

### 2. Regression Pipeline
Loads numerical datasets, fits a multivariate linear regression equation using raw matrix gradient descent optimization, and calculates performance error metrics (MAE, MSE, RMSE) concurrently:

```bash
python3 examples/regression_example.py
```

## Tests

To run the automated tests, ensure pytest is installed and run the command from the root directory:

```bash
pytest -v
```

