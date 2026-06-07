# 🤖 Custom Machine Learning Library from Scratch

This repository contains a modular, production-grade **Machine Learning and Data Science Library** built entirely from scratch using plain Python and NumPy. It was developed as a term project for the **Advanced Programming (YZM1022)** course at Yıldız Technical University (YTÜ).

The core objective of this library is to demonstrate clean software engineering practices by implementing end-to-end data pipelines, recursive model fitting, and multi-threaded cross-validation without relying on high-level frameworks like `scikit-learn`.

---

## 🏗️ Architectural Highlights & Design Patterns

The entire library is strictly designed around **SOLID Principles** to ensure loose coupling, high testability, and clean separation of concerns:

* **Polymorphic Model Interface (`BaseModel`):** All models inherit from an abstract structural contract, enforcing the implementation of `.fit()` and `.predict()`.
    * *Classification:* Features a custom **Recursive (Özyinelemeli)** `SimpleDecisionTree` built from scratch.
    * *Regression:* Features a vector-optimized `LinearRegressionFromScratch` solved via Gradient Descent loops.
* **The Facade Pattern (`MLDataManager`):** Hides the complexity of data ingestion. Users do not need to specify file extensions or call factories manually. A single static call (`MLDataManager.load("file.csv")`) automatically parses the extension and maps it to the appropriate polymorphic loader (`CSVDataLoader` or `JSONDataLoader`).
* **The Strategy Pattern (`DataProcessor`):** Isolates data imputation algorithms. Users can pass a `MeanImputer` or `MedianImputer` strategy. The processor dynamically scans the entire DataFrame, automatically locating and cleaning missing items across all numerical columns in one operation.
* **Template Method Pattern with Concurrency (`BaseEvaluator`):** To eliminate duplication and satisfy the **DRY (Don't Repeat Yourself)** principle, the K-Fold data splitting and threading engine is abstracted into the parent class. It leverages a `ThreadPoolExecutor` to train and grade independent validation folds **concurrently (asynchronously)** across multiple background threads.
* **Functional Programming (Closure):** Uses higher-order functions to dynamically generate stateful custom data transformers during the feature engineering stage.

---

## ⚙️ Installation Guide

The library utilizes modern Python packaging standards specified in `pyproject.toml`. Follow these steps to install the package cleanly on your local machine:

### 1. Clone the Repository
```bash
git clone [https://github.com/ayldzli/Machine_Learning.git](https://github.com/ayldzli/Machine_Learning.git)
cd Machine_Learning
