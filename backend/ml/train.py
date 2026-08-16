"""
SignSense AI - Machine Learning Classifier Benchmark & Training Pipeline (Phase 4)

Runs model training, 5-Fold Stratified Cross Validation, validation/test metrics computation,
and model artifact exports for Random Forest, SVM, KNN, and Logistic Regression classifiers.
"""
import sys
import logging
from pathlib import Path

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

try:
    from ml.training.trainer import train_and_benchmark_models
except ImportError:
    from backend.ml.training.trainer import train_and_benchmark_models

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    report = train_and_benchmark_models()
    print("\nPhase 4 Classifier Training & Benchmarking Complete!")
    print(f"Best Classifier: {report.get('best_model')}")


if __name__ == "__main__":
    main()

