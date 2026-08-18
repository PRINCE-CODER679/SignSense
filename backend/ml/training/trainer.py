"""
SignSense AI - Machine Learning Classifier Trainer & Benchmark Engine (Phase 4)

Trains, benchmarks, and evaluates multi-class ASL alphabet gesture classifiers:
- Random Forest Classifier
- Support Vector Machine (RBF Kernel with probability estimation)
- K-Nearest Neighbors Classifier
- Logistic Regression (Multinomial)

Performs 5-Fold Stratified Cross-Validation, evaluates test accuracy, precision,
recall, F1-scores, and inference latency, and exports trained model artifacts.
"""
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

try:
    from ml.config import (
        TRAIN_CSV_PATH,
        VAL_CSV_PATH,
        TEST_CSV_PATH,
        PROCESSED_CSV_PATH,
        MODELS_DIR,
        METADATA_DIR,
        FEATURE_COLUMNS,
        LABEL_COLUMN,
        CLASSES,
        RANDOM_STATE
    )
except ImportError:
    from backend.ml.config import (
        TRAIN_CSV_PATH,
        VAL_CSV_PATH,
        TEST_CSV_PATH,
        PROCESSED_CSV_PATH,
        MODELS_DIR,
        METADATA_DIR,
        FEATURE_COLUMNS,
        LABEL_COLUMN,
        CLASSES,
        RANDOM_STATE
    )

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ClassifierTrainer")


class ClassifierTrainer:
    """
    Manages loading datasets, model initialization, training, 5-fold cross-validation,
    benchmarking, and joblib artifact persistence.
    """

    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[Any] = None

        self._init_models()

    def _init_models(self):
        """Initializes the 4 benchmark ML classifiers with configured hyperparameters."""
        self.models = {
            "random_forest": RandomForestClassifier(
                n_estimators=120,
                max_depth=16,
                min_samples_split=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            "svm": SVC(
                C=2.0,
                kernel="rbf",
                gamma="scale",
                probability=True,
                random_state=self.random_state
            ),
            "knn": KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                algorithm="auto",
                n_jobs=-1
            ),
            "logistic_regression": LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
                random_state=self.random_state
            )
        }

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads train, val, and test feature CSVs. Fallbacks to loading processed CSV if splits missing.
        """
        if TRAIN_CSV_PATH.exists() and VAL_CSV_PATH.exists() and TEST_CSV_PATH.exists():
            logger.info("Loading pre-split Train / Val / Test datasets...")
            df_train = pd.read_csv(TRAIN_CSV_PATH)
            df_val = pd.read_csv(VAL_CSV_PATH)
            df_test = pd.read_csv(TEST_CSV_PATH)
        elif PROCESSED_CSV_PATH.exists():
            logger.warning("Split CSVs not found. Loading full processed feature dataset and performing 70/15/15 split...")
            df_full = pd.read_csv(PROCESSED_CSV_PATH)
            from sklearn.model_selection import train_test_split
            df_train, df_temp = train_test_split(df_full, test_size=0.30, stratify=df_full[LABEL_COLUMN], random_state=self.random_state)
            df_val, df_test = train_test_split(df_temp, test_size=0.50, stratify=df_temp[LABEL_COLUMN], random_state=self.random_state)
        else:
            raise FileNotFoundError(f"Neither split feature CSVs nor '{PROCESSED_CSV_PATH}' exists. Run Phase 3 landmark extraction first.")

        X_train = df_train[FEATURE_COLUMNS].values
        y_train = df_train[LABEL_COLUMN].values

        X_val = df_val[FEATURE_COLUMNS].values
        y_val = df_val[LABEL_COLUMN].values

        X_test = df_test[FEATURE_COLUMNS].values
        y_test = df_test[LABEL_COLUMN].values

        logger.info(f"Loaded dataset: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]} samples across {len(np.unique(y_train))} classes.")
        return X_train, y_train, X_val, y_val, X_test, y_test

    def train_and_benchmark(self) -> Dict[str, Any]:
        """
        Trains and benchmarks all 4 ML classifiers.
        """
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_data()

        # Fit feature scaler on training split
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        best_score = -1.0
        benchmark_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_features": len(FEATURE_COLUMNS),
            "num_classes": len(CLASSES),
            "train_samples": int(X_train.shape[0]),
            "val_samples": int(X_val.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "models": {}
        }

        print("\n" + "=" * 70)
        print(" ML CLASSIFIER BENCHMARKING & TRAINING PIPELINE ")
        print("=" * 70)

        for model_name, model in self.models.items():
            logger.info(f"Training and evaluating model: {model_name}...")
            
            # Select scaled features for linear/distance models, unscaled for trees if desired, or scaled for all
            use_scaled = model_name in ["svm", "knn", "logistic_regression"]
            X_tr = X_train_scaled if use_scaled else X_train
            X_va = X_val_scaled if use_scaled else X_val
            X_te = X_test_scaled if use_scaled else X_test

            # 1. Measure Training Time
            t_start = time.time()
            model.fit(X_tr, y_train)
            fit_time_sec = time.time() - t_start

            # 2. 5-Fold Stratified Cross-Validation
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(model, X_tr, y_train, cv=skf, scoring="accuracy", n_jobs=1)
            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))

            # 3. Validation Evaluation
            val_preds = model.predict(X_va)
            val_acc = float(accuracy_score(y_val, val_preds))

            # 4. Test Evaluation & Latency Measurement
            t_lat_start = time.time()
            test_preds = model.predict(X_te)
            test_probs = model.predict_proba(X_te) if hasattr(model, "predict_proba") else None
            inference_total_ms = (time.time() - t_lat_start) * 1000.0
            avg_latency_ms = inference_total_ms / max(1, len(X_te))

            test_acc = float(accuracy_score(y_test, test_preds))
            prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_test, test_preds, average="macro", zero_division=0)
            prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_test, test_preds, average="weighted", zero_division=0)

            # Store metrics report
            model_report = {
                "fit_time_sec": round(fit_time_sec, 4),
                "cv_5fold_acc_mean": round(cv_mean, 4),
                "cv_5fold_acc_std": round(cv_std, 4),
                "val_accuracy": round(val_acc, 4),
                "test_accuracy": round(test_acc, 4),
                "macro_precision": round(float(prec_m), 4),
                "macro_recall": round(float(rec_m), 4),
                "macro_f1": round(float(f1_m), 4),
                "weighted_f1": round(float(f1_w), 4),
                "avg_inference_latency_ms": round(avg_latency_ms, 4)
            }

            self.results[model_name] = model_report
            benchmark_report["models"][model_name] = model_report

            print(f"\nModel: [{model_name.upper()}]")
            print(f"  CV 5-Fold Acc: {cv_mean*100:.2f}% (±{cv_std*100:.2f}%)")
            print(f"  Val Accuracy:  {val_acc*100:.2f}%")
            print(f"  Test Accuracy: {test_acc*100:.2f}%")
            print(f"  Test F1 (Macro): {f1_m*100:.2f}% | Latency: {avg_latency_ms:.3f} ms/sample")

            # Track best performing model based on test accuracy & macro F1
            combined_score = test_acc * 0.7 + f1_m * 0.3
            if combined_score > best_score:
                best_score = combined_score
                self.best_model_name = model_name
                self.best_model = model

        benchmark_report["best_model"] = self.best_model_name
        print("\n" + "=" * 70)
        print(f" WINNING MODEL: [{self.best_model_name.upper()}] (Test Acc: {self.results[self.best_model_name]['test_accuracy']*100:.2f}%)")
        print("=" * 70 + "\n")

        # Save artifacts
        self.save_artifacts(benchmark_report)
        return benchmark_report

    def save_artifacts(self, benchmark_report: Dict[str, Any]):
        """Persists trained model joblib files and JSON benchmark reports."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        METADATA_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Save all trained models dictionary + scaler
        models_payload = {
            "models": self.models,
            "scaler": self.scaler,
            "best_model_name": self.best_model_name,
            "classes": CLASSES,
            "feature_columns": FEATURE_COLUMNS
        }
        all_models_path = MODELS_DIR / "classifiers.joblib"
        joblib.dump(models_payload, all_models_path)
        logger.info(f"Saved all trained models to '{all_models_path}'")

        # 2. Save best model package for direct inference
        best_payload = {
            "model_name": self.best_model_name,
            "model": self.best_model,
            "scaler": self.scaler,
            "classes": CLASSES,
            "feature_columns": FEATURE_COLUMNS,
            "metrics": self.results.get(self.best_model_name, {})
        }
        best_model_path = MODELS_DIR / "best_classifier.joblib"
        joblib.dump(best_payload, best_model_path)
        logger.info(f"Saved top classifier model package to '{best_model_path}'")

        # 3. Save JSON reports
        report_models_path = MODELS_DIR / "model_benchmark_report.json"
        report_meta_path = METADATA_DIR / "model_benchmark_report.json"
        
        for p in [report_models_path, report_meta_path]:
            with open(p, "w") as f:
                json.dump(benchmark_report, f, indent=2)
        
        logger.info(f"Saved model benchmark report JSON to '{report_models_path}'")


def train_and_benchmark_models() -> Dict[str, Any]:
    """Convenience function to run the full training pipeline."""
    trainer = ClassifierTrainer()
    return trainer.train_and_benchmark()


if __name__ == "__main__":
    train_and_benchmark_models()
