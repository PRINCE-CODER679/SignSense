"""
SignSense AI - ML Classifier Training & Benchmarking Unit Test Suite (Phase 4)
"""
import pytest
import numpy as np
import os
from pathlib import Path
from backend.ml.training.trainer import ClassifierTrainer
from backend.ml.config import FEATURE_COLUMNS, CLASSES, MODELS_DIR

def test_classifier_trainer_initialization():
    trainer = ClassifierTrainer(random_state=42)
    assert "random_forest" in trainer.models
    assert "svm" in trainer.models
    assert "knn" in trainer.models
    assert "logistic_regression" in trainer.models

def test_synthetic_model_fit_and_prediction():
    trainer = ClassifierTrainer(random_state=42)
    
    # Generate dummy training data (20 samples, 63 features, 3 classes: A, B, C)
    X_dummy = np.random.randn(30, 63).astype(np.float32)
    y_dummy = np.array(["A", "B", "C"] * 10)
    
    for name, model in trainer.models.items():
        model.fit(X_dummy, y_dummy)
        preds = model.predict(X_dummy[:3])
        assert len(preds) == 3
        assert all(p in ["A", "B", "C"] for p in preds)
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_dummy[:3])
            assert probs.shape == (3, 3)
            assert np.allclose(np.sum(probs, axis=1), 1.0)

def test_model_artifact_export_paths():
    assert MODELS_DIR.exists()
