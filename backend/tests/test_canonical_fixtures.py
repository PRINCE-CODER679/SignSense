import pytest
import numpy as np
from app.core.predictor import PredictorService
from ml.scripts.generate_asl_dataset import build_base_hand_skeleton, CLASSES
from app.schemas.predict import LandmarkPoint

def test_canonical_asl_fixtures_high_confidence():
    """
    Passes known canonical ASL landmark fixtures for letters A, B, C, D, L, V, Y
    through the normalizer and asserts that the classifier returns the correct letter
    with >90% confidence.
    """
    predictor = PredictorService.get_instance()
    test_letters = ['A', 'B', 'C', 'D', 'L', 'V', 'Y']

    for letter in test_letters:
        raw_coords_21 = build_base_hand_skeleton(letter, variant=0)
        landmarks = [LandmarkPoint(x=float(pt[0]), y=float(pt[1]), z=float(pt[2])) for pt in raw_coords_21]

        response = predictor.predict(landmarks=landmarks, classifier_name="random_forest", handedness="Right")
        
        assert response.predicted_letter == letter, (
            f"Expected letter '{letter}', but predicted '{response.predicted_letter}' "
            f"with confidence {response.confidence*100:.1f}%"
        )
        assert response.confidence >= 0.90, (
            f"Letter '{letter}' confidence too low: {response.confidence*100:.1f}% (expected >= 90.0%)"
        )
        assert len(response.top_probabilities) >= 5, (
            f"Expected at least 5 top probabilities, received {len(response.top_probabilities)}"
        )

def test_left_hand_mirroring_invariance():
    """
    Tests that horizontally flipped (Left Hand) landmark fixtures pass through handedness flip
    and yield the correct letter with >90% confidence.
    """
    predictor = PredictorService.get_instance()
    test_letters = ['A', 'B', 'C', 'D', 'L', 'V', 'Y']

    for letter in test_letters:
        raw_coords_21 = build_base_hand_skeleton(letter, variant=0)
        
        # Simulate Left Hand image coordinates by inverting X around palm center
        left_coords = raw_coords_21.copy()
        wrist_x = left_coords[0, 0]
        left_coords[:, 0] = 2.0 * wrist_x - left_coords[:, 0]

        landmarks_left = [LandmarkPoint(x=float(pt[0]), y=float(pt[1]), z=float(pt[2])) for pt in left_coords]

        # 1. Test with explicit handedness="Left"
        response_explicit = predictor.predict(landmarks=landmarks_left, classifier_name="random_forest", handedness="Left")
        assert response_explicit.predicted_letter == letter, (
            f"Explicit Left hand simulation expected '{letter}', got '{response_explicit.predicted_letter}'"
        )
        assert response_explicit.confidence >= 0.90, (
            f"Explicit Left hand letter '{letter}' confidence too low: {response_explicit.confidence*100:.1f}%"
        )

        # 2. Test with auto-detection fallback (handedness=None)
        response_auto = predictor.predict(landmarks=landmarks_left, classifier_name="random_forest", handedness=None)
        assert response_auto.predicted_letter == letter, (
            f"Auto-detected Left hand simulation expected '{letter}', got '{response_auto.predicted_letter}'"
        )
        assert response_auto.confidence >= 0.90, (
            f"Auto-detected Left hand letter '{letter}' confidence too low: {response_auto.confidence*100:.1f}%"
        )

