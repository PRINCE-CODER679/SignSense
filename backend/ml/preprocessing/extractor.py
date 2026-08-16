"""
SignSense AI - MediaPipe Hand Landmark Extractor (Phase 3)

Loads image samples, runs MediaPipe Hands detection, and extracts 21 3D landmarks (x, y, z).
Supports both MediaPipe Solutions API (Python <=3.12) and MediaPipe Tasks API (Python 3.13+).
Records invalid/skipped samples with detailed diagnostic logs.
"""
import logging
import urllib.request
from pathlib import Path
from typing import Optional, Tuple, List, Union
import cv2
import numpy as np
import mediapipe as mp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model asset location for MediaPipe Tasks API
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"


def ensure_model_asset() -> Path:
    """Downloads hand_landmarker.task if not present locally."""
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading MediaPipe Hand Landmarker model to '{MODEL_PATH}'...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        logger.info("MediaPipe Hand Landmarker model downloaded successfully.")
    return MODEL_PATH


class LandmarkExtractor:
    """
    Reusable MediaPipe Hands extraction pipeline supporting both Solutions & Tasks APIs.
    """

    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initializes MediaPipe Hands detector.
        """
        self.mode = None
        self.detector = None

        # 1. Try legacy mp.solutions.hands API if available
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            try:
                self.mp_hands = mp.solutions.hands
                self.detector = self.mp_hands.Hands(
                    static_image_mode=static_image_mode,
                    max_num_hands=max_num_hands,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                self.mode = "SOLUTIONS"
            except Exception as e:
                logger.debug(f"MediaPipe Solutions initialization fallback: {e}")

        # 2. Fallback to MediaPipe Tasks API (Python 3.13+)
        if self.detector is None:
            try:
                from mediapipe.tasks.python import vision
                from mediapipe.tasks.python.core.base_options import BaseOptions

                model_file = ensure_model_asset()
                base_options = BaseOptions(model_asset_path=str(model_file))
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=max_num_hands,
                    min_hand_detection_confidence=min_detection_confidence,
                    min_hand_presence_confidence=min_tracking_confidence,
                )
                self.detector = vision.HandLandmarker.create_from_options(options)
                self.mode = "TASKS"
            except Exception as e:
                logger.error(f"Failed to initialize MediaPipe HandLandmarker: {e}")
                raise RuntimeError("Could not initialize any MediaPipe Hand detector backend.") from e

    def extract_from_image_path(
        self, image_path: Union[str, Path]
    ) -> Tuple[bool, Optional[List[Tuple[float, float, float]]], str]:
        """
        Loads image file from disk and extracts 21 3D hand landmarks.

        Args:
            image_path: Path to target image file.

        Returns:
            Tuple of (success: bool, landmarks: Optional[List[(x,y,z)]], reason: str)
        """
        path_obj = Path(image_path)
        if not path_obj.exists():
            return False, None, f"FILE_NOT_FOUND: {path_obj}"

        image_bgr = cv2.imread(str(path_obj))
        if image_bgr is None or image_bgr.size == 0:
            return False, None, f"IMAGE_READ_FAILED: {path_obj.name}"

        return self.extract_from_frame(image_bgr)

    def extract_from_frame(
        self, frame_bgr: np.ndarray
    ) -> Tuple[bool, Optional[List[Tuple[float, float, float]]], str]:
        """
        Processes BGR image array and extracts 21 3D hand landmarks.

        Args:
            frame_bgr: BGR image numpy array.

        Returns:
            Tuple of (success: bool, landmarks: Optional[List[(x,y,z)]], reason: str)
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return False, None, "EMPTY_FRAME_PROVIDED"

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        landmarks_xyz: List[Tuple[float, float, float]] = []

        if self.mode == "SOLUTIONS":
            results = self.detector.process(frame_rgb)
            if not results.multi_hand_landmarks:
                return False, None, "NO_HAND_DETECTED"

            hand_landmarks = results.multi_hand_landmarks[0]
            for lm in hand_landmarks.landmark:
                landmarks_xyz.append((float(lm.x), float(lm.y), float(lm.z)))

        elif self.mode == "TASKS":
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = self.detector.detect(mp_image)

            if not results.hand_landmarks:
                return False, None, "NO_HAND_DETECTED"

            hand_landmarks = results.hand_landmarks[0]
            for lm in hand_landmarks:
                landmarks_xyz.append((float(lm.x), float(lm.y), float(lm.z)))

        if len(landmarks_xyz) != 21:
            return False, None, f"INCOMPLETE_LANDMARKS_COUNT_{len(landmarks_xyz)}"

        return True, landmarks_xyz, "SUCCESS"

    def close(self):
        """Releases MediaPipe detector resources."""
        if self.detector:
            try:
                self.detector.close()
            except Exception:
                pass
            self.detector = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
