"""
SignSense AI - Real-time Webcam Landmark Collector (Phase 3 Utility)

Interactive OpenCV + MediaPipe workflow for recording real human hand landmarks.
- Displays target letter and recorded sample count
- Toggle collection via [SPACE] key
- Press [N] for Next letter, [P] for Previous letter
- Rejects frames where no hand is detected
- Reuses exact normalize_landmarks preprocessing pipeline
- Appends 63D normalized vectors to backend/ml/data/processed/asl_features.csv
"""
import sys
import time
import logging
from pathlib import Path

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import cv2
import pandas as pd
from ml.config import (
    PROCESSED_CSV_PATH,
    CLASSES,
    FEATURE_COLUMNS,
    LABEL_COLUMN
)
from ml.preprocessing.extractor import LandmarkExtractor
from ml.preprocessing.normalizer import normalize_landmarks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LandmarkRecorder")


def run_webcam_collector(target_samples_per_class: int = 100):
    """
    Launches interactive webcam collection window for recording ASL letters A-Z.
    """
    print("\n" + "=" * 65)
    print(" SIGNSENSE AI - INTERACTIVE WEBCAM LANDMARK RECORDING UTILITY ")
    print("=" * 65)
    print(" Controls:")
    print("  [SPACE] - Toggle Recording ON/OFF")
    print("  [N]     - Next Letter")
    print("  [P]     - Previous Letter")
    print("  [Q/ESC] - Quit and Save Dataset")
    print("=" * 65 + "\n")

    current_class_idx = 0
    is_recording = False
    class_counts = {c: 0 for c in CLASSES}

    # Load existing processed CSV if present
    existing_df = None
    if PROCESSED_CSV_PATH.exists():
        try:
            existing_df = pd.read_csv(PROCESSED_CSV_PATH)
            if LABEL_COLUMN in existing_df.columns:
                counts = existing_df[LABEL_COLUMN].value_counts().to_dict()
                for c in CLASSES:
                    class_counts[c] = counts.get(c, 0)
                logger.info(f"Loaded existing dataset from '{PROCESSED_CSV_PATH}' with {len(existing_df)} samples.")
        except Exception as e:
            logger.warning(f"Could not parse existing CSV: {e}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam video stream.")
        return

    recorded_rows = []

    with LandmarkExtractor(static_image_mode=False, max_num_hands=1) as extractor:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.error("Failed to read frame from webcam.")
                break

            # Mirror frame horizontally for intuitive interaction
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            target_letter = CLASSES[current_class_idx]
            current_count = class_counts[target_letter]

            # Extract 21 landmarks
            success, landmarks_xyz, reason = extractor.extract_from_frame(frame)

            status_color = (0, 0, 255) # Red if no hand
            status_text = f"HAND: NOT DETECTED ({reason})"

            if success and landmarks_xyz is not None:
                status_color = (0, 255, 0) # Green if detected
                status_text = "HAND: DETECTED (21 Landmarks)"

                # If recording is active, normalize & store
                if isRecording:
                    try:
                        feat_63 = normalize_landmarks(landmarks_xyz)
                        row_dict = {LABEL_COLUMN: target_letter}
                        for col, val in zip(FEATURE_COLUMNS, feat_63):
                            row_dict[col] = float(val)
                        recorded_rows.append(row_dict)
                        class_counts[target_letter] += 1
                        time.sleep(0.05) # 20Hz sampling rate limit
                    except Exception as exc:
                        logger.error(f"Normalization error: {exc}")

            # Draw HUD Overlays
            cv2.rectangle(display_frame, (10, 10), (450, 130), (0, 0, 0), -1)
            cv2.putText(display_frame, f"TARGET LETTER: {target_letter}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            cv2.putText(display_frame, f"SAMPLES: {current_count} / {target_samples_per_class}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            rec_str = "[RECORDING ACTIVE]" if is_recording else "[RECORDING PAUSED]"
            rec_col = (0, 255, 0) if is_recording else (0, 165, 255)
            cv2.putText(display_frame, rec_str, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rec_col, 2)
            cv2.putText(display_frame, status_text, (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

            cv2.imshow("SignSense AI - Landmark Collector", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 32: # SPACE
                is_recording = not is_recording
            elif key == ord('n') or key == ord('N'):
                current_class_idx = (current_class_idx + 1) % len(CLASSES)
            elif key == ord('p') or key == ord('P'):
                current_class_idx = (current_class_idx - 1) % len(CLASSES)
            elif key == ord('q') or key == ord('Q') or key == 27: # ESC
                break

    cap.release()
    cv2.destroyAllWindows()

    if recorded_rows:
        new_df = pd.DataFrame(recorded_rows)
        new_df = new_df[[LABEL_COLUMN] + FEATURE_COLUMNS]
        if existing_df is not None:
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            final_df = new_df

        PROCESSED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(PROCESSED_CSV_PATH, index=False)
        logger.info(f"Saved {len(new_df)} new landmark samples to '{PROCESSED_CSV_PATH}'. Total: {len(final_df)}")


if __name__ == "__main__":
    run_webcam_collector()
