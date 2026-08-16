from fastapi import APIRouter, HTTPException
import logging
from app.schemas.predict import LandmarkPredictionRequest, LandmarkPredictionResponse
from app.core.predictor import PredictorService

router = APIRouter()
logger = logging.getLogger("PredictEndpoint")

@router.post("/predict", response_model=LandmarkPredictionResponse, summary="ASL Real-time Landmark Inference Endpoint")
def predict_asl_gesture(payload: LandmarkPredictionRequest):
    """
    Accepts 21 3D MediaPipe hand landmark coordinates, normalizes features,
    and returns real-time ASL letter gesture prediction and probabilities.
    """
    if len(payload.landmarks) != 21:
        raise HTTPException(status_code=400, detail="Must provide exactly 21 hand landmarks.")

    try:
        predictor = PredictorService.get_instance()
        response = predictor.predict(
            landmarks=payload.landmarks,
            classifier_name=payload.classifier,
            handedness=payload.handedness
        )
        top5_str = ", ".join([f"{p.label}:{p.confidence*100:.1f}%" for p in response.top_probabilities[:5]])
        logger.info(f"Inference [{response.classifier_used}] (Handedness: {payload.handedness}): {response.predicted_letter} ({response.confidence*100:.1f}%) | Top 5: [{top5_str}]")
        return response
    except Exception as exc:
        logger.error(f"Inference error: {exc}")
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {str(exc)}")

