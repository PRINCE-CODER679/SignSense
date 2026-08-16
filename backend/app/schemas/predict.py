from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class LandmarkPoint(BaseModel):
    x: float = Field(..., description="Normalized X coordinate")
    y: float = Field(..., description="Normalized Y coordinate")
    z: float = Field(..., description="Normalized Z coordinate")

class LandmarkPredictionRequest(BaseModel):
    landmarks: List[LandmarkPoint] = Field(..., min_length=21, max_length=21, description="21 MediaPipe hand landmark points")
    classifier: Optional[str] = Field("random_forest", description="ML model selection (random_forest, svm, knn, logistic_regression)")
    handedness: Optional[str] = Field("Right", description="Handedness of detected hand ('Left' or 'Right')")

class PredictionProbability(BaseModel):
    label: str
    confidence: float

class LandmarkPredictionResponse(BaseModel):
    predicted_letter: str
    confidence: float
    classifier_used: str
    top_probabilities: List[PredictionProbability]
    processing_time_ms: float
