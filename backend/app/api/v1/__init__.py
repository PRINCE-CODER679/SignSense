from fastapi import APIRouter
from app.api.v1.health import router as health_router
from app.api.v1.predict import router as predict_router

api_v1_router = APIRouter()
api_v1_router.include_router(health_router, tags=["Health Check"])
api_v1_router.include_router(predict_router, tags=["Inference Engine"])
