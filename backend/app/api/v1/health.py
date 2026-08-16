from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()

@router.get("/health", summary="Backend Health Check")
def health_check():
    return {
        "status": "ok",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": "development"
    }
