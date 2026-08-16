import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "SignSense AI Backend")
    VERSION: str = os.getenv("VERSION", "1.0.0")
    API_V1_STR: str = "/api/v1"
    
    # CORS Origins
    CORS_ORIGINS: list[str] = [
        origin.strip() for origin in os.getenv(
            "CORS_ORIGINS", 
            "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000"
        ).split(",")
    ]

settings = Settings()
