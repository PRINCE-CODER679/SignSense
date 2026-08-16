# SignSense AI - FastAPI Backend Service

Production-ready FastAPI backend providing health checks and machine learning inference endpoints for real-time ASL alphabet recognition.

## Running Locally

1. Create Python virtual environment and install dependencies:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

2. Start FastAPI development server:
```bash
uvicorn app.main:app --reload --port 8000
```

3. Access interactive Swagger API documentation:
Open browser at `http://localhost:8000/docs`.
