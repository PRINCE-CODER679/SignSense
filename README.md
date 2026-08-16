# SignSense AI 🤟✨

**SignSense AI** is a real-time American Sign Language (ASL) alphabet & fingerspelling recognition web application.

It extracts 21 3D hand landmarks via MediaPipe, normalizes features into scale- and origin-invariant 63-dimensional vectors, and uses trained machine learning classifiers to predict letters A–Z in real time.

---

## 🏗️ Architecture & Stack

### **Frontend (`/frontend`)**
- **Framework**: React 18 + Vite
- **Styling**: Tailwind CSS (Dark theme, neon accents, glassmorphism)
- **Animations**: Anime.js
- **Background**: tsParticles (`@tsparticles/react`)
- **Webcam**: Browser `getUserMedia()` API
- **Deployment**: Vercel

### **Backend (`/backend`)**
- **Framework**: Python 3.13 + FastAPI
- **Computer Vision**: OpenCV + MediaPipe
- **Machine Learning**: scikit-learn (Random Forest, SVM, KNN, Logistic Regression)
- **Deployment**: Render

---

## 🚀 Quick Start Guide

### 1. Launch Backend (FastAPI)
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
- API Docs: `http://localhost:8000/docs`
- Health Endpoint: `http://localhost:8000/api/v1/health`

### 2. Launch Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev
```
- Local URL: `http://localhost:5173`

---

## 📅 Implementation Roadmap (10 Phases)

- [x] **Phase 1**: Project foundation, UI architecture, and FastAPI skeleton
- [x] **Phase 2**: Webcam + MediaPipe hand tracking integration
- [x] **Phase 3**: Dataset generation, landmark feature extraction & normalization
- [x] **Phase 4**: Train and benchmark ML classifiers (Random Forest, SVM, KNN, LogReg)
- [x] **Phase 5**: Real-time prediction engine build
- [x] **Phase 6**: FastAPI ML inference API endpoints
- [x] **Phase 7**: Integrate prediction engine into React UI
- [x] **Phase 8**: Word builder and practice mode
- [x] **Phase 9**: Premium UI, Anime.js animations & tsParticles polish
- [ ] **Phase 10**: Testing, production build & deployment (Vercel + Render)


