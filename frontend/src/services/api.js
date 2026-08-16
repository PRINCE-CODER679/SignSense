import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000,
});

export const checkBackendHealth = async () => {
  try {
    const response = await apiClient.get('/api/v1/health');
    return response.data;
  } catch (error) {
    console.error('Backend health check error:', error);
    return { status: 'offline', error: error.message };
  }
};

export const predictLandmarks = async (landmarks, classifier = 'random_forest', handedness = 'Right') => {
  if (!landmarks || !Array.isArray(landmarks) || landmarks.length !== 21) {
    console.warn('predictLandmarks called with invalid landmarks array (expected 21 points):', landmarks);
    return null;
  }

  try {
    const payload = {
      landmarks: landmarks.map(pt => ({
        x: Number(pt.x),
        y: Number(pt.y),
        z: Number(pt.z ?? 0.0)
      })),
      classifier: classifier,
      handedness: handedness
    };
    const response = await apiClient.post('/api/v1/predict', payload);
    if (response && response.data) {
      // console.log("Prediction received:", response.data);
      return response.data;
    }
    return null;
  } catch (error) {
    console.error('Prediction API error:', error.response?.data || error.message);
    return null;
  }
};

