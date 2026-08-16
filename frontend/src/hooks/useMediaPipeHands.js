import { useEffect, useRef, useState, useCallback } from 'react';
import { Hands } from '@mediapipe/hands';

/**
 * Custom React Hook for Real-time MediaPipe Hand Tracking
 * 
 * Manages webcam media streams, MediaPipe WASM lifecycle, frame loop,
 * hand detection status, and high-performance landmark extraction.
 */
export function useMediaPipeHands() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  
  // High-frequency data stored in refs to avoid 60fps React state re-renders
  const landmarksRef = useRef(null);
  const handednessRef = useRef('Right');
  const handsInstanceRef = useRef(null);
  const animationFrameIdRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const isRunningRef = useRef(false);


  // Performance metrics stored in refs for lightweight UI sampling
  const fpsRef = useRef(0);
  const frameLatencyRef = useRef(0);
  const lastFrameTimeRef = useRef(performance.now());
  const frameCountRef = useRef(0);

  // Low-frequency UI state (only updated on status transitions)
  const [cameraState, setCameraState] = useState('OFFLINE'); // 'OFFLINE' | 'CONNECTING' | 'ONLINE' | 'PERMISSION_DENIED' | 'UNAVAILABLE'
  const [trackingState, setTrackingState] = useState('INACTIVE'); // 'INACTIVE' | 'INITIALIZING' | 'ACTIVE' | 'ERROR'
  const [handDetected, setHandDetected] = useState(false);
  const [detectedHandCount, setDetectedHandCount] = useState(0);
  const [errorMessage, setErrorMessage] = useState(null);
  const [fps, setFps] = useState(0);
  const [latency, setLatency] = useState(0);
  const [isMirrored, setIsMirrored] = useState(true);
  const [showSkeletonOverlay, setShowSkeletonOverlay] = useState(true);

  // 1. Initialize MediaPipe Hands Instance
  const initMediaPipe = useCallback(async () => {
    if (handsInstanceRef.current) return handsInstanceRef.current;

    try {
      setTrackingState('INITIALIZING');
      const hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
      });

      hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });

      handsInstanceRef.current = hands;
      setTrackingState('ACTIVE');
      return hands;
    } catch (err) {
      console.error('Failed to initialize MediaPipe Hands:', err);
      setTrackingState('ERROR');
      setErrorMessage(`MediaPipe Initialization Error: ${err.message || 'Failed to load tracking WASM model'}`);
      return null;
    }
  }, []);

  const isProcessingRef = useRef(false);

  // 2. Continuous Video Frame Processing Loop
  const processFrame = useCallback(async () => {
    if (!isRunningRef.current || !videoRef.current || !handsInstanceRef.current) return;
    if (isProcessingRef.current) {
      if (isRunningRef.current) {
        animationFrameIdRef.current = requestAnimationFrame(processFrame);
      }
      return;
    }

    isProcessingRef.current = true;
    const video = videoRef.current;

    // Ensure video elements have valid dimensions and frame data
    if (video && video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
      const startTime = performance.now();

      try {
        await handsInstanceRef.current.send({ image: video });
      } catch (err) {
        console.error('Error sending video frame to MediaPipe:', err);
      }

      // Calculate latency
      const processDuration = performance.now() - startTime;
      frameLatencyRef.current = Math.round(processDuration);

      // Calculate FPS
      frameCountRef.current += 1;
      const now = performance.now();
      const delta = now - lastFrameTimeRef.current;
      if (delta >= 1000) {
        fpsRef.current = Math.round((frameCountRef.current * 1000) / delta);
        setFps(fpsRef.current);
        setLatency(frameLatencyRef.current);
        frameCountRef.current = 0;
        lastFrameTimeRef.current = now;
      }
    }

    isProcessingRef.current = false;

    if (isRunningRef.current) {
      animationFrameIdRef.current = requestAnimationFrame(processFrame);
    }
  }, []);

  // 3. Start Camera and MediaPipe Loop
  const startCamera = useCallback(async () => {
    if (isRunningRef.current) return;
    setErrorMessage(null);
    setCameraState('CONNECTING');

    // Ensure MediaPipe is ready
    const hands = await initMediaPipe();
    if (!hands) return;

    // MediaPipe Results Callback
    hands.onResults((results) => {
      const hasHand = Boolean(
        results.multiHandLandmarks && results.multiHandLandmarks.length > 0
      );

      if (hasHand) {
        const handLandmarks = results.multiHandLandmarks[0];
        landmarksRef.current = handLandmarks;
        handednessRef.current = results.multiHandedness?.[0]?.label || 'Right';

        setHandDetected((prev) => {
          if (!prev) return true;
          return prev;
        });
        setDetectedHandCount(results.multiHandLandmarks.length);
      } else {
        landmarksRef.current = null;
        handednessRef.current = 'Right';
        setHandDetected((prev) => {
          if (prev) return false;
          return prev;
        });
        setDetectedHandCount(0);
      }
    });

    // Request getUserMedia webcam stream
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Webcam mediaDevices API is not supported in this browser environment.');
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        },
        audio: false
      });

      mediaStreamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        if (videoRef.current.readyState >= 1) {
          await videoRef.current.play().catch(() => {});
        } else {
          await new Promise((resolve) => {
            videoRef.current.onloadedmetadata = () => {
              videoRef.current.play().then(resolve).catch(resolve);
            };
          });
        }
      }

      isRunningRef.current = true;
      setCameraState('ONLINE');

      // Start animation loop
      animationFrameIdRef.current = requestAnimationFrame(processFrame);

    } catch (err) {
      console.error('Camera access error:', err);
      isRunningRef.current = false;
      
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setCameraState('PERMISSION_DENIED');
        setErrorMessage('Camera access denied. Please click the camera lock icon in your browser URL bar to grant permission.');
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        setCameraState('UNAVAILABLE');
        setErrorMessage('No camera device detected. Please connect a webcam and try again.');
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        setCameraState('UNAVAILABLE');
        setErrorMessage('Camera is currently in use by another application.');
      } else {
        setCameraState('UNAVAILABLE');
        setErrorMessage(`Camera Error: ${err.message || 'Unable to start camera stream'}`);
      }
    }
  }, [initMediaPipe, processFrame]);

  // 4. Stop Camera and Cleanup
  const stopCamera = useCallback(() => {
    isRunningRef.current = false;

    if (animationFrameIdRef.current) {
      cancelAnimationFrame(animationFrameIdRef.current);
      animationFrameIdRef.current = null;
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    landmarksRef.current = null;
    setCameraState('OFFLINE');
    setHandDetected(false);
    setDetectedHandCount(0);
    setFps(0);
    setLatency(0);
  }, []);

  // Cleanup on Component Unmount
  useEffect(() => {
    return () => {
      stopCamera();
      if (handsInstanceRef.current) {
        handsInstanceRef.current.close().catch(() => {});
        handsInstanceRef.current = null;
      }
    };
  }, [stopCamera]);

  return {
    videoRef,
    canvasRef,
    landmarksRef,
    handednessRef,
    cameraState,
    trackingState,
    handDetected,
    detectedHandCount,
    errorMessage,
    fps,
    latency,
    isMirrored,
    setIsMirrored,
    showSkeletonOverlay,
    setShowSkeletonOverlay,
    startCamera,
    stopCamera
  };
}
