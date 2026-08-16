import { useEffect, useRef, useState } from 'react';
import anime from 'animejs';
import { 
  Camera, 
  CameraOff, 
  RefreshCw, 
  AlertTriangle, 
  CheckCircle2, 
  XCircle, 
  Activity, 
  Sliders, 
  Eye, 
  EyeOff, 
  FlipHorizontal,
  Info,
  Sparkles,
  Layers,
  Brain
} from 'lucide-react';
import { useMediaPipeHands } from '../hooks/useMediaPipeHands';
import { drawNeonHandLandmarks } from '../utils/handDrawer';
import { predictLandmarks } from '../services/api';
import { classifyASLGeometry } from '../utils/aslGeometry';

export default function Recognition() {
  const {
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
  } = useMediaPipeHands();

  const [inspectorLandmarks, setInspectorLandmarks] = useState(null);
  const [selectedClassifier, setSelectedClassifier] = useState('deterministic_geometry');
  const [predictionResult, setPredictionResult] = useState(null);
  const [isInferring, setIsInferring] = useState(false);

  const prevPredictedLetterRef = useRef(null);
  const letterBadgeRef = useRef(null);

  // Entrance animation for page panels
  useEffect(() => {
    anime({
      targets: '.rec-panel-animate',
      translateY: [20, 0],
      opacity: [0, 1],
      delay: anime.stagger(100),
      duration: 700,
      easing: 'easeOutExpo'
    });
  }, []);

  // Auto-start camera when page mounts
  useEffect(() => {
    startCamera();
  }, [startCamera]);

  // Anime.js trigger when predicted letter changes
  useEffect(() => {
    if (predictionResult && predictionResult.predicted_letter !== prevPredictedLetterRef.current) {
      prevPredictedLetterRef.current = predictionResult.predicted_letter;
      if (letterBadgeRef.current) {
        anime({
          targets: letterBadgeRef.current,
          scale: [0.8, 1.08, 1],
          duration: 350,
          easing: 'easeOutBack'
        });
      }
    }
  }, [predictionResult]);

  // Canvas drawing loop synchronized with animation frames
  useEffect(() => {
    let animId;

    const renderCanvas = () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (canvas && video && video.readyState >= 2 && video.videoWidth > 0) {
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }

        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);

          if (showSkeletonOverlay && landmarksRef.current) {
            drawNeonHandLandmarks(ctx, landmarksRef.current, canvas.width, canvas.height, isMirrored);
          }
        }
      }

      animId = requestAnimationFrame(renderCanvas);
    };

    animId = requestAnimationFrame(renderCanvas);
    return () => cancelAnimationFrame(animId);
  }, [videoRef, canvasRef, landmarksRef, showSkeletonOverlay, isMirrored]);

  // Real-time ML Prediction Inference Loop (throttled to 10Hz)
  useEffect(() => {
    let isSubscribed = true;

    const inferenceInterval = setInterval(async () => {
      if (landmarksRef.current && cameraState === 'ONLINE') {
        setIsInferring(true);
        let res = null;
        if (selectedClassifier === 'deterministic_geometry') {
          res = classifyASLGeometry(landmarksRef.current, handednessRef.current);
        } else {
          res = await predictLandmarks(landmarksRef.current, selectedClassifier, handednessRef.current);
        }
        if (isSubscribed && res) {
          setPredictionResult(res);
        }
        if (isSubscribed) setIsInferring(false);
      } else if (!landmarksRef.current && predictionResult) {
        setPredictionResult(null);
      }
    }, 120);

    return () => {
      isSubscribed = false;
      clearInterval(inferenceInterval);
    };
  }, [landmarksRef, cameraState, selectedClassifier]);

  // Periodic landmark inspector sampler (throttled to 5Hz)
  useEffect(() => {
    const interval = setInterval(() => {
      if (landmarksRef.current) {
        setInspectorLandmarks([...landmarksRef.current]);
      } else {
        setInspectorLandmarks(null);
      }
    }, 200);

    return () => clearInterval(interval);
  }, [landmarksRef]);

  const keyNodes = [
    { index: 0, name: 'Wrist (Anchor)' },
    { index: 4, name: 'Thumb Tip' },
    { index: 8, name: 'Index Tip' },
    { index: 12, name: 'Middle Tip' },
    { index: 16, name: 'Ring Tip' },
    { index: 20, name: 'Pinky Tip' },
  ];

  return (
    <div className="space-y-8 py-6">
      {/* Header & Page Title */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-brand-border/60 pb-5">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight flex items-center gap-2.5">
            <Camera className="w-8 h-8 text-brand-primary" /> AI Recognition Control Center
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Real-time 21 hand landmark extraction via MediaPipe WASM and multi-model machine learning inference
          </p>
        </div>

        {/* Quick View Controls */}
        <div className="flex items-center gap-2 bg-brand-surface/90 p-1.5 rounded-2xl border border-brand-border/80 text-xs shadow-inner">
          <button
            onClick={() => setIsMirrored(!isMirrored)}
            className={`flex items-center gap-1.5 px-3.5 py-2 rounded-xl transition-all font-mono focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-primary ${
              isMirrored
                ? 'bg-brand-primary/20 text-brand-primary border border-brand-primary/40 shadow-glow-cyan'
                : 'text-slate-400 hover:text-slate-200'
            }`}
            title="Toggle Horizontal Video Mirroring"
            aria-label="Toggle camera mirror mode"
          >
            <FlipHorizontal className="w-3.5 h-3.5" />
            <span>Mirror: {isMirrored ? 'ON' : 'OFF'}</span>
          </button>

          <button
            onClick={() => setShowSkeletonOverlay(!showSkeletonOverlay)}
            className={`flex items-center gap-1.5 px-3.5 py-2 rounded-xl transition-all font-mono focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-primary ${
              showSkeletonOverlay
                ? 'bg-brand-secondary/20 text-brand-secondary border border-brand-secondary/40 shadow-glow-violet'
                : 'text-slate-400 hover:text-slate-200'
            }`}
            title="Toggle Neon 21 Landmark Overlay"
            aria-label="Toggle landmark overlay"
          >
            {showSkeletonOverlay ? <Eye className="w-3.5 h-3.5 text-brand-secondary" /> : <EyeOff className="w-3.5 h-3.5" />}
            <span>Overlay: {showSkeletonOverlay ? 'ON' : 'OFF'}</span>
          </button>
        </div>
      </div>

      {/* Main Grid: Video Frame + Dashboard Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left Column: Interactive Video & Canvas View */}
        <div className="lg:col-span-2 space-y-4 rec-panel-animate">
          <div className="relative aspect-video rounded-3xl baklit-panel overflow-hidden border border-brand-border/70 bg-black/70 shadow-2xl flex items-center justify-center">
            
            {/* HTML5 Video Stream */}
            <video
              ref={videoRef}
              playsInline
              muted
              aria-label="Webcam camera feed"
              className={`w-full h-full object-cover transition-transform duration-200 ${
                isMirrored ? 'scale-x-[-1]' : ''
              } ${cameraState === 'ONLINE' ? 'block' : 'hidden'}`}
            />

            {/* Synchronized HTML5 Canvas Overlay */}
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full pointer-events-none z-10"
            />

            {/* Offline / Error Placeholder */}
            {cameraState !== 'ONLINE' && (
              <div className="text-center space-y-4 p-8 max-w-md">
                <div className="w-20 h-20 rounded-3xl bg-brand-surface/90 border border-brand-border flex items-center justify-center mx-auto text-slate-500 shadow-inner">
                  {cameraState === 'CONNECTING' ? (
                    <RefreshCw className="w-10 h-10 text-brand-primary animate-spin" />
                  ) : cameraState === 'PERMISSION_DENIED' ? (
                    <AlertTriangle className="w-10 h-10 text-rose-400" />
                  ) : (
                    <CameraOff className="w-10 h-10 text-slate-500" />
                  )}
                </div>

                <div className="space-y-1.5">
                  <h3 className="text-lg font-semibold text-white">
                    {cameraState === 'CONNECTING' && 'Connecting Camera & MediaPipe...'}
                    {cameraState === 'PERMISSION_DENIED' && 'Camera Permission Required'}
                    {cameraState === 'UNAVAILABLE' && 'Camera Device Unavailable'}
                    {cameraState === 'OFFLINE' && 'Webcam Feed Inactive'}
                  </h3>
                  <p className="text-slate-400 text-xs leading-relaxed">
                    {errorMessage || 'Click "Start Camera Stream" below to initialize browser camera permission and MediaPipe 21 landmark extraction.'}
                  </p>
                </div>
              </div>
            )}

            {/* Real-time Video Stream Top Badges */}
            {cameraState === 'ONLINE' && (
              <div className="absolute top-4 left-4 right-4 flex items-center justify-between pointer-events-none z-20">
                <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-2xl bg-black/80 backdrop-blur-md border border-white/10 text-xs font-mono">
                  <span className={`w-2.5 h-2.5 rounded-full ${handDetected ? 'bg-brand-accent animate-ping' : 'bg-amber-400'}`} />
                  <span className="text-slate-200">
                    {handDetected ? 'HAND TRACKING ACTIVE' : 'SEARCHING FOR HAND'}
                  </span>
                </div>

                <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-2xl bg-black/80 backdrop-blur-md border border-white/10 text-xs font-mono text-slate-300">
                  <Activity className="w-3.5 h-3.5 text-brand-primary" />
                  <span>{fps} FPS</span>
                  <span className="text-slate-600">|</span>
                  <span>{latency} ms</span>
                </div>
              </div>
            )}
          </div>

          {/* Controls Bar */}
          <div className="flex flex-wrap items-center justify-between gap-4 baklit-card p-4 rounded-3xl border border-brand-border/70">
            <div className="flex items-center gap-3">
              {cameraState === 'ONLINE' ? (
                <button
                  onClick={stopCamera}
                  className="px-6 py-3 rounded-2xl bg-rose-500/20 text-rose-300 border border-rose-500/40 hover:bg-rose-500/30 font-semibold text-sm flex items-center gap-2 transition-all shadow-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-400"
                >
                  <CameraOff className="w-4 h-4" /> Stop Camera Stream
                </button>
              ) : (
                <button
                  onClick={startCamera}
                  className="px-6 py-3 rounded-2xl bg-gradient-to-r from-brand-primary to-brand-secondary text-black font-bold text-sm flex items-center gap-2 transition-all shadow-glow-cyan hover:scale-105 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-primary"
                >
                  <Camera className="w-4 h-4 fill-black" /> Start Camera Stream
                </button>
              )}
            </div>

            <div className="text-xs font-mono text-slate-400 flex items-center gap-4">
              <span>Model: <strong className="text-brand-primary">MediaPipe WASM v0.10</strong></span>
              <span>Resolution: <strong className="text-slate-200">720p Ideal</strong></span>
            </div>
          </div>
        </div>

        {/* Right Column: Real-Time ML Gesture Prediction & Telemetry Dashboard */}
        <div className="space-y-6 rec-panel-animate">
          
          {/* Baklit Real-Time ASL Gesture Prediction Card */}
          <div className="baklit-card p-6 rounded-3xl border border-brand-border/70 space-y-5 bg-gradient-to-b from-brand-surface/95 to-brand-dark/95 shadow-2xl relative overflow-hidden">
            <div className="flex items-center justify-between border-b border-brand-border/60 pb-3">
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-brand-primary animate-pulse" /> Real-time ASL Prediction
              </h2>
              <span className="text-[10px] font-mono px-2.5 py-0.5 rounded-full bg-brand-accent/20 text-brand-accent border border-brand-accent/40 font-bold">
                Phase 5/7 Live
              </span>
            </div>

            {/* Classifier Selection Dropdown */}
            <div className="space-y-1.5 font-mono text-xs">
              <div className="flex items-center justify-between text-slate-400">
                <label htmlFor="classifier-select" className="font-semibold text-slate-300">CLASSIFIER ENGINE</label>
                {isInferring && <span className="text-brand-primary text-[10px] animate-pulse">Inferring...</span>}
              </div>
              <select
                id="classifier-select"
                value={selectedClassifier}
                onChange={(e) => setSelectedClassifier(e.target.value)}
                className="w-full px-3 py-2.5 rounded-2xl bg-brand-dark border border-brand-border text-slate-200 focus:outline-none focus:border-brand-primary text-xs cursor-pointer font-sans shadow-inner"
              >
                <optgroup label="Client-Side Rule Engine">
                  <option value="deterministic_geometry">⚡ Deterministic Geometry (Client-Side - Instant)</option>
                </optgroup>
                <optgroup label="FastAPI ML Models">
                  <option value="random_forest">🌲 Random Forest (FastAPI ML - 99.9% Acc)</option>
                  <option value="svm">⚡ Support Vector Machine (FastAPI ML)</option>
                  <option value="knn">🎯 K-Nearest Neighbors (FastAPI ML)</option>
                  <option value="logistic_regression">📈 Logistic Regression (FastAPI ML)</option>
                </optgroup>
              </select>
            </div>

            {/* Baklit Prediction Display Badge */}
            {predictionResult && handDetected ? (
              <div className="space-y-4 pt-2">
                <div className="flex items-center justify-around p-4 rounded-2xl bg-black/70 border border-brand-border/80 shadow-inner">
                  {/* Large Animated Letter Badge with Backlit Halo */}
                  <div ref={letterBadgeRef} className="w-24 h-24 rounded-2xl bg-gradient-to-tr from-brand-secondary via-brand-primary to-cyan-400 p-0.5 shadow-glow-cyan relative">
                    <div className="absolute inset-0 rounded-2xl bg-brand-primary/20 blur-md animate-pulse" />
                    <div className="relative w-full h-full rounded-[14px] bg-black/90 flex items-center justify-center">
                      <span className="text-5xl font-black tracking-tighter text-transparent bg-clip-text bg-gradient-to-r from-brand-primary via-cyan-200 to-brand-accent text-glow">
                        {predictionResult.predicted_letter}
                      </span>
                    </div>
                  </div>

                  {/* Confidence & Speed Stats */}
                  <div className="space-y-2 text-left font-mono">
                    <div>
                      <div className="text-[11px] text-slate-400 font-sans">CONFIDENCE SCORE</div>
                      <div className="text-2xl font-bold text-white">
                        {(predictionResult.confidence * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="text-[10px] text-slate-400">
                      Latency: <span className="text-brand-primary font-bold">{predictionResult.processing_time_ms} ms</span>
                    </div>
                  </div>
                </div>

                {/* Top Class Probabilities Distribution */}
                <div className="space-y-2">
                  <div className="text-xs font-mono text-slate-400">TOP GESTURE PROBABILITIES</div>
                  <div className="space-y-2">
                    {predictionResult.top_probabilities.map((prob, idx) => (
                      <div key={idx} className="space-y-1 text-xs font-mono">
                        <div className="flex items-center justify-between text-slate-300 text-[11px]">
                          <span>Letter <strong>{prob.label}</strong></span>
                          <span>{(prob.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full h-2 rounded-full bg-brand-dark overflow-hidden border border-brand-border/50 shadow-inner">
                          <div
                            className={`h-full rounded-full transition-all duration-500 ${
                              idx === 0
                                ? 'bg-gradient-to-r from-brand-primary via-cyan-400 to-brand-secondary shadow-glow-cyan'
                                : 'bg-slate-600'
                            }`}
                            style={{ width: `${Math.max(5, prob.confidence * 100)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-8 rounded-2xl bg-black/50 border border-brand-border/60 text-center space-y-2">
                <div className="w-12 h-12 rounded-2xl bg-brand-surface border border-brand-border flex items-center justify-center mx-auto text-slate-500 shadow-inner">
                  <Brain className="w-6 h-6 text-slate-500 animate-pulse" />
                </div>
                <p className="text-xs font-mono text-slate-400">
                  {cameraState !== 'ONLINE'
                    ? 'Start camera stream to activate real-time ML inference'
                    : 'Show hand in frame to predict ASL letter...'}
                </p>
              </div>
            )}
          </div>

          {/* Baklit System Telemetry Metrics */}
          <div className="baklit-card p-6 rounded-3xl border border-brand-border/70 space-y-6 shadow-xl">
            <h2 className="text-lg font-bold text-white flex items-center justify-between border-b border-brand-border/60 pb-3">
              <span className="flex items-center gap-2">
                <Sliders className="w-5 h-5 text-brand-primary" /> System Telemetry
              </span>
              <span className="text-[10px] font-mono px-2.5 py-0.5 rounded-full bg-brand-primary/10 text-brand-primary border border-brand-primary/40 font-bold">
                Phase 2/6
              </span>
            </h2>

            <div className="space-y-3 font-mono text-xs">
              <div className="p-3.5 rounded-2xl bg-brand-surface/80 border border-brand-border flex items-center justify-between shadow-inner">
                <span className="text-slate-400 font-medium">CAMERA STATUS</span>
                {cameraState === 'ONLINE' && (
                  <span className="flex items-center gap-1.5 font-bold text-brand-accent">
                    <span className="w-2 h-2 rounded-full bg-brand-accent animate-ping" /> ● ONLINE
                  </span>
                )}
                {cameraState === 'CONNECTING' && (
                  <span className="flex items-center gap-1.5 font-bold text-amber-400">
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" /> ● CONNECTING
                  </span>
                )}
                {cameraState === 'PERMISSION_DENIED' && (
                  <span className="flex items-center gap-1.5 font-bold text-rose-400">
                    <XCircle className="w-3.5 h-3.5" /> ● DENIED
                  </span>
                )}
                {cameraState === 'OFFLINE' && (
                  <span className="flex items-center gap-1.5 font-medium text-slate-500">
                    ● OFFLINE
                  </span>
                )}
              </div>

              <div className="p-3.5 rounded-2xl bg-brand-surface/80 border border-brand-border flex items-center justify-between shadow-inner">
                <span className="text-slate-400 font-medium">HAND TRACKING</span>
                {trackingState === 'ACTIVE' && cameraState === 'ONLINE' ? (
                  <span className="flex items-center gap-1.5 font-bold text-brand-primary">
                    ● ACTIVE
                  </span>
                ) : (
                  <span className="flex items-center gap-1.5 font-medium text-slate-500">
                    ● INACTIVE
                  </span>
                )}
              </div>

              <div className="p-3.5 rounded-2xl bg-brand-surface/80 border border-brand-border flex items-center justify-between shadow-inner">
                <span className="text-slate-400 font-medium">HAND DETECTION</span>
                {handDetected ? (
                  <span className="flex items-center gap-1.5 font-bold text-brand-accent">
                    <CheckCircle2 className="w-4 h-4" /> DETECTED ({detectedHandCount})
                  </span>
                ) : (
                  <span className="flex items-center gap-1.5 font-medium text-amber-400/80">
                    ✕ NOT DETECTED
                  </span>
                )}
              </div>

              <div className="p-3.5 rounded-2xl bg-brand-surface/80 border border-brand-border flex items-center justify-between shadow-inner">
                <span className="text-slate-400 font-medium">LANDMARK COUNT</span>
                <span className={`font-bold ${handDetected ? 'text-brand-primary text-sm' : 'text-slate-500'}`}>
                  {handDetected ? '21 / 21' : '0 / 21'}
                </span>
              </div>
            </div>

            {/* 21 Landmark Coordinates Inspector */}
            <div className="space-y-3 pt-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-mono text-slate-300 font-bold flex items-center gap-1.5">
                  <Layers className="w-4 h-4 text-brand-secondary" /> Landmark Coordinates (x, y, z)
                </span>
                <span className="text-[10px] text-slate-500 font-mono">Sampled @ 5Hz</span>
              </div>

              <div className="p-3.5 rounded-2xl bg-brand-dark/95 border border-brand-border space-y-2 text-[11px] font-mono max-h-44 overflow-y-auto shadow-inner">
                {inspectorLandmarks ? (
                  keyNodes.map((node) => {
                    const lm = inspectorLandmarks[node.index];
                    return (
                      <div key={node.index} className="flex items-center justify-between border-b border-brand-border/40 pb-1.5 last:border-0 last:pb-0">
                        <span className="text-slate-300">
                          #{node.index} {node.name}
                        </span>
                        <span className="text-brand-primary font-bold">
                          x:{lm.x.toFixed(2)} y:{lm.y.toFixed(2)} z:{lm.z.toFixed(2)}
                        </span>
                      </div>
                    );
                  })
                ) : (
                  <div className="text-slate-500 italic text-center py-4">
                    Position hand in camera frame to inspect 3D landmark values...
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
