import { useState, useEffect, useRef } from 'react';
import anime from 'animejs';
import { 
  BookOpen, 
  Delete, 
  RotateCcw, 
  Volume2, 
  Sparkles, 
  CheckCircle2, 
  ArrowRight, 
  Award, 
  Flame, 
  Camera, 
  RefreshCw,
  Eye,
  EyeOff
} from 'lucide-react';
import { useMediaPipeHands } from '../hooks/useMediaPipeHands';
import { drawNeonHandLandmarks } from '../utils/handDrawer';
import { predictLandmarks } from '../services/api';
import { classifyASLGeometry } from '../utils/aslGeometry';

export default function Practice() {
  const [wordBuffer, setWordBuffer] = useState("SIGNSENSE");
  const [selectedTargetLetter, setSelectedTargetLetter] = useState("A");
  const [score, setScore] = useState(0);
  const [streak, setStreak] = useState(0);
  const [feedback, setFeedback] = useState(null); // { type: 'success' | 'attempt', text: string }
  const [selectedClassifier, setSelectedClassifier] = useState('logistic_regression');

  const flashcardRef = useRef(null);
  const aslAlphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

  const {
    videoRef,
    canvasRef,
    landmarksRef,
    handednessRef,
    cameraState,
    handDetected,
    showSkeletonOverlay,
    isMirrored,
    startCamera
  } = useMediaPipeHands();

  const [livePrediction, setLivePrediction] = useState(null);
  const isCooldownRef = useRef(false);

  // Auto-start camera when Practice page mounts
  useEffect(() => {
    startCamera();
  }, [startCamera]);

  // Reset target letter when selection changes
  useEffect(() => {
    setFeedback(null);
    if (flashcardRef.current) {
      anime({
        targets: flashcardRef.current,
        scale: [0.9, 1],
        opacity: [0.7, 1],
        duration: 300,
        easing: 'easeOutExpo'
      });
    }
  }, [selectedTargetLetter]);

  // Live prediction loop to check target gesture completion & update telemetry
  useEffect(() => {
    let isSubscribed = true;

    const practiceInterval = setInterval(async () => {
      if (landmarksRef.current && cameraState === 'ONLINE') {
        let result = null;
        if (selectedClassifier === 'deterministic_geometry') {
          result = classifyASLGeometry(landmarksRef.current, handednessRef.current);
        } else {
          result = await predictLandmarks(landmarksRef.current, selectedClassifier, handednessRef.current);
          if (!result) {
            result = classifyASLGeometry(landmarksRef.current, handednessRef.current);
          }
        }
        if (!isSubscribed) return;

        if (result) {
          setLivePrediction(result);

          // Check target match if not in success cooldown (threshold >= 90% structural fit)
          if (!isCooldownRef.current && result.predicted_letter === selectedTargetLetter && result.confidence >= 0.90) {
            isCooldownRef.current = true;

            // Trigger Anime.js target flashcard success animation
            if (flashcardRef.current) {
              anime({
                targets: flashcardRef.current,
                scale: [1, 1.25, 1],
                rotateZ: [0, 8, -8, 0],
                duration: 600,
                easing: 'easeInOutBack'
              });
            }

            // Success! Target letter matched with >= 90% structural fit!
            setScore(prev => prev + 10);
            setStreak(prev => prev + 1);
            setWordBuffer(prev => prev + selectedTargetLetter);
            setFeedback({ type: 'success', text: `Great Job! Letter '${selectedTargetLetter}' Verified (90%+ Fit)!` });

            // Pick next target letter cleanly after toast animation
            setTimeout(() => {
              if (isSubscribed) {
                const nextIdx = (aslAlphabet.indexOf(selectedTargetLetter) + 1) % 26;
                setSelectedTargetLetter(aslAlphabet[nextIdx]);
                setFeedback(null);
                isCooldownRef.current = false;
              }
            }, 1500);
          }
        }
      } else if (!landmarksRef.current && livePrediction) {
        setLivePrediction(null);
      }
    }, 200);

    return () => {
      isSubscribed = false;
      clearInterval(practiceInterval);
    };
  }, [landmarksRef, cameraState, selectedTargetLetter, selectedClassifier]);

  // Canvas render loop for Practice mode camera view
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

  const handleAddLetter = (letter) => {
    setWordBuffer((prev) => prev + letter);
  };

  const handleClear = () => {
    setWordBuffer("");
  };

  const handleBackspace = () => {
    setWordBuffer((prev) => prev.slice(0, -1));
  };

  return (
    <div className="space-y-8 py-6">
      {/* Baklit UI Page Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-brand-border/60 pb-5">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight flex items-center gap-2.5">
            <BookOpen className="w-8 h-8 text-brand-secondary" /> Word Builder & ASL Flashcards
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Practice target ASL gestures in real time with live feedback and assemble fingerspelling word buffers
          </p>
        </div>

        {/* Baklit UI Score & Streak Stats Badges */}
        <div className="flex items-center gap-3 font-mono text-xs">
          <div className="flex items-center gap-2 px-4 py-2.5 rounded-2xl bg-brand-surface/90 border border-brand-border/80 shadow-glow-cyan">
            <Award className="w-4 h-4 text-amber-400" />
            <span>SCORE: <strong className="text-amber-400 text-sm">{score}</strong></span>
          </div>
          <div className="flex items-center gap-2 px-4 py-2.5 rounded-2xl bg-brand-surface/90 border border-brand-border/80 shadow-glow-rose">
            <Flame className="w-4 h-4 text-rose-400 animate-pulse" />
            <span>STREAK: <strong className="text-rose-400 text-sm">{streak}</strong></span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left Column: Word Builder & Reference Grid */}
        <div className="lg:col-span-2 space-y-6">
          
          {/* Buffered Word Output Box */}
          <div className="baklit-card p-6 rounded-3xl border border-brand-border/70 space-y-4 shadow-xl">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-brand-primary animate-pulse" /> Buffered Word Output
              </h2>
              <div className="flex items-center gap-2">
                <button 
                  onClick={handleBackspace}
                  className="px-3.5 py-1.5 rounded-xl bg-brand-surface border border-brand-border text-slate-300 hover:text-white text-xs flex items-center gap-1.5 font-mono focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-primary hover:border-brand-primary/40 transition-all"
                  aria-label="Backspace buffer"
                >
                  <Delete className="w-3.5 h-3.5" /> Backspace
                </button>
                <button 
                  onClick={handleClear}
                  className="px-3.5 py-1.5 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 hover:bg-rose-500/20 text-xs flex items-center gap-1.5 font-mono focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-400 transition-all"
                  aria-label="Clear word buffer"
                >
                  <RotateCcw className="w-3.5 h-3.5" /> Clear Buffer
                </button>
              </div>
            </div>

            {/* Baklit Live Buffer Text Box */}
            <div className="min-h-[105px] p-6 rounded-2xl bg-brand-dark/95 border border-brand-border flex items-center justify-between gap-4 font-mono shadow-inner relative overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-r from-brand-primary/5 via-transparent to-brand-secondary/5 pointer-events-none" />
              <span className="text-3xl sm:text-4xl font-extrabold tracking-widest text-brand-primary text-glow break-all relative z-10">
                {wordBuffer || <span className="text-slate-600 italic text-xl font-normal">Sign letters to build word...</span>}
              </span>
              <button 
                onClick={() => {
                  if ('speechSynthesis' in window && wordBuffer) {
                    const utterance = new SpeechSynthesisUtterance(wordBuffer);
                    window.speechSynthesis.speak(utterance);
                  }
                }}
                className="p-3.5 rounded-2xl bg-brand-surface border border-brand-border text-brand-primary hover:bg-brand-card hover:border-brand-primary/50 transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-primary shadow-lg shadow-brand-primary/10 relative z-10"
                title="Speak Word Audio"
                aria-label="Speak word"
              >
                <Volume2 className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Interactive ASL Letter Reference Grid */}
          <div className="baklit-card p-6 rounded-3xl border border-brand-border/70 space-y-4 shadow-xl">
            <h2 className="text-lg font-bold text-white">ASL Alphabet Reference Grid</h2>
            <div className="grid grid-cols-6 sm:grid-cols-9 gap-2.5">
              {aslAlphabet.map((letter) => (
                <button
                  key={letter}
                  onClick={() => {
                    setSelectedTargetLetter(letter);
                    handleAddLetter(letter);
                  }}
                  className={`aspect-square rounded-2xl flex flex-col items-center justify-center font-mono font-bold text-lg transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-primary ${
                    selectedTargetLetter === letter
                      ? 'bg-gradient-to-tr from-brand-secondary via-brand-primary to-cyan-400 text-black shadow-glow-cyan scale-105 border-transparent font-extrabold'
                      : 'bg-brand-surface/90 border border-brand-border/80 text-slate-200 hover:border-brand-primary/60 hover:text-brand-primary hover:scale-105'
                  }`}
                  aria-label={`Select letter ${letter}`}
                >
                  {letter}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column: Flashcard & Practice Camera Verification */}
        <div className="space-y-6">
          
          {/* Baklit Animated Flashcard Target Box */}
          <div className="baklit-card p-6 rounded-3xl border border-brand-border/70 space-y-4 text-center bg-gradient-to-b from-brand-surface/95 to-brand-dark/95 shadow-2xl relative overflow-hidden">
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-40 h-1 bg-gradient-to-r from-transparent via-brand-primary to-transparent" />
            <h2 className="text-lg font-bold text-white">ASL Target Flashcard</h2>
            <p className="text-slate-400 text-xs">Sign this letter into your camera to verify:</p>

            <div 
              ref={flashcardRef}
              className="w-36 h-36 rounded-3xl bg-gradient-to-br from-brand-secondary/30 via-brand-primary/20 to-cyan-500/30 border border-brand-primary/60 flex items-center justify-center mx-auto shadow-glow-cyan relative overflow-hidden"
            >
              <div className="absolute inset-0 bg-brand-primary/10 blur-xl animate-pulse" />
              <span className="text-7xl font-black text-brand-primary font-mono text-glow relative z-10">
                {selectedTargetLetter}
              </span>
            </div>

            {/* Target Verification Feedback Toast */}
            {feedback && (
              <div className={`p-3.5 rounded-2xl border text-xs font-mono font-bold flex items-center justify-center gap-2 animate-bounce ${
                feedback.type === 'success' ? 'bg-brand-accent/20 border-brand-accent/60 text-brand-accent shadow-glow-emerald' : 'bg-amber-400/20 border-amber-400/60 text-amber-300'
              }`}>
                <CheckCircle2 className="w-4.5 h-4.5" /> {feedback.text}
              </div>
            )}

            <button
              onClick={() => {
                const randomIdx = Math.floor(Math.random() * 26);
                setSelectedTargetLetter(aslAlphabet[randomIdx]);
                setFeedback(null);
              }}
              className="w-full py-3 rounded-2xl bg-brand-card hover:bg-brand-border/80 text-slate-100 border border-brand-border text-xs font-semibold flex items-center justify-center gap-2 transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-primary shadow-md hover:border-brand-primary/40"
            >
              Next Random Target <ArrowRight className="w-4 h-4 text-brand-primary" />
            </button>
          </div>

          {/* Practice Camera Stream Verification Panel */}
          <div className="baklit-card p-4 rounded-3xl border border-brand-border/70 space-y-3.5 shadow-xl">
            <div className="flex items-center justify-between text-xs">
              <span className="font-mono text-slate-200 font-bold flex items-center gap-1.5">
                <Camera className="w-4 h-4 text-brand-primary" /> Verification Camera
              </span>
              <span className="text-[10px] text-slate-400 font-mono">
                {cameraState === 'ONLINE' ? 'LIVE' : 'OFFLINE'}
              </span>
            </div>

            {/* Classifier Engine Selection Dropdown */}
            <div className="space-y-1 font-mono text-[11px]">
              <label htmlFor="practice-engine-select" className="text-slate-400 font-semibold block">CLASSIFICATION ENGINE</label>
              <select
                id="practice-engine-select"
                value={selectedClassifier}
                onChange={(e) => setSelectedClassifier(e.target.value)}
                className="w-full px-3 py-2 rounded-xl bg-brand-dark border border-brand-border text-slate-200 focus:outline-none focus:border-brand-primary text-xs cursor-pointer font-sans shadow-inner"
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

            <div className="relative aspect-video rounded-2xl bg-black/70 border border-brand-border/70 overflow-hidden flex items-center justify-center shadow-inner">
              <video
                ref={videoRef}
                playsInline
                muted
                className={`w-full h-full object-cover ${isMirrored ? 'scale-x-[-1]' : ''} ${cameraState === 'ONLINE' ? 'block' : 'hidden'}`}
              />
              <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full pointer-events-none z-10"
              />
              {cameraState === 'ONLINE' && livePrediction && (
                <div className="absolute top-3 left-3 right-3 flex items-center justify-between pointer-events-none z-20 font-mono text-[11px]">
                  <div className={`px-3 py-1.5 rounded-xl backdrop-blur-md border flex items-center gap-1.5 font-bold ${
                    livePrediction.predicted_letter === selectedTargetLetter
                      ? 'bg-brand-accent/30 text-brand-accent border-brand-accent/60 shadow-glow-emerald'
                      : 'bg-black/75 text-slate-200 border-white/10'
                  }`}>
                    <span>AI DETECTS: <strong className="text-sm">{livePrediction.predicted_letter}</strong></span>
                    <span>({(livePrediction.confidence * 100).toFixed(0)}%)</span>
                  </div>
                  <div className="px-3 py-1.5 rounded-xl bg-black/75 backdrop-blur-md border border-white/10 text-slate-400">
                    TARGET: <strong className="text-white">{selectedTargetLetter}</strong>
                  </div>
                </div>
              )}

              {cameraState !== 'ONLINE' && (
                <button
                  onClick={startCamera}
                  className="px-5 py-2.5 rounded-2xl bg-gradient-to-r from-brand-primary to-brand-secondary text-black font-bold text-xs flex items-center gap-2 shadow-glow-cyan hover:scale-105 transition-transform"
                >
                  <Camera className="w-4 h-4 fill-black" /> Enable Camera
                </button>
              )}
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
