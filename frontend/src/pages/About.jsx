import { useEffect } from 'react';
import anime from 'animejs';
import { 
  Info, 
  Cpu, 
  Layers, 
  GitCompare, 
  ShieldCheck, 
  Video, 
  Target, 
  Brain, 
  Award,
  CheckCircle2
} from 'lucide-react';

export default function About() {
  useEffect(() => {
    anime({
      targets: '.about-card-animate',
      translateY: [20, 0],
      opacity: [0, 1],
      delay: anime.stagger(120),
      duration: 750,
      easing: 'easeOutExpo'
    });
  }, []);

  const pipelineStages = [
    { step: 1, title: "Webcam Input Stream", detail: "Browser `getUserMedia()` API streams live video frames at 720p.", icon: Video, color: "text-cyan-400 border-cyan-500/30" },
    { step: 2, title: "MediaPipe Vision Engine", detail: "MediaPipe WASM extracts 21 3D hand landmarks (x, y, z) per frame.", icon: Target, color: "text-purple-400 border-purple-500/30" },
    { step: 3, title: "21 Hand Landmarks", detail: "Produces 21 spatial node coordinates (x0, y0, z0 ... x20, y20, z20).", icon: Layers, color: "text-indigo-400 border-indigo-500/30" },
    { step: 4, title: "63 Normalized Features", detail: "Translates origin to wrist (#0) and normalizes by max Euclidean distance for scale invariance.", icon: Cpu, color: "text-emerald-400 border-emerald-500/30" },
    { step: 5, title: "Random Forest Classifier", detail: "Trained 120-tree Random Forest model predicts gesture class A–Z with confidence probability.", icon: Brain, color: "text-amber-400 border-amber-500/30" },
    { step: 6, title: "A-Z Letter Prediction", detail: "Returns live letter prediction badge, top-3 probabilities, and processing latency (ms).", icon: Award, color: "text-rose-400 border-rose-500/30" },
  ];

  return (
    <div className="space-y-10 py-6 max-w-5xl mx-auto">
      {/* Page Title */}
      <div className="border-b border-brand-border/40 pb-4">
        <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight flex items-center gap-2">
          <Info className="w-7 h-7 text-brand-primary" /> System Architecture & Technical Specs
        </h1>
        <p className="text-slate-400 text-sm">
          Technical breakdown of the MediaPipe + Machine Learning classification engine
        </p>
      </div>

      {/* What is SignSense AI */}
      <section className="about-card-animate glass-panel p-6 rounded-2xl border border-brand-border/60 space-y-4">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-brand-primary" /> What is SignSense AI?
        </h2>
        <p className="text-slate-300 text-sm leading-relaxed">
          <strong>SignSense AI</strong> is a real-time American Sign Language (ASL) alphabet and fingerspelling recognition web application. It parses 21 3D hand landmarks from a standard webcam stream, normalizes coordinates into origin- and scale-invariant numerical vectors, and uses trained machine learning classifiers to predict letters A–Z instantly in the browser.
        </p>
      </section>

      {/* 6-Stage Pipeline Architecture */}
      <section className="space-y-4">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <Layers className="w-5 h-5 text-brand-secondary" /> How It Works (6-Stage Pipeline)
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {pipelineStages.map((stage) => {
            const Icon = stage.icon;
            return (
              <div key={stage.step} className={`about-card-animate p-5 rounded-2xl glass-panel border ${stage.color} space-y-3`}>
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-bold px-2 py-0.5 rounded bg-brand-surface border border-brand-border">
                    STAGE {stage.step}
                  </span>
                  <Icon className="w-5 h-5" />
                </div>
                <h3 className="text-base font-bold text-white">{stage.title}</h3>
                <p className="text-slate-400 text-xs leading-relaxed">{stage.detail}</p>
              </div>
            );
          })}
        </div>
      </section>

      {/* Technology Stack & Decoupled Architecture */}
      <section className="about-card-animate glass-panel p-6 rounded-2xl border border-brand-border/60 space-y-4">
        <h2 className="text-xl font-bold text-white flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-brand-accent" /> Technology Stack & Architecture
        </h2>
        
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="p-4 rounded-xl bg-brand-surface border border-brand-border space-y-2 text-xs">
            <span className="font-bold text-brand-primary font-mono text-sm block">Frontend Stack</span>
            <ul className="text-slate-300 space-y-1 list-disc list-inside">
              <li>React 18 + Vite</li>
              <li>Tailwind CSS (Dark theme & Glassmorphism)</li>
              <li>Anime.js UI micro-animations</li>
              <li>tsParticles ambient background</li>
              <li>Browser `getUserMedia()` Web API</li>
            </ul>
          </div>

          <div className="p-4 rounded-xl bg-brand-surface border border-brand-border space-y-2 text-xs">
            <span className="font-bold text-brand-secondary font-mono text-sm block">Backend & ML Stack</span>
            <ul className="text-slate-300 space-y-1 list-disc list-inside">
              <li>Python 3.13 + FastAPI</li>
              <li>MediaPipe Hands Vision WASM</li>
              <li>scikit-learn (Random Forest, SVM, KNN, LogReg)</li>
              <li>NumPy & Pandas preprocessing</li>
              <li>joblib model persistence</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Responsible AI ML Disclaimer */}
      <section className="about-card-animate p-5 rounded-2xl bg-brand-surface/90 border border-brand-border text-xs text-slate-300 space-y-2">
        <div className="font-bold text-white flex items-center gap-2 text-sm">
          <CheckCircle2 className="w-4 h-4 text-brand-primary" /> Scope & Machine Learning Disclaimer
        </div>
        <p className="text-slate-400 leading-relaxed">
          SignSense AI is specifically trained for <strong>static ASL alphabet (A–Z) fingerspelling gesture recognition</strong>. It is not designed to translate dynamic full-body American Sign Language grammar or conversational sentences. Model confidence ratings reflect statistical probability on normalized landmark geometries.
        </p>
      </section>
    </div>
  );
}

function Sparkles(props) {
  return (
    <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/>
    </svg>
  );
}
