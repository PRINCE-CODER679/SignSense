import { useEffect } from 'react';
import anime from 'animejs';
import { 
  Camera, 
  BookOpen, 
  Cpu, 
  Video,
  Target,
  Brain,
  Award
} from 'lucide-react';
import KineticHero from '../components/KineticHero';

export default function Home({ setActiveTab }) {
  useEffect(() => {
    // Diagram pipeline nodes pulsing animation
    anime({
      targets: '.pipeline-step',
      scale: [0.96, 1],
      opacity: [0.5, 1],
      delay: anime.stagger(180, { start: 500 }),
      duration: 700,
      easing: 'easeOutCubic'
    });

    // Feature card stagger animation
    anime({
      targets: '.feature-card-animate',
      translateY: [30, 0],
      opacity: [0, 1],
      delay: anime.stagger(120, { start: 700 }),
      duration: 800,
      easing: 'easeOutQuad'
    });
  }, []);

  const features = [
    {
      icon: Camera,
      title: "REAL-TIME RECOGNITION",
      description: "Low-latency browser camera stream processing predicting fingerspelling letters in milliseconds.",
      color: "from-cyan-500 to-blue-600"
    },
    {
      icon: Cpu,
      title: "21-LANDMARK TRACKING",
      description: "MediaPipe Vision WASM extracts 21 3D hand landmarks (x, y, z) per hand at 60 FPS.",
      color: "from-purple-500 to-indigo-600"
    },
    {
      icon: Brain,
      title: "AI-POWERED PREDICTION",
      description: "Trained Random Forest, SVM, KNN & Logistic Regression classifiers normalize coordinates into 63 features.",
      color: "from-emerald-400 to-teal-600"
    },
    {
      icon: BookOpen,
      title: "PRACTICE MODE",
      description: "Interactive ASL alphabet flashcards, real-time target sign verification, and word builder buffer.",
      color: "from-pink-500 to-rose-600"
    }
  ];

  return (
    <div className="space-y-16 py-4">
      {/* Ported Vanta.js GLOBE + GSAP Kinetic Hero Section */}
      <KineticHero setActiveTab={setActiveTab} />

      {/* Interactive Visual Representation of the Pipeline */}
      <section className="space-y-6">
        <div className="text-center space-y-2">
          <h2 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">Real-Time Inference Pipeline</h2>
          <p className="text-slate-400 text-sm">How your camera feed transforms into instant ASL letter predictions</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 relative">
          
          {/* Step 1: Webcam */}
          <div className="pipeline-step p-6 rounded-2xl glass-panel border border-brand-border/60 text-center space-y-3 relative">
            <div className="w-14 h-14 rounded-2xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 flex items-center justify-center mx-auto shadow-inner">
              <Video className="w-7 h-7" />
            </div>
            <div className="font-mono text-xs text-brand-primary font-bold">STEP 1</div>
            <h3 className="text-base font-bold text-white">Webcam Feed</h3>
            <p className="text-slate-400 text-xs leading-relaxed">Browser `getUserMedia()` API streams live video frames.</p>
          </div>

          {/* Step 2: MediaPipe Hand Landmarks */}
          <div className="pipeline-step p-6 rounded-2xl glass-panel border border-brand-border/60 text-center space-y-3 relative">
            <div className="w-14 h-14 rounded-2xl bg-purple-500/10 border border-purple-500/30 text-purple-400 flex items-center justify-center mx-auto shadow-inner">
              <Target className="w-7 h-7" />
            </div>
            <div className="font-mono text-xs text-purple-400 font-bold">STEP 2</div>
            <h3 className="text-base font-bold text-white">Hand Landmarks</h3>
            <p className="text-slate-400 text-xs leading-relaxed">MediaPipe extracts 21 3D coordinates (x, y, z) per frame.</p>
          </div>

          {/* Step 3: AI Model */}
          <div className="pipeline-step p-6 rounded-2xl glass-panel border border-brand-border/60 text-center space-y-3 relative">
            <div className="w-14 h-14 rounded-2xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 flex items-center justify-center mx-auto shadow-inner">
              <Brain className="w-7 h-7" />
            </div>
            <div className="font-mono text-xs text-emerald-400 font-bold">STEP 3</div>
            <h3 className="text-base font-bold text-white">AI Model</h3>
            <p className="text-slate-400 text-xs leading-relaxed">FastAPI normalizes coordinates into 63 features & classifies.</p>
          </div>

          {/* Step 4: Letter Prediction */}
          <div className="pipeline-step p-6 rounded-2xl glass-panel border border-brand-border/60 text-center space-y-3 relative">
            <div className="w-14 h-14 rounded-2xl bg-amber-500/10 border border-amber-500/30 text-amber-400 flex items-center justify-center mx-auto shadow-inner">
              <Award className="w-7 h-7" />
            </div>
            <div className="font-mono text-xs text-amber-400 font-bold">STEP 4</div>
            <h3 className="text-base font-bold text-white">Letter Prediction</h3>
            <p className="text-slate-400 text-xs leading-relaxed">Returns predicted letter A–Z with confidence probability.</p>
          </div>

        </div>
      </section>

      {/* Feature Cards Grid (4 Core Cards) */}
      <section className="space-y-6">
        <div className="text-center space-y-2">
          <h2 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">Core System Capabilities</h2>
          <p className="text-slate-400 text-sm">Engineered for precision, invariance, and interactive sign language learning</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feat, idx) => {
            const Icon = feat.icon;
            return (
              <div key={idx} className="feature-card-animate glass-card p-6 rounded-2xl space-y-4 relative overflow-hidden group border border-brand-border/60 hover:border-brand-primary/40 transition-all">
                <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${feat.color} p-2.5 text-white flex items-center justify-center shadow-md`}>
                  <Icon className="w-6 h-6" />
                </div>
                <h3 className="text-base font-bold text-white group-hover:text-brand-primary transition-colors tracking-wide">
                  {feat.title}
                </h3>
                <p className="text-slate-400 text-xs leading-relaxed">
                  {feat.description}
                </p>
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}
