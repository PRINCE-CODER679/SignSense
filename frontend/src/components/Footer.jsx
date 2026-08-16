import { Cpu, ShieldCheck, GitBranch } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="mt-auto border-t border-brand-border/40 bg-brand-dark/90 py-8 text-slate-400 text-xs font-mono">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <Cpu className="w-4 h-4 text-brand-primary" />
          <span>SignSense AI v1.0.0</span>
          <span className="text-slate-600">|</span>
          <span className="text-slate-300">ASL Alphabet / Fingerspelling Recognition</span>
        </div>
        
        <div className="flex items-center gap-6 text-slate-400">
          <span className="flex items-center gap-1">
            <ShieldCheck className="w-3.5 h-3.5 text-brand-accent" /> MediaPipe 21 Landmarks
          </span>
          <span className="flex items-center gap-1">
            <GitBranch className="w-3.5 h-3.5 text-brand-secondary" /> ML Classifier Engine
          </span>
        </div>

        <div className="text-slate-400 text-[11px]">
          Target Stack: React + Vite + FastAPI + OpenCV
        </div>
      </div>
    </footer>
  );
}
