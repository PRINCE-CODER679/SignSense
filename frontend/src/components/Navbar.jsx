import { useState, useEffect } from 'react';
import anime from 'animejs';
import { Camera, BookOpen, Info, Activity, Sparkles, CheckCircle2, XCircle, Menu, X } from 'lucide-react';
import { checkBackendHealth } from '../services/api';

export default function Navbar({ activeTab, setActiveTab }) {
  const [backendStatus, setBackendStatus] = useState('checking'); // 'online' | 'offline' | 'checking'
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const verifyHealth = async () => {
      const health = await checkBackendHealth();
      if (health && health.status === 'ok') {
        setBackendStatus('online');
      } else {
        setBackendStatus('offline');
      }
    };
    verifyHealth();
    const interval = setInterval(verifyHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  // Anime.js tab indicator animation on tab switch
  useEffect(() => {
    anime({
      targets: `.nav-tab-${activeTab}`,
      scale: [0.95, 1],
      opacity: [0.7, 1],
      duration: 350,
      easing: 'easeOutQuad'
    });
  }, [activeTab]);

  const navItems = [
    { id: 'home', label: 'Home', icon: Sparkles },
    { id: 'recognition', label: 'Recognition Engine', icon: Camera },
    { id: 'practice', label: 'Word Builder & Practice', icon: BookOpen },
    { id: 'about', label: 'Pipeline Architecture', icon: Info },
  ];

  const handleTabChange = (tabId) => {
    setActiveTab(tabId);
    setMobileMenuOpen(false);
  };

  return (
    <header className="sticky top-0 z-50 baklit-panel border-b border-brand-border/60 bg-brand-dark/85 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Baklit UI Brand Logo */}
        <button
          onClick={() => handleTabChange('home')}
          className="flex items-center gap-3 cursor-pointer group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-primary rounded-xl p-1 text-left"
          aria-label="SignSense AI Home"
        >
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-brand-secondary via-brand-primary to-cyan-400 p-0.5 shadow-lg shadow-brand-primary/30 group-hover:scale-105 transition-transform duration-300 relative">
            <div className="absolute inset-0 rounded-xl bg-brand-primary/20 blur-md group-hover:blur-lg transition-all" />
            <div className="relative w-full h-full bg-brand-dark rounded-[10px] flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-brand-primary animate-pulse" />
            </div>
          </div>
          <div>
            <span className="text-xl font-bold tracking-tight text-white flex items-center gap-1.5">
              SignSense <span className="text-transparent bg-clip-text bg-gradient-to-r from-brand-primary via-brand-accent to-brand-secondary font-mono text-lg text-glow">AI</span>
            </span>
            <span className="text-[10px] text-slate-400 block tracking-widest uppercase font-mono">ASL Recognition Engine</span>
          </div>
        </button>

        {/* Baklit UI Desktop Navigation Tabs */}
        <nav className="hidden md:flex items-center gap-1.5 bg-brand-surface/80 p-1.5 rounded-2xl border border-brand-border/70 shadow-inner" aria-label="Main Navigation">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => handleTabChange(item.id)}
                aria-current={isActive ? 'page' : undefined}
                className={`nav-tab-${item.id} flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-primary ${
                  isActive
                    ? 'bg-gradient-to-r from-brand-primary/20 via-brand-secondary/30 to-brand-primary/10 text-brand-primary border border-brand-primary/50 shadow-glow-cyan font-semibold'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-brand-card/60'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-brand-primary text-glow' : 'text-slate-400'}`} />
                {item.label}
              </button>
            );
          })}
        </nav>

        {/* Baklit UI Status Badge & Mobile Menu */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-brand-surface/90 border border-brand-border/80 text-xs font-mono shadow-inner" aria-label={`Backend status: ${backendStatus}`}>
            <Activity className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-slate-400 hidden sm:inline">Backend:</span>
            {backendStatus === 'checking' && (
              <span className="flex items-center gap-1 text-amber-400">
                <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping" /> Checking...
              </span>
            )}
            {backendStatus === 'online' && (
              <span className="flex items-center gap-1 text-brand-accent font-semibold">
                <CheckCircle2 className="w-3.5 h-3.5" /> Ready (FastAPI)
              </span>
            )}
            {backendStatus === 'offline' && (
              <span className="flex items-center gap-1 text-rose-400">
                <XCircle className="w-3.5 h-3.5" /> Standby
              </span>
            )}
          </div>

          {/* Mobile Hamburger Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2 rounded-xl bg-brand-surface border border-brand-border text-slate-300 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-primary"
            aria-label="Toggle mobile menu"
            aria-expanded={mobileMenuOpen}
          >
            {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>
      </div>

      {/* Baklit UI Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="md:hidden border-t border-brand-border/50 py-3 px-4 bg-brand-dark/95 backdrop-blur-2xl space-y-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => handleTabChange(item.id)}
                aria-current={isActive ? 'page' : undefined}
                className={`w-full flex items-center gap-3 px-4 py-2.5 rounded-xl text-sm font-medium transition-all ${
                  isActive
                    ? 'bg-brand-primary/20 text-brand-primary border border-brand-primary/40 shadow-glow-cyan font-semibold'
                    : 'text-slate-300 hover:bg-brand-surface'
                }`}
              >
                <Icon className="w-4 h-4" />
                {item.label}
              </button>
            );
          })}
        </div>
      )}
    </header>
  );
}
