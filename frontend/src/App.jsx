import { useState } from 'react';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import ParticlesBg from './components/ParticlesBg';
import Home from './pages/Home';
import Recognition from './pages/Recognition';
import Practice from './pages/Practice';
import About from './pages/About';

export default function App() {
  const [activeTab, setActiveTab] = useState('home');

  return (
    <div className="min-h-screen flex flex-col relative bg-brand-dark text-slate-100 selection:bg-brand-primary selection:text-black">
      {/* Background Particles */}
      <ParticlesBg />

      {/* Navigation */}
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Main View Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 relative z-10 py-4">
        {activeTab === 'home' && <Home setActiveTab={setActiveTab} />}
        {activeTab === 'recognition' && <Recognition />}
        {activeTab === 'practice' && <Practice />}
        {activeTab === 'about' && <About />}
      </main>

      {/* Footer */}
      <Footer />
    </div>
  );
}
