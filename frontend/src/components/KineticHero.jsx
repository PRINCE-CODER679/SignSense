import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import GLOBE from 'vanta/dist/vanta.globe.min';
import gsap from 'gsap';

export default function KineticHero({ setActiveTab }) {
  const vantaRef = useRef(null);
  const containerRef = useRef(null);
  const [vantaEffect, setVantaEffect] = useState(null);

  const glyphs = {
    H: 'M18 66 L18 30 Q18 16 30 16 Q42 16 30 16 Q42 16 42 30 L42 66 M14 40 L46 40 M22 16 L22 44 M38 16 L38 44',
    I: 'M30 16 L30 66 M18 16 L42 16 M18 66 L42 66'
  };
  const wordLetters = ["H", "I"];

  // Initialize Vanta.js GLOBE effect with clean unmount lifecycle
  useEffect(() => {
    let effect = null;
    if (!vantaEffect && vantaRef.current) {
      try {
        effect = GLOBE({
          el: vantaRef.current,
          THREE: THREE,
          mouseControls: true,
          touchControls: true,
          gyroControls: false,
          minHeight: 200.00,
          minWidth: 200.00,
          backgroundColor: 0x08060f,
          color: 0x00f0ff,
          color2: 0xff2e9a,
          size: 1.1
        });
        setVantaEffect(effect);
      } catch (err) {
        console.error("Vanta.js initialization error:", err);
      }
    }

    return () => {
      if (effect) effect.destroy();
    };
  }, []);

  // GSAP Entrance Animations & Path Drawing
  useEffect(() => {
    if (!containerRef.current) return;

    // Calculate path dash offsets for hand glyph draw-in effect
    const pathElements = containerRef.current.querySelectorAll(".kinetic-letter-wrap path");
    pathElements.forEach((path) => {
      const len = path.getTotalLength();
      path.style.strokeDasharray = len;
      path.style.strokeDashoffset = len;
    });

    const ctx = gsap.context(() => {
      const tl = gsap.timeline({ defaults: { ease: "power3.out" } });

      tl.to(".kinetic-kicker", { opacity: 1, y: 0, duration: 0.6 }, 0.1)
        .from(".kinetic-letter-wrap", { y: 40, opacity: 0, duration: 0.7, stagger: 0.15 }, 0.25)
        .to(".kinetic-letter-wrap path", { strokeDashoffset: 0, duration: 0.7, stagger: 0.15, ease: "power2.inOut" }, 0.3)
        .to(".kinetic-word-text", { opacity: 1, y: 0, duration: 0.6 }, 0.9)
        .to(".kinetic-subcopy", { opacity: 1, y: 0, duration: 0.6 }, 1.0)
        .to(".kinetic-actions", { opacity: 1, y: 0, duration: 0.6 }, 1.1)
        .from(".kinetic-sticker", { scale: 0, duration: 0.5, stagger: 0.1, ease: "back.out(2)" }, 1.2);

      gsap.to(".kinetic-sticker", { y: -8, duration: 2.2, repeat: -1, yoyo: true, ease: "sine.inOut", stagger: 0.3, delay: 2 });
    }, containerRef);

    return () => ctx.revert();
  }, []);

  return (
    <div ref={vantaRef} id="vanta" className="relative min-h-[90vh] rounded-3xl overflow-hidden shadow-2xl border border-[#23304a]/40 my-2">
      <div className="kinetic-veil" />

      <div ref={containerRef} className="kinetic-stage">
        <div className="kinetic-kicker">Real-time sign translation</div>

        <div className="kinetic-sticker kinetic-s1">120ms latency</div>
        <div className="kinetic-sticker kinetic-s2">40+ languages</div>
        <div className="kinetic-sticker kinetic-s3">96% accurate</div>

        <div className="kinetic-word" id="word">
          {wordLetters.map((ch, idx) => (
            <div key={idx} className="kinetic-letter-wrap">
              <svg viewBox="0 0 60 70">
                <path className="fillshape" d={glyphs[ch] || 'M20 60 L40 60'} />
              </svg>
            </div>
          ))}
        </div>

        <div className="kinetic-word-text">is what we translate.</div>

        <p className="kinetic-subcopy">
          SignSense AI reads sign language as it happens and turns it into speech, text, or a signed reply back — so nobody waits for a translator.
        </p>

        <div className="kinetic-actions">
          <button 
            onClick={() => setActiveTab && setActiveTab('recognition')} 
            className="kinetic-btn-primary"
            aria-label="Start signing in recognition engine"
          >
            Start signing
          </button>
          <button 
            onClick={() => setActiveTab && setActiveTab('practice')} 
            className="kinetic-btn-ghost"
            aria-label="Watch practice word builder demo"
          >
            Watch it work
          </button>
        </div>
      </div>
    </div>
  );
}
