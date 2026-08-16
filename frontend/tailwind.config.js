/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          bg: "#08060F",
          dark: "#08060F",
          surface: "rgba(18, 14, 34, 0.75)",
          card: "rgba(24, 18, 45, 0.65)",
          border: "rgba(156, 150, 196, 0.15)",
          cyan: "#00F0FF",      // Primary accent
          magenta: "#FF2E9A",   // Secondary accent
          lime: "#C6FF3C",      // Accent 3 (sparingly)
          primary: "#00F0FF",
          secondary: "#FF2E9A",
          accent: "#C6FF3C",
          ink: "#F4F2FF",
          muted: "#9C96C4",
        }
      },
      fontFamily: {
        display: ['Fraunces', 'serif'],
        sans: ['Plus Jakarta Sans', 'system-ui', 'sans-serif'],
        body: ['Plus Jakarta Sans', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      boxShadow: {
        'glow-cyan': '0 0 26px -4px rgba(0, 240, 255, 0.7)',
        'glow-magenta': '0 0 26px -4px rgba(255, 46, 154, 0.7)',
        'glow-lime': '0 0 26px -4px rgba(198, 255, 60, 0.7)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow-pulse': 'glowPulse 2s ease-in-out infinite alternate',
        'ambient-glow': 'ambientGlow 4s ease-in-out infinite alternate',
      },
      keyframes: {
        glowPulse: {
          '0%': { boxShadow: '0 0 15px rgba(0, 240, 255, 0.3)' },
          '100%': { boxShadow: '0 0 35px rgba(0, 240, 255, 0.7)' },
        },
        ambientGlow: {
          '0%': { opacity: '0.4', transform: 'scale(1)' },
          '100%': { opacity: '0.8', transform: 'scale(1.05)' },
        }
      }
    },
  },
  plugins: [],
}
