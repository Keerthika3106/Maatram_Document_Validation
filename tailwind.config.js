/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#FFC928',
          light: '#FFD75A',
          dark: '#E6B320',
        },
        pale: '#FFF6D8',
        background: '#FFFEF9',
        card: '#FFFFFF',
        border: '#EAE6D8',
        // Flattened for Tailwind class generation
        'text-primary': '#171717',
        'text-secondary': '#777777',
        'success-bg': '#E5F5EB',
        'success-text': '#27834C',
        'success-border': '#B7DFC5',
        'failure-bg': '#FCE8E7',
        'failure-text': '#C64B45',
        'failure-border': '#F5C5C3',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.5rem',
        '4xl': '2rem',
      },
      boxShadow: {
        'card': '0 1px 4px 0 rgba(0,0,0,0.06), 0 2px 12px 0 rgba(0,0,0,0.04)',
        'card-hover': '0 4px 16px 0 rgba(0,0,0,0.10)',
        'nav': '0 2px 12px 0 rgba(0,0,0,0.07)',
      },
    },
  },
  plugins: [],
}
