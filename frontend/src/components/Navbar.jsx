import React from 'react';

const Navbar = () => {
  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const scrollToLadder = () => {
    document.getElementById('ladder')?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <nav className="flex justify-center pt-6 pb-8 sticky top-0 z-50">
      <div className="glass-panel px-6 py-3 rounded-full border border-white/10 flex items-center gap-6">
        <button 
          onClick={scrollToTop}
          className="font-bold text-xl text-gray-400 hover:text-white tracking-tight transition-colors"
        >
          Home
        </button>
        <span className="font-bold text-2xl text-white tracking-tight relative">
          AFL Predictor
          <span className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent blur-sm"></span>
        </span>
        <button 
          onClick={scrollToLadder}
          className="font-bold text-xl text-gray-400 hover:text-white tracking-tight transition-colors"
        >
          Ladder
        </button>
      </div>
    </nav>
  );
};

export default Navbar;
