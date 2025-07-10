import React from 'react';

const AnimatedBackground = () => {
  return (
    <div className="absolute inset-0 pointer-events-none">
      <div className="absolute top-10 left-5 w-32 h-32 bg-white/3 rounded-full blur-2xl animate-pulse floating"></div>
      <div className="absolute bottom-20 right-1/3 w-48 h-48 bg-cyan-400/2 rounded-full blur-3xl animate-pulse floating" style={{ animationDelay: '2s' }}></div>
      <div className="absolute top-1/3 left-1/5 w-24 h-24 bg-blue-400/3 rounded-full blur-xl animate-pulse floating" style={{ animationDelay: '1s' }}></div>
      <div className="absolute top-1/4 right-12 w-20 h-20 bg-purple-400/2 rounded-full blur-lg animate-pulse floating" style={{ animationDelay: '3s' }}></div>
      <div className="absolute bottom-1/3 left-10 w-28 h-28 bg-emerald-400/2 rounded-full blur-2xl animate-pulse floating" style={{ animationDelay: '4s' }}></div>
    </div>
  );
};

export default AnimatedBackground;
