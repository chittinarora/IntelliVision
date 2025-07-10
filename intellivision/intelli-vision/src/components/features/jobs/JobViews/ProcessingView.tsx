
import React from 'react';
import { LoaderCircle } from 'lucide-react';

const ProcessingView: React.FC = () => {
  return (
    <div className="pt-6 flex flex-col items-center justify-center space-y-4 text-center h-auto bg-white/5 backdrop-blur-sm rounded-3xl border border-white/10 p-8">
      <LoaderCircle className="w-16 h-16 text-cyan-400 animate-spin" />
      <div className="w-full max-w-md text-center">
        <p className="text-2xl font-bold text-white tracking-tight">Analyzing Images</p>
        <p className="text-white/70 mt-2 text-base">
          This may take a few moments. The status will update automatically.
        </p>
      </div>
    </div>
  );
};

export default ProcessingView;
