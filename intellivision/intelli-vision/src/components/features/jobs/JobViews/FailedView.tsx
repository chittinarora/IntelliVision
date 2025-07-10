
import React from 'react';
import { AlertCircle } from 'lucide-react';

const FailedView: React.FC = () => {
  return (
    <div className="pt-6 flex items-center text-base font-semibold text-red-400 bg-red-500/10 backdrop-blur-sm p-6 rounded-3xl border border-red-400/30">
      <AlertCircle className="w-6 h-6 mr-3 flex-shrink-0" />
      <span>Processing failed. Please try uploading another image or check the job list for details.</span>
    </div>
  );
};

export default FailedView;
