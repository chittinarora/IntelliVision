
import React from 'react';
import { FileVideo } from 'lucide-react';

interface VideoDropzoneProps {
  onDrop: (e: React.DragEvent) => void;
  onDragOver: (e: React.DragEvent) => void;
  onClick: () => void;
}

const VideoDropzone: React.FC<VideoDropzoneProps> = ({ onDrop, onDragOver, onClick }) => {
  return (
    <div
      className="border-2 border-dashed border-white/30 rounded-5xl p-8 text-center hover:border-white/50 transition-all duration-300 cursor-pointer bg-white/5 hover:bg-white/10 backdrop-blur-sm card-hover scale-in"
      onDrop={onDrop}
      onDragOver={onDragOver}
      onClick={onClick}
    >
      <div className="bg-white/10 backdrop-blur-sm p-4 rounded-3xl w-20 h-20 mx-auto mb-6 flex items-center justify-center shadow-xl border border-white/20 fade-in-up">
        <FileVideo className="w-10 h-10 text-white" />
      </div>
      <h3 className="text-xl font-extrabold text-white mb-2">
        Drop your video here
      </h3>
      <p className="text-white/90 mb-4 text-base font-semibold">
        or click to browse files
      </p>
      <p className="text-sm text-white/70">
        Supports MP4, AVI, MOV, and other video formats
      </p>
    </div>
  );
};

export default VideoDropzone;
