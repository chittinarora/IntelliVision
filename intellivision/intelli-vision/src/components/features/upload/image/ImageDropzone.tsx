
import React from 'react';
import { ImageIcon, Plus, Bed } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface ImageDropzoneProps {
  onDrop: (e: React.DragEvent) => void;
  onDragOver: (e: React.DragEvent) => void;
  onClick: () => void;
  hasImages?: boolean;
  analysisType?: string | null;
}

const ImageDropzone: React.FC<ImageDropzoneProps> = ({ 
  onClick, 
  hasImages = false, 
  analysisType 
}) => {
  // Only show the initial dropzone when no images are uploaded
  if (hasImages) {
    return null;
  }

  const isRoomReadiness = analysisType === 'room_readiness';

  return (
    <div
      className="border-2 border-dashed border-white/30 rounded-3xl p-12 text-center hover:border-white/50 transition-all duration-300 cursor-pointer bg-white/5 hover:bg-white/10 backdrop-blur-sm card-hover scale-in mb-8"
      onClick={onClick}
    >
      <div className="bg-white/10 backdrop-blur-sm rounded-3xl w-20 h-20 mx-auto flex items-center justify-center shadow-xl border border-white/20 fade-in-up relative mb-6">
        {isRoomReadiness ? (
          <Bed className="text-white w-10 h-10" />
        ) : (
          <ImageIcon className="text-white w-10 h-10" />
        )}
        <Plus className="text-white absolute -top-2 -right-2 bg-cyan-500 rounded-full p-1 w-6 h-6" />
      </div>
      <h3 className="text-2xl font-extrabold text-white mb-4">
        {isRoomReadiness ? 'Upload hotel room photo' : 'Drop your images here'}
      </h3>
      <p className="text-white/90 font-semibold text-lg mb-4">
        or click to browse files
      </p>
      <p className="text-sm text-white/70">
        {isRoomReadiness 
          ? 'Take a clear photo showing the room area for AI inspection analysis'
          : 'Supports multiple JPEG, PNG files for batch analysis'
        }
      </p>
    </div>
  );
};

export default ImageDropzone;
