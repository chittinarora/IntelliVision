
import React, { useState, useRef, useEffect } from 'react';
import ReactCrop, { Crop, PixelCrop } from 'react-image-crop';
import 'react-image-crop/dist/ReactCrop.css';
import { Button } from '@/components/ui/button';
import { RotateCcw, AlertTriangle, CheckCircle } from 'lucide-react';

interface ROISelectorProps {
  videoFile: File;
  crop: Crop;
  setCrop: (crop: Crop) => void;
  onROISelect: (roi: { x: number; y: number; width: number; height: number } | null) => void;
  selectedROI: { x: number; y: number; width: number; height: number } | null;
}

const ROISelector: React.FC<ROISelectorProps> = ({ 
  videoFile, 
  crop, 
  setCrop, 
  onROISelect, 
  selectedROI 
}) => {
  const [imageSrc, setImageSrc] = useState<string>('');
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    const extractFrame = async () => {
      const video = document.createElement('video');
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');

      video.src = URL.createObjectURL(videoFile);
      video.currentTime = 1; // Get frame at 1 second

      video.onloadeddata = () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx?.drawImage(video, 0, 0);
        
        const imageDataUrl = canvas.toDataURL('image/jpeg', 0.8);
        setImageSrc(imageDataUrl);
        
        URL.revokeObjectURL(video.src);
      };
    };

    extractFrame();
  }, [videoFile]);

  const handleCropComplete = (crop: PixelCrop) => {
    if (imgRef.current && crop.width && crop.height && crop.width > 0 && crop.height > 0) {
      const image = imgRef.current;
      const normalizedROI = {
        x: crop.x / image.clientWidth,
        y: crop.y / image.clientHeight,
        width: crop.width / image.clientWidth,
        height: crop.height / image.clientHeight,
      };
      onROISelect(normalizedROI);
    }
  };

  const handleCropChange = (crop: Crop) => {
    setCrop(crop);
  };

  const resetROI = () => {
    const newCrop = {
      unit: '%' as const,
      x: 25,
      y: 25,
      width: 50,
      height: 50,
    };
    setCrop(newCrop);
    onROISelect(null);
  };

  // Check if ROI meets minimum size requirement - fix area calculation
  const isROITooSmall = () => {
    if (!selectedROI) return false;
    const minArea = 0.05; // 5% of frame area
    const roiArea = selectedROI.width * selectedROI.height; // Actual area calculation
    return roiArea < minArea;
  };

  const getROIAreaPercentage = () => {
    if (!selectedROI) return 0;
    return (selectedROI.width * selectedROI.height * 100).toFixed(1);
  };

  if (!imageSrc) {
    return (
      <div className="flex items-center justify-center h-64 bg-white/5 rounded-2xl border border-white/10">
        <div className="text-white/60">Extracting video frame...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h4 className="text-xl font-bold text-white">Select Region of Interest</h4>
        <Button
          variant="ghost"
          size="sm"
          onClick={resetROI}
          className="bg-white/10 text-white hover:bg-white/20 rounded-xl border border-white/20 transition-all"
        >
          <RotateCcw className="w-4 h-4 mr-2" />
          Reset
        </Button>
      </div>
      
      {/* Main ROI Card */}
      <div className="bg-white/5 rounded-2xl border border-white/10 overflow-hidden">
        {/* Instructions */}
        <div className="p-6 border-b border-white/10">
          <p className="text-white/80 text-base leading-relaxed">
            Draw a rectangle around the area where people will enter/exit. This defines the counting zone for accurate person detection.
          </p>
        </div>
        
        {/* Image Crop Area */}
        <div className="p-6">
          <div className="relative max-w-full overflow-hidden rounded-xl bg-black/20">
            <ReactCrop
              crop={crop}
              onChange={handleCropChange}
              onComplete={handleCropComplete}
              aspect={undefined}
              className="max-w-full"
              minWidth={20}
              minHeight={20}
            >
              <img
                ref={imgRef}
                src={imageSrc}
                alt="Video frame for ROI selection"
                className="max-w-full h-auto rounded-xl"
                style={{ maxHeight: '400px' }}
              />
            </ReactCrop>
          </div>
        </div>
        
        {/* Status Card */}
        {selectedROI && (
          <div className="p-6 pt-0">
            <div className={`p-4 rounded-xl border-2 transition-all ${
              isROITooSmall() 
                ? 'bg-orange-500/10 border-orange-500/30 shadow-orange-500/10' 
                : 'bg-emerald-500/10 border-emerald-500/30 shadow-emerald-500/10'
            } shadow-lg`}>
              <div className="flex items-start gap-3">
                {isROITooSmall() ? (
                  <AlertTriangle className="w-5 h-5 text-orange-400 flex-shrink-0 mt-0.5" />
                ) : (
                  <CheckCircle className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-0.5" />
                )}
                <div className="flex-1">
                  <p className={`font-semibold text-base mb-1 ${
                    isROITooSmall() ? 'text-orange-300' : 'text-emerald-300'
                  }`}>
                    {isROITooSmall() ? 'Region Too Small' : 'Region Selected'}
                  </p>
                  <div className="space-y-1">
                    <p className={`text-sm ${
                      isROITooSmall() ? 'text-orange-200/80' : 'text-emerald-200/80'
                    }`}>
                      Dimensions: {Math.round(selectedROI.width * 100)}% × {Math.round(selectedROI.height * 100)}%
                    </p>
                    <p className={`text-sm ${
                      isROITooSmall() ? 'text-orange-200/80' : 'text-emerald-200/80'
                    }`}>
                      Total area: {getROIAreaPercentage()}% of frame
                    </p>
                    {isROITooSmall() && (
                      <p className="text-xs text-orange-300/90 mt-2">
                        Please select a larger area (minimum 5% of total frame area required)
                      </p>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ROISelector;
