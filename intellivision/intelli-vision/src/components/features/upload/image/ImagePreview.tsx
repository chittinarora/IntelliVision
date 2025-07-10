import React, { useState, useMemo, useEffect } from 'react';
import { ImageIcon, Play, Activity, X, Calendar, Trash2, ChevronLeft, ChevronRight, Plus, UploadCloud } from 'lucide-react';
import { Card, CardContent } from "@/components/ui/card";
import AppButton from '@/components/ui/app-button';

// ============================================================================
// 1. Reusable Sub-Component for Image Display
// ============================================================================
interface ImageItemProps {
  file: File;
  previewUrl: string;
  onRemove: () => void;
  className?: string;
}

const ImageItem: React.FC<ImageItemProps> = ({ file, previewUrl, onRemove, className = "" }) => (
  <div className={`relative group w-full h-full rounded-3xl border border-white/30 overflow-hidden backdrop-blur-sm flex-shrink-0 ${className}`}>
    <img
      src={previewUrl}
      alt={`Preview of ${file.name}`}
      className="w-full h-full object-cover transition-transform duration-300 ease-out group-hover:scale-105"
    />
    <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/90 via-black/60 to-transparent p-3">
      <p className="text-white text-sm font-semibold truncate" title={file.name}>
        {file.name.split('.').slice(0, -1).join('.')}
      </p>
      <p className="text-white/70 text-xs">
        {Math.round(file.size / 1024)} KB
      </p>
    </div>
    <button
      onClick={(e) => { e.stopPropagation(); onRemove(); }}
      className="absolute top-3 right-3 w-8 h-8 bg-gray-800/60 hover:bg-gray-900/80 backdrop-blur-xl border border-white/50 text-white rounded-full flex items-center justify-center shadow-xl transition-all duration-200 hover:scale-110 hover:shadow-2xl"
      aria-label={`Remove ${file.name}`}
    >
      <X className="w-4 h-4" />
    </button>
  </div>
);

// ============================================================================
// 2. Empty State Component
// ============================================================================
const EmptyState = () => (
    <div className="w-full h-full min-h-[350px] flex flex-col items-center justify-center bg-white/5 rounded-3xl border-2 border-dashed border-white/20 p-6 text-center">
        <UploadCloud className="w-16 h-16 text-white/40 mb-4" />
        <h4 className="text-lg font-bold text-white">No Images Selected</h4>
        <p className="text-white/60 text-sm">Use the buttons on the left to add images.</p>
    </div>
);


// ============================================================================
// 3. Main ImagePreview Component
// ============================================================================
interface ImagePreviewProps {
  files: File[];
  imagePreviewUrls: string[];
  uploading: boolean;
  submitted: boolean;
  submitImages: () => void;
  resetUpload: () => void;
  removeImage: (index: number) => void; 
  onAddMoreImages: () => void;
  jobIds?: number[];
}

const ImagePreview: React.FC<ImagePreviewProps> = ({
  files,
  imagePreviewUrls,
  uploading,
  submitted,
  submitImages,
  resetUpload,
  removeImage: removeImageProp,
  onAddMoreImages,
  jobIds = [],
}) => {
  const [currentImageIndex, setCurrentImageIndex] = useState(0);

  useEffect(() => {
    if (files.length > 0 && currentImageIndex >= files.length) {
        setCurrentImageIndex(files.length - 1);
    }
  }, [files.length, currentImageIndex]);

  const removeImage = (indexToRemove: number) => {
    removeImageProp(indexToRemove);
  };
  
  const formattedDate = useMemo(() => {
    if (files.length === 0) return "No date";
    const displayDate = new Date("2025-06-19T16:30:00");
    return displayDate.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
  }, [files]);

  const totalSizeDisplay = useMemo(() => {
    const totalSizeKB = files.reduce((sum, file) => sum + Math.round(file.size / 1024), 0);
    return totalSizeKB < 1024 ? `${totalSizeKB} KB` : `${(totalSizeKB / 1024).toFixed(1)} MB`;
  }, [files]);

  const nextImage = () => {
    setCurrentImageIndex((prev) => (prev + 1) % files.length);
  };

  const previousImage = () => {
    setCurrentImageIndex((prev) => (prev - 1 + files.length) % files.length);
  };

  return (
    <div className="fade-in-up">
      <Card className="bg-white/5 backdrop-blur-3xl shadow-2xl border border-white/20 rounded-5xl scale-in card-hover transition-all duration-500 ring-1 ring-inset ring-white/10">
        <CardContent className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Left Column: Info & Action Buttons */}
            <div className="md:col-span-1 flex flex-col">
              <div className="space-y-3 px-4 pt-4 pb-4 bg-white/5 rounded-2xl border border-white/10">
                <h4 className="text-lg font-bold text-white">Upload Details</h4>
                <p className="flex items-center text-base font-semibold text-white/80">
                  <ImageIcon className="w-5 h-5 mr-3 text-white/60 flex-shrink-0" />
                  {files.length} image{files.length !== 1 ? 's' : ''} selected
                </p>
                {files.length > 0 && (
                  <>
                    <p className="flex items-center text-base font-semibold text-white/80">
                      <Calendar className="w-5 h-5 mr-3 text-white/60 flex-shrink-0" />
                      {formattedDate}
                    </p>
                    <p className="flex items-center text-base font-semibold text-white/80">
                      <Trash2 className="w-5 h-5 mr-3 text-white/60 flex-shrink-0" />
                      Total Size: {totalSizeDisplay}
                    </p>
                  </>
                )}
                <div className="pt-2 border-t border-white/10 mt-2">
                  {uploading && (<p className="flex items-center text-base font-semibold text-yellow-400"><Activity className="w-5 h-5 mr-3 animate-spin flex-shrink-0" />Analyzing images...</p>)}
                  {submitted && !uploading && (
                    <div className="space-y-2">
                        <p className="flex items-center text-base font-semibold text-green-400">
                            <Play className="w-5 h-5 mr-3 flex-shrink-0" />
                            Upload completed
                            {jobIds.length > 0 && (<span className="ml-2 text-sm text-white/60">(Job ID: {jobIds[0]})</span>)}
                        </p>
                         {jobIds.length > 0 && (<div className="text-sm text-white/70 pl-8"><p className="text-xs mt-1">Results will appear in the Latest Upload Status section.</p></div>)}
                    </div>
                  )}
                  {!uploading && !submitted && (
                      <p className="flex items-center text-base font-semibold text-white/60">
                        <ImageIcon className="w-5 h-5 mr-3 flex-shrink-0" />
                        {files.length > 0 ? "Ready for analysis" : "Awaiting images"}
                      </p>
                  )}
                </div>
              </div>
              
              <div className="flex flex-col gap-3 mt-6">
                 <AppButton color="primary" onClick={onAddMoreImages}>
                    <Plus className="w-5 h-5 mr-3" />
                    {files.length > 0 ? "Add More" : "Select Images"}
                </AppButton>
                <AppButton color="secondary" onClick={submitImages} disabled={uploading || submitted || files.length === 0}>
                    {uploading ? (<><Activity className="w-5 h-5 mr-3 animate-spin" />Analyzing...</>) : submitted ? (<><Play className="w-5 h-5 mr-3" />Complete</>) : (<><Play className="w-5 h-5 mr-3" />Analyze {files.length} Image{files.length !== 1 ? 's' : ''}</>)}
                </AppButton>
                <AppButton color="tertiary" onClick={resetUpload} disabled={files.length === 0}>
                    <Trash2 className="h-5 w-5 mr-3" />
                    Clear All
                </AppButton>
              </div>
            </div>

            {/* Right Column: Image Previews */}
            <div className="md:col-span-2 flex flex-col justify-center">
              {files.length === 0 && <EmptyState />}
              
              {files.length > 0 && (
                <div className="w-full h-full flex flex-col">
                  <div className="w-full overflow-hidden flex-1">
                    <div 
                      className="flex h-full transition-transform duration-500 ease-in-out"
                      style={{ transform: `translateX(-${currentImageIndex * 100}%)` }}
                    >
                      {files.map((file, index) => (
                        <div key={index} className="w-full h-full flex-shrink-0 px-1">
                          <ImageItem file={file} previewUrl={imagePreviewUrls[index]} onRemove={() => removeImage(index)} />
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="flex items-center justify-center w-full pt-4 space-x-4">
                    {files.length > 1 ? (
                      <button onClick={previousImage} className="w-10 h-10 border border-white/20 text-white/70 hover:text-white hover:bg-white/10 rounded-full flex items-center justify-center transition-all backdrop-blur-sm" aria-label="Previous image">
                        <ChevronLeft className="w-5 h-5" />
                      </button>
                    ) : <div className="w-10 h-10"></div>}

                    <div className="flex justify-center items-center space-x-2">
                      {files.map((_, index) => (
                        <button
                            key={index}
                            onClick={() => setCurrentImageIndex(index)}
                            className={`w-2.5 h-2.5 rounded-full transition-all duration-300 ${
                              index === currentImageIndex
                                ? 'bg-teal-400 shadow-[0_0_8px_theme(colors.teal.400)]'
                                : 'bg-white/20 hover:bg-white/40'
                            }`}
                            aria-label={`Go to image ${index + 1}`}
                        />
                      ))}
                    </div>
                    
                    {files.length > 1 ? (
                      <button onClick={nextImage} className="w-10 h-10 border border-white/20 text-white/70 hover:text-white hover:bg-white/10 rounded-full flex items-center justify-center transition-all backdrop-blur-sm" aria-label="Next image">
                        <ChevronRight className="w-5 h-5" />
                      </button>
                    ) : <div className="w-10 h-10"></div>}
                  </div>
                </div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ImagePreview;
