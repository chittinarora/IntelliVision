import React, { useState } from "react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Upload as UploadIcon, Video, Image } from "lucide-react";
import VideoUploadForm from "@/components/features/upload/video/VideoUploadForm";
import ImageUploadForm from "@/components/features/upload/image/ImageUploadForm";

interface UnifiedUploadSectionProps {
  apiBase: string;
  onUploadSuccess: () => void;
  analysisType?: string | null;
}

const UnifiedUploadSection: React.FC<UnifiedUploadSectionProps> = ({
  apiBase,
  onUploadSuccess,
  analysisType,
}) => {
  const [isVideoMode, setIsVideoMode] = useState(true);

  const getUploadDescription = () => {
    if (
      analysisType === "pest_detection" ||
      analysisType === "pest_monitoring"
    ) {
      return isVideoMode
        ? "Select your video file for pest detection analysis"
        : "Select your image file for pest detection analysis";
    }
    return isVideoMode
      ? "Select your video file for processing"
      : "Select your image file for analysis";
  };

  return (
    <Card className="bg-white/10 backdrop-blur-3xl shadow-2xl border border-white/20 mb-8 rounded-5xl scale-in card-hover transition-all duration-500">
      <CardHeader className="bg-transparent rounded-t-5xl border-b border-white/20">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <UploadIcon className="w-7 h-7 mr-3 text-white/90" />
            <div>
              <CardTitle className="text-white text-2xl font-bold tracking-tight mb-2">
                Upload Content
              </CardTitle>
              <CardDescription className="text-white/80 text-base font-semibold">
                {getUploadDescription()}
              </CardDescription>
            </div>
          </div>

          {/* Enhanced Toggle Switch */}
          <div className="flex items-center gap-3 bg-white/8 backdrop-blur-md rounded-3xl p-4 border border-white/15 shadow-xl">
            <div
              className={`flex items-center gap-2 px-3 py-2 rounded-xl transition-all duration-300 ${
                isVideoMode
                  ? "bg-cyan-500/20 text-cyan-300 shadow-lg shadow-cyan-500/10"
                  : "text-white/60 hover:text-white/80"
              }`}
            >
              <Video className="w-5 h-5" />
              <span className="text-sm font-semibold">Video</span>
            </div>

            <Switch
              checked={!isVideoMode}
              onCheckedChange={(checked) => setIsVideoMode(!checked)}
              className="data-[state=checked]:bg-gradient-to-r data-[state=checked]:from-green-500 data-[state=checked]:to-emerald-500 data-[state=unchecked]:bg-gradient-to-r data-[state=unchecked]:from-cyan-500 data-[state=unchecked]:to-blue-500 shadow-lg"
            />

            <div
              className={`flex items-center gap-2 px-3 py-2 rounded-xl transition-all duration-300 ${
                !isVideoMode
                  ? "bg-green-500/20 text-green-300 shadow-lg shadow-green-500/10"
                  : "text-white/60 hover:text-white/80"
              }`}
            >
              <Image className="w-5 h-5" />
              <span className="text-sm font-semibold">Image</span>
            </div>
          </div>
        </div>
      </CardHeader>

      {/* Dynamic Content */}
      {isVideoMode ? (
        <VideoUploadForm
          apiBase={apiBase}
          onUploadSuccess={onUploadSuccess}
          analysisType={analysisType}
        />
      ) : (
        <ImageUploadForm
          onUploadSuccess={onUploadSuccess}
          analysisType={analysisType}
        />
      )}
    </Card>
  );
};

export default UnifiedUploadSection;
