import React from "react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Upload as UploadIcon } from "lucide-react";
import ImageUploadForm from "@/components/features/upload/image/ImageUploadForm";

interface ImageUploadSectionProps {
  onUploadSuccess: () => void;
  analysisType?: string | null;
}

const ImageUploadSection: React.FC<ImageUploadSectionProps> = ({
  onUploadSuccess,
  analysisType,
}) => {
  const getUploadDescription = () => {
    if (analysisType === "room_readiness") {
      return "Take or upload a clear photo of the hotel room for AI inspection analysis";
    }
    return "Select your image files for processing";
  };

  return (
    <Card className="bg-white/10 backdrop-blur-3xl shadow-2xl border border-white/20 mb-8 rounded-5xl scale-in card-hover transition-all duration-500">
      <CardHeader className="bg-transparent rounded-t-5xl border-b border-white/20">
        <div className="flex items-center">
          <UploadIcon className="w-7 h-7 mr-3 text-white/90" />
          <div>
            <CardTitle className="text-white text-2xl font-bold tracking-tight mb-2">
              Upload Images
            </CardTitle>
            <CardDescription className="text-white/80 text-base font-semibold">
              {getUploadDescription()}
            </CardDescription>
          </div>
        </div>
      </CardHeader>

      <ImageUploadForm
        onUploadSuccess={onUploadSuccess}
        analysisType={analysisType}
      />
    </Card>
  );
};

export default ImageUploadSection;
