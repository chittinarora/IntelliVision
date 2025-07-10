import React from "react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { Upload as UploadIcon } from "lucide-react";
import VideoUploadForm from "./VideoUploadForm";

interface VideoUploadSectionProps {
  apiBase: string;
  onUploadSuccess: () => void;
  analysisType?: string | null;
}

const VideoUploadSection: React.FC<VideoUploadSectionProps> = ({
  apiBase,
  onUploadSuccess,
  analysisType,
}) => (
  <Card className="glass-intense backdrop-blur-3xl shadow-2xl border border-white/15 mb-8 rounded-4xl scale-in card-hover transition-all duration-500 overflow-hidden group">
    <CardHeader className="bg-gradient-to-r from-white/5 to-white/2 rounded-t-4xl border-b border-white/15 py-6">
      <CardTitle className="text-white flex items-center text-2xl font-bold tracking-tight mb-2">
        <div className="bg-gradient-to-br from-cyan-500/20 to-blue-500/20 backdrop-blur-sm p-2.5 rounded-xl mr-3 shadow-lg border border-white/10 transition-transform duration-300 group-hover:scale-110">
          <UploadIcon className="w-6 h-6 text-cyan-300" />
        </div>
        Input Video
      </CardTitle>
      <CardDescription className="text-white/70 text-base font-medium leading-relaxed ml-11">
        Select your surveillance video file for comprehensive AI-powered
        analysis
      </CardDescription>
    </CardHeader>
    <VideoUploadForm
      apiBase={apiBase}
      onUploadSuccess={onUploadSuccess}
      analysisType={analysisType}
    />
  </Card>
);

export default VideoUploadSection;
