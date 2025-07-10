
import React from 'react';
import { Button } from "@/components/ui/button";
import { Eye, EyeOff, Download } from 'lucide-react';
import { downloadVideo } from '@/lib/videoUtils';
import AppButton from '@/components/ui/app-button';

interface ActionButtonsProps {
  hasOutputVideo: boolean;
  showVideo?: boolean;
  isCollapsed: boolean;
  onToggleVideo?: () => void;
  onToggleCollapsed: () => void;
  outputVideo?: string;
}

const ActionButtons: React.FC<ActionButtonsProps> = ({
  hasOutputVideo,
  showVideo = false,
  isCollapsed,
  onToggleVideo,
  onToggleCollapsed,
  outputVideo
}) => {
  return (
    <div className="flex flex-col gap-3 w-full">
      {hasOutputVideo && outputVideo && onToggleVideo ? (
        <>
          <Button
            onClick={onToggleVideo}
            className="w-full bg-cyan-500/30 backdrop-blur-lg text-white shadow-xl hover:bg-cyan-400/40 border-2 border-cyan-300/50 hover:border-cyan-200/60 h-12 text-base font-bold rounded-3xl transition-all duration-300 hover:shadow-cyan-500/25 hover:shadow-lg"
          >
            <Eye className="w-5 h-5 mr-3" />
            {showVideo ? 'Hide' : 'View'}
          </Button>
          <Button
            onClick={() => downloadVideo(outputVideo)}
            className="w-full bg-emerald-500/30 backdrop-blur-lg text-white shadow-xl hover:bg-emerald-400/40 border-2 border-emerald-300/50 hover:border-emerald-200/60 h-12 text-base font-bold rounded-3xl transition-all duration-300 hover:shadow-emerald-500/25 hover:shadow-lg"
          >
            <Download className="w-5 h-5 mr-3" />
            Download
          </Button>
        </>
      ) : (
        <Button
          onClick={onToggleCollapsed}
          className="w-full bg-cyan-500/30 backdrop-blur-lg text-white shadow-xl hover:bg-cyan-400/40 border-2 border-cyan-300/50 hover:border-cyan-200/60 h-12 text-base font-bold rounded-3xl transition-all duration-300 hover:shadow-cyan-500/25 hover:shadow-lg"
        >
          {isCollapsed ? <Eye className="w-5 h-5 mr-3" /> : <EyeOff className="w-5 h-5 mr-3" />}
          {isCollapsed ? 'Show Details' : 'Hide Details'}
        </Button>
      )}
    </div>
  );
};

export default ActionButtons;
