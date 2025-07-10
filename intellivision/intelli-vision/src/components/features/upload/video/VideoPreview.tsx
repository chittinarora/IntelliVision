import React, { useState, useEffect } from "react";
import { FileVideo, Play, Activity, X, Calendar, Clock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Crop } from "react-image-crop";
import ROISelector from "../drawing/ROISelector";
import LineDrawingTool from "../drawing/LineDrawingTool";
import PolygonZoneTool from "@/components/features/upload/drawing/PolygonZoneTool";
import SharedDrawingLayout from "../drawing/SharedDrawingLayout";

// VideoPreview.tsx
// This component provides a preview and annotation interface for video uploads, supporting ROI, line, and zone selection for analytics jobs.
// It manages video preview, annotation state, and submission logic for different analysis types.

// Props for the VideoPreview component
interface VideoPreviewProps {
  file: File; // The video file to preview
  videoPreviewUrl: string; // URL for the video preview
  uploading: boolean; // Whether the video is being uploaded
  submitted: boolean; // Whether the video has been submitted
  submitVideo: () => void; // Callback to submit the video
  resetUpload: () => void; // Callback to reset the upload
  analysisType?: string | null; // Type of analysis (e.g., in_out, emergency, lobby_crowd_detection)
  onROIChange?: (
    roi: { x: number; y: number; width: number; height: number } | null
  ) => void; // Callback for ROI selection
  onLinesChange?: (
    lines:
      | [
          { startX: number; startY: number; endX: number; endY: number },
          { startX: number; startY: number; endX: number; endY: number }
        ]
      | null
  ) => void; // Callback for line selection
  onZonesChange?: (zones: any[]) => void; // Callback for zone selection
}

const VideoPreview: React.FC<VideoPreviewProps> = ({
  file,
  videoPreviewUrl,
  uploading,
  submitted,
  submitVideo,
  resetUpload,
  analysisType,
  onROIChange,
  onLinesChange,
  onZonesChange,
}) => {
  // State for video duration
  const [duration, setDuration] = useState<number | null>(null);
  // State for selected ROI
  const [selectedROI, setSelectedROI] = useState<{
    x: number;
    y: number;
    width: number;
    height: number;
  } | null>(null);
  // State for selected lines (for line drawing jobs)
  const [selectedLines, setSelectedLines] = useState<
    | [
        { startX: number; startY: number; endX: number; endY: number },
        { startX: number; startY: number; endX: number; endY: number }
      ]
    | null
  >(null);
  // State for selected zones (for zone jobs)
  const [selectedZones, setSelectedZones] = useState<any[]>([]);
  // State for crop (ROI selection)
  const [crop, setCrop] = useState<Crop>({
    unit: "%",
    x: 25,
    y: 25,
    width: 50,
    height: 50,
  });

  // Determine job type for annotation tools
  const isLineDrawingJob =
    analysisType === "in_out" || analysisType === "emergency";
  const isPolygonZoneJob = analysisType === "lobby_crowd_detection";

  // Extract video duration when preview URL changes
  useEffect(() => {
    if (videoPreviewUrl) {
      const video = document.createElement("video");
      video.src = videoPreviewUrl;
      video.onloadedmetadata = () => {
        setDuration(video.duration);
      };
    }
  }, [videoPreviewUrl]);

  // Handle ROI selection
  const handleROISelect = (
    roi: { x: number; y: number; width: number; height: number } | null
  ) => {
    setSelectedROI(roi);
    onROIChange?.(roi);
  };

  // Handle line selection
  const handleLinesSelect = (
    lines:
      | [
          { startX: number; startY: number; endX: number; endY: number },
          { startX: number; startY: number; endX: number; endY: number }
        ]
      | null
  ) => {
    setSelectedLines(lines);
    onLinesChange?.(lines);
  };

  // Handle zone selection
  const handleZonesSelect = (zones: any[]) => {
    setSelectedZones(zones);
    onZonesChange?.(zones);
  };

  // Validation helpers for annotation requirements
  const isROIValid = () => {
    return true;
  };
  const isLinesValid = () => {
    if (!isLineDrawingJob) return true;
    return selectedLines !== null;
  };
  const isZonesValid = () => {
    if (!isPolygonZoneJob) return true;
    return (
      selectedZones.length > 0 &&
      selectedZones.every(
        (zone) =>
          zone.name &&
          zone.points &&
          zone.points.length >= 3 &&
          zone.threshold &&
          zone.threshold > 0 &&
          zone.isComplete
      )
    );
  };

  // Whether the video can be submitted
  const canSubmit = isROIValid() && isLinesValid() && isZonesValid();

  // Get reason for blocking submission (if any)
  const getSubmitBlockedReason = () => {
    if (isLineDrawingJob && !selectedLines)
      return "Please draw both counting lines to continue";
    if (isPolygonZoneJob && !isZonesValid()) {
      if (selectedZones.length === 0)
        return "Please create at least one detection zone";
      const incompleteZones = selectedZones.filter(
        (zone) => !zone.isComplete || zone.points.length < 3
      );
      if (incompleteZones.length > 0)
        return "Please complete all zones with at least 3 points";
      const invalidZones = selectedZones.filter(
        (zone) => !zone.name || !zone.threshold || zone.threshold <= 0
      );
      if (invalidZones.length > 0)
        return "Please set valid names and thresholds for all zones";
    }
    return null;
  };

  // Format file date for display
  const formattedDate = (() => {
    const jobDate = new Date(file.lastModified);
    const today = new Date();

    const isToday =
      jobDate.getFullYear() === today.getFullYear() &&
      jobDate.getMonth() === today.getMonth() &&
      jobDate.getDate() === today.getDate();

    if (isToday) {
      return `Today at ${jobDate.toLocaleTimeString("en-US", {
        hour: "2-digit",
        minute: "2-digit",
        hour12: true,
      })}`;
    } else {
      return jobDate.toLocaleDateString("en-US", {
        year: "numeric",
        month: "short",
        day: "numeric",
      });
    }
  })();

  // Format duration for display
  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);

    if (mins === 0) return `${secs} second${secs !== 1 ? "s" : ""}`;
    if (secs === 0) return `${mins} minute${mins !== 1 ? "s" : ""}`;
    return `${mins} minute${mins !== 1 ? "s" : ""} and ${secs} second${
      secs !== 1 ? "s" : ""
    }`;
  };

  return (
    <div className="space-y-6 scale-in">
      {/* General layout: file info, video preview, submit/clear */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-start">
        {/* Left Column: File Info & Controls */}
        <div className="md:col-span-1 space-y-4">
          <div className="p-4 bg-white/5 rounded-2xl border border-white/10 space-y-2">
            <h4 className="text-lg font-bold text-white">Selected File</h4>
            <p className="flex items-start text-base font-semibold text-white/80">
              <FileVideo className="w-5 h-5 mr-3 mt-1 text-white/60 flex-shrink-0" />
              <span className="truncate break-all" title={file.name}>
                {file.name}
              </span>
            </p>
            <p className="flex items-center text-base font-semibold text-white/80">
              <Calendar className="w-5 h-5 mr-3 text-white/60 flex-shrink-0" />
              {formattedDate}
            </p>
            {duration && (
              <p className="flex items-center text-base font-semibold text-white/80">
                <Clock className="w-5 h-5 mr-3 text-white/60 flex-shrink-0" />
                Duration: {formatDuration(duration)}
              </p>
            )}
          </div>
          <div className="flex flex-col gap-3">
            <Button
              onClick={submitVideo}
              disabled={uploading || submitted || !canSubmit}
              className="flex-1 bg-cyan-500/60 backdrop-blur-lg text-white shadow-lg hover:bg-cyan-500/70 h-12 text-base font-extrabold button-hover rounded-3xl border border-cyan-400 transition-all disabled:opacity-50"
            >
              {uploading ? (
                <>
                  <Activity className="w-5 h-5 mr-3 animate-spin" />
                  Submitting...
                </>
              ) : submitted ? (
                <>
                  <Play className="w-5 h-5 mr-3" />
                  Submitted
                </>
              ) : (
                <>
                  <Play className="w-5 h-5 mr-3" />
                  Submit
                </>
              )}
            </Button>
            {!canSubmit && (
              <p className="text-orange-300 text-xs text-center">
                {getSubmitBlockedReason()}
              </p>
            )}
            <Button
              variant="ghost"
              onClick={resetUpload}
              className="flex-1 bg-white/10 text-white hover:bg-white/20 h-12 text-base font-extrabold button-hover rounded-3xl border border-white/20"
            >
              <X className="h-5 w-5 mr-3" />
              Clear
            </Button>
          </div>
        </div>
        {/* Right Column: Video Preview */}
        <div className="md:col-span-2 flex flex-col min-h-full">
          <div className="relative aspect-video bg-black rounded-3xl border border-white/20 shadow-2xl fade-in-up mb-6">
            <video
              src={videoPreviewUrl}
              controls
              autoPlay
              muted
              loop
              className="w-full h-full object-contain rounded-3xl"
            />
            <Button
              variant="ghost"
              size="icon"
              onClick={resetUpload}
              className="absolute -top-3 -right-3 z-10 h-10 w-10 rounded-full border border-white/20 bg-white/70 text-slate-800 backdrop-blur-sm transition-all duration-300 hover:scale-125"
              aria-label="Clear video"
            >
              <X className="h-5 w-5" />
            </Button>
          </div>
        </div>
      </div>

      {/* Drawing Tools - Only show the tool without SharedDrawingLayout wrapper */}
      {isPolygonZoneJob && (
        <PolygonZoneTool videoFile={file} onZonesChange={handleZonesSelect} />
      )}
      {isLineDrawingJob && (
        <LineDrawingTool
          videoFile={file}
          onLinesChange={handleLinesSelect}
          selectedLines={selectedLines}
        />
      )}
    </div>
  );
};

export default VideoPreview;
// End of VideoPreview.tsx
