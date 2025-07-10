import React, { useRef, useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { CardContent } from "@/components/ui/card";
import { toast } from "@/components/ui/use-toast";
import VideoDropzone from "@/components/features/upload/video/VideoDropzone";
import VideoPreview from "@/components/features/upload/video/VideoPreview";
import { authFetch } from "@/utils/authFetch";
import { useJobPolling } from "@/hooks/useJobPolling";

interface VideoUploadFormProps {
  apiBase: string;
  onUploadSuccess: () => void;
  analysisType?: string | null;
}

const VideoUploadForm: React.FC<VideoUploadFormProps> = ({
  apiBase,
  onUploadSuccess,
  analysisType,
}) => {
  const [file, setFile] = useState<File | null>(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [selectedROI, setSelectedROI] = useState<{
    x: number;
    y: number;
    width: number;
    height: number;
  } | null>(null);
  const [selectedLines, setSelectedLines] = useState<
    | [
        {
          startX: number;
          startY: number;
          endX: number;
          endY: number;
          inDirection?: "UP" | "DOWN" | "LR" | "RL";
        },
        {
          startX: number;
          startY: number;
          endX: number;
          endY: number;
          inDirection?: "UP" | "DOWN" | "LR" | "RL";
        }
      ]
    | null
  >(null);
  const [selectedZones, setSelectedZones] = useState<any[]>([]);
  const [currentJobId, setCurrentJobId] = useState<number | null>(null);
  const [videoDimensions, setVideoDimensions] = useState<{
    width: number;
    height: number;
  } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const {
    job: polledJob,
    isPolling,
    startPolling,
  } = useJobPolling({
    jobId: currentJobId,
    onJobComplete: (completedJob) => {
      console.log("Job completed:", completedJob);
      onUploadSuccess();
    },
  });

  useEffect(() => {
    if (file) {
      const url = URL.createObjectURL(file);
      setVideoPreviewUrl(url);
      setSubmitted(false);

      // Extract video dimensions
      const video = document.createElement("video");
      video.src = url;
      video.addEventListener("loadedmetadata", () => {
        setVideoDimensions({
          width: video.videoWidth,
          height: video.videoHeight,
        });
        console.log(
          "Video dimensions:",
          video.videoWidth,
          "x",
          video.videoHeight
        );
      });

      return () => {
        URL.revokeObjectURL(url);
        setVideoPreviewUrl(null);
      };
    } else {
      setVideoPreviewUrl(null);
      setVideoDimensions(null);
    }
  }, [file]);

  const handleFileSelect = (selectedFile: File) => {
    setFile(selectedFile);
    setSelectedROI(null);
    setSelectedLines(null);
    setSelectedZones([]);
    setCurrentJobId(null);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.type.startsWith("video/")) {
      handleFileSelect(droppedFile);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      handleFileSelect(selectedFile);
    }
  };

  const resetUpload = () => {
    setFile(null);
    setSubmitted(false);
    setSelectedROI(null);
    setSelectedLines(null);
    setSelectedZones([]);
    setCurrentJobId(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleROIChange = (
    roi: { x: number; y: number; width: number; height: number } | null
  ) => {
    setSelectedROI(roi);
  };

  const handleLinesChange = (
    lines:
      | [
          {
            startX: number;
            startY: number;
            endX: number;
            endY: number;
            inDirection?: "UP" | "DOWN" | "LR" | "RL";
          },
          {
            startX: number;
            startY: number;
            endX: number;
            endY: number;
            inDirection?: "UP" | "DOWN" | "LR" | "RL";
          }
        ]
      | null
  ) => {
    setSelectedLines(lines);
  };

  const handleZonesChange = (zones: any[]) => {
    setSelectedZones(zones);
  };

  const submitVideo = async () => {
    if (!file) return;

    // Validate lines for in_out and emergency jobs
    if (analysisType === "in_out" || analysisType === "emergency") {
      if (!selectedLines) {
        toast({
          title: "Lines Required",
          description: "Please draw both counting lines before submitting.",
          variant: "destructive",
        });
        return;
      }
    }

    // Validate zones for lobby crowd detection
    if (analysisType === "lobby_crowd_detection") {
      if (!selectedZones || selectedZones.length === 0) {
        toast({
          title: "Zones Required",
          description:
            "Please create at least one detection zone before submitting.",
          variant: "destructive",
        });
        return;
      }

      const invalidZones = selectedZones.filter(
        (zone) =>
          !zone.name ||
          !zone.points ||
          zone.points.length < 3 ||
          !zone.threshold ||
          zone.threshold <= 0 ||
          !zone.isComplete
      );

      if (invalidZones.length > 0) {
        toast({
          title: "Invalid Zones",
          description:
            "Please ensure all zones have valid names, at least 3 points, and positive thresholds.",
          variant: "destructive",
        });
        return;
      }
    }

    setUploading(true);

    try {
      const formData = new FormData();

      // Handle room readiness with specific endpoint
      if (analysisType === "room_readiness") {
        formData.append("video", file);

        const response = await authFetch("/room-readiness/", {
          method: "POST",
          requireAuth: true,
          body: formData,
        });

        if (response.ok) {
          const result = await response.json();
          setSubmitted(true);
          setCurrentJobId(result.job_id);

          toast({
            title: "Upload Successful!",
            description:
              "Your video has been submitted for room readiness analysis. Processing status will update automatically.",
          });

          // Start polling for this specific job if job_id is provided
          if (result.job_id) {
            startPolling(result.job_id);
          } else {
            onUploadSuccess();
          }
        } else {
          const errorText = await response.text();
          toast({
            title: "Upload Failed",
            description:
              "Server error during room readiness upload. Please try again.",
            variant: "destructive",
          });
        }
        return;
      }

      // Handle lobby crowd detection with specific endpoint
      if (analysisType === "lobby_crowd_detection") {
        formData.append("video", file);

        // Add video dimensions
        if (videoDimensions) {
          formData.append("video_width", videoDimensions.width.toString());
          formData.append("video_height", videoDimensions.height.toString());
        }

        // Convert zones to the required format
        const lobbyZones = selectedZones.map((zone) => ({
          name: zone.name,
          points: zone.points.map((point: any) => [
            Math.round(point.x * (videoDimensions?.width || 1920)),
            Math.round(point.y * (videoDimensions?.height || 1080)),
          ]),
          threshold: zone.threshold,
        }));

        formData.append("lobby_zones", JSON.stringify(lobbyZones));

        console.log("Form data for lobby crowd detection:", {
          video: file.name,
          video_width: videoDimensions?.width,
          video_height: videoDimensions?.height,
          lobby_zones: lobbyZones,
        });

        const response = await authFetch("/lobby-detection/", {
          method: "POST",
          requireAuth: true,
          body: formData,
        });

        if (response.ok) {
          const result = await response.json();
          setSubmitted(true);
          setCurrentJobId(result.job_id);

          toast({
            title: "Upload Successful!",
            description:
              "Your video has been submitted for lobby/crowd detection analysis. Processing status will update automatically.",
          });

          // Start polling for this specific job if job_id is provided
          if (result.job_id) {
            startPolling(result.job_id);
          } else {
            onUploadSuccess();
          }
        } else {
          const errorText = await response.text();
          toast({
            title: "Upload Failed",
            description:
              "Server error during lobby detection upload. Please try again.",
            variant: "destructive",
          });
        }
        return;
      }

      if (analysisType === "in_out") {
        formData.append("video", file);

        // Add video dimensions
        if (videoDimensions) {
          formData.append("video_width", videoDimensions.width.toString());
          formData.append("video_height", videoDimensions.height.toString());
        }

        // Add emergency_lines as a JSON string
        if (selectedLines) {
          const emergencyLines = [
            {
              start_x: selectedLines[0].startX,
              start_y: selectedLines[0].startY,
              end_x: selectedLines[0].endX,
              end_y: selectedLines[0].endY,
              inDirection: selectedLines[0].inDirection || "UP",
            },
            {
              start_x: selectedLines[1].startX,
              start_y: selectedLines[1].startY,
              end_x: selectedLines[1].endX,
              end_y: selectedLines[1].endY,
              inDirection: selectedLines[1].inDirection || "UP",
            },
          ];
          formData.append("emergency_lines", JSON.stringify(emergencyLines));
        }

        console.log("Form data for in_out analysis:", {
          video: file.name,
          video_width: videoDimensions?.width,
          video_height: videoDimensions?.height,
          emergency_lines: selectedLines,
        });

        const response = await authFetch("/emergency-count/", {
          method: "POST",
          requireAuth: true,
          body: formData,
        });

        if (response.ok) {
          const result = await response.json();
          setSubmitted(true);
          setCurrentJobId(result.job_id);

          toast({
            title: "Upload Successful!",
            description:
              "Your video has been submitted for in/out counting analysis. Processing status will update automatically.",
          });

          startPolling(result.job_id);
        } else {
          const errorText = await response.text();
          toast({
            title: "Upload Failed",
            description:
              "Server error during in/out counting upload. Please try again.",
            variant: "destructive",
          });
        }
        return;
      }

      // Handle emergency count with specific endpoint
      if (analysisType === "emergency") {
        formData.append("video", file);

        if (videoDimensions) {
          formData.append("video_width", videoDimensions.width.toString());
          formData.append("video_height", videoDimensions.height.toString());
        }

        if (selectedLines) {
          const emergencyLines = [
            {
              start_x: selectedLines[0].startX,
              start_y: selectedLines[0].startY,
              end_x: selectedLines[0].endX,
              end_y: selectedLines[0].endY,
              inDirection: selectedLines[0].inDirection || "UP",
            },
            {
              start_x: selectedLines[1].startX,
              start_y: selectedLines[1].startY,
              end_x: selectedLines[1].endX,
              end_y: selectedLines[1].endY,
              inDirection: selectedLines[1].inDirection || "UP",
            },
          ];
          formData.append("emergency_lines", JSON.stringify(emergencyLines));
        }

        const response = await authFetch("/emergency-count/", {
          method: "POST",
          requireAuth: true,
          body: formData,
        });

        if (response.ok) {
          const result = await response.json();
          setSubmitted(true);
          setCurrentJobId(result.job_id);

          toast({
            title: "Upload Successful!",
            description:
              "Your video has been submitted for emergency counting analysis. Processing status will update automatically.",
          });

          startPolling(result.job_id);
        } else {
          toast({
            title: "Upload Failed",
            description:
              "Server error during emergency count upload. Please try again.",
            variant: "destructive",
          });
        }
        return;
      }

      if (analysisType === "car_count" || analysisType === "parking_analysis") {
        formData.append("video", file);
        formData.append("job_type", analysisType);

        // Debug logs
        console.log("[VideoUploadForm] Submitting job:", {
          analysisType,
          endpoint:
            analysisType === "car_count" ? "/car-count/" : "/parking-analysis/",
          job_type: analysisType,
          file: file.name,
        });
        for (let [key, value] of formData.entries()) {
          console.log(`[VideoUploadForm] FormData: ${key} =`, value);
        }

        let response;
        if (analysisType === "car_count") {
          response = await authFetch("/car-count/", {
            method: "POST",
            requireAuth: true,
            body: formData,
          });
        } else {
          response = await authFetch("/parking-analysis/", {
            method: "POST",
            requireAuth: true,
            body: formData,
          });
        }

        if (response.ok) {
          const result = await response.json();
          setSubmitted(true);
          setCurrentJobId(result.job_id);

          toast({
            title: "Upload Successful!",
            description:
              "Your video has been submitted for analysis. Processing status will update automatically.",
          });

          startPolling(result.job_id);
        } else {
          toast({
            title: "Upload Failed",
            description: "Server error during upload. Please try again.",
            variant: "destructive",
          });
        }
        return;
      }

      if (analysisType === "pothole_detection") {
        formData.append("video", file);

        const response = await authFetch("/pothole-detection/video/", {
          method: "POST",
          requireAuth: true,
          body: formData,
        });

        if (response.ok) {
          const result = await response.json();
          setSubmitted(true);

          toast({
            title: "Upload Successful!",
            description:
              "Your video has been submitted for pothole detection analysis.",
          });

          onUploadSuccess();
        } else {
          const errorText = await response.text();
          toast({
            title: "Upload Failed",
            description:
              "Server error, please check your video file and try again.",
            variant: "destructive",
          });
        }
        return;
      }

      if (
        analysisType === "pest_detection" ||
        analysisType === "pest_monitoring"
      ) {
        formData.append("video", file);

        const response = await authFetch("/pest-monitoring/video/", {
          method: "POST",
          requireAuth: true,
          body: formData,
        });

        if (response.ok) {
          const result = await response.json();
          setSubmitted(true);

          toast({
            title: "Upload Successful!",
            description:
              "Your video has been submitted for pest monitoring analysis.",
          });

          onUploadSuccess();
        } else {
          const errorText = await response.text();
          toast({
            title: "Upload Failed",
            description:
              "Server error during pest monitoring upload. Please try again.",
            variant: "destructive",
          });
        }
        return;
      }

      if (analysisType === "food_waste") {
        formData.append("video", file);

        const response = await authFetch("/food_waste_estimation/", {
          method: "POST",
          requireAuth: true,
          body: formData,
        });

        if (response.ok) {
          const result = await response.json();
          setSubmitted(true);

          toast({
            title: "Upload Successful!",
            description:
              "Your video has been submitted for food waste estimation analysis.",
          });

          onUploadSuccess();
        } else {
          const errorText = await response.text();
          toast({
            title: "Upload Failed",
            description:
              "Server error during food waste estimation upload. Please try again.",
            variant: "destructive",
          });
        }
        return;
      }

      // Handle other job types with existing logic
      formData.append("input_video", file);

      if (analysisType === "people_count") {
        formData.append("job_type", "people_count");
      } else if (analysisType === "food_waste_estimation") {
        formData.append("job_type", "food_waste_estimation");
      } else if (analysisType) {
        formData.append("job_type", analysisType);
      }

      const response = await authFetch("/jobs/", {
        method: "POST",
        requireAuth: true,
        body: formData,
        headers: {},
      });

      if (response.ok) {
        const contentType = response.headers.get("content-type");
        if (contentType && contentType.includes("application/json")) {
          const result = await response.json();
          setSubmitted(true);

          toast({
            title: "Upload Successful!",
            description: "Your video has been submitted for processing.",
          });

          onUploadSuccess();
        } else {
          toast({
            title: "Server Error",
            description:
              "Received unexpected response from server. Please try again.",
            variant: "destructive",
          });
        }
      } else {
        toast({
          title: "Upload Failed",
          description:
            "Server error, please check your video file and try again.",
        });
      }
    } catch (error) {
      console.error("Video upload error:", error);

      if (error instanceof Error && error.message.includes("Authentication")) {
        toast({
          title: "Authentication Required",
          description: "Please log in to upload videos.",
          variant: "destructive",
        });
        return;
      }

      toast({
        title: "Network Error",
        description:
          "Could not reach the server. Please check your connection.",
        variant: "destructive",
      });
    } finally {
      setUploading(false);
    }
  };

  return (
    <CardContent className="p-6 rounded-b-5xl transition-all duration-300">
      {!file ? (
        <>
          <VideoDropzone
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
            onClick={() => fileInputRef.current?.click()}
          />
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleFileInput}
            className="hidden"
          />
        </>
      ) : (
        videoPreviewUrl && (
          <VideoPreview
            file={file}
            videoPreviewUrl={videoPreviewUrl}
            uploading={uploading || isPolling}
            submitted={submitted}
            submitVideo={submitVideo}
            resetUpload={resetUpload}
            analysisType={analysisType}
            onROIChange={handleROIChange}
            onLinesChange={handleLinesChange}
            onZonesChange={handleZonesChange}
          />
        )
      )}

      {/* Show polling status for jobs with polling */}
      {isPolling && polledJob && (
        <div className="mt-4 p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
          <p className="text-white text-sm">
            Job Status:{" "}
            <span className="font-semibold capitalize">{polledJob.status}</span>
          </p>
          {polledJob.status === "processing" && (
            <p className="text-white/70 text-xs mt-1">
              Processing your video...
            </p>
          )}
        </div>
      )}
    </CardContent>
  );
};

export default VideoUploadForm;
