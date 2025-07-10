import React, { useRef, useEffect, useState, useCallback } from "react";
import { Camera, CameraOff, Loader2 } from "lucide-react";
import { AppButton } from "@/components/common/Common.tsx";

interface WebcamCaptureProps {
  onCapture: (imageBlob: Blob) => void;
  isCapturing?: boolean;
  buttonText?: string;
}

/**
 * WebcamCapture component for capturing face photos
 * Handles camera permissions, video stream, and photo capture functionality
 */
const WebcamCapture: React.FC<WebcamCaptureProps> = ({
  onCapture,
  isCapturing = false,
  buttonText = "Capture Photo",
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [error, setError] = useState<string>("");

  /**
   * Stops the camera stream and cleans up resources
   */
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  }, []);

  /**
   * Initializes and starts the camera stream
   */
  const startCamera = useCallback(async () => {
    try {
      setIsLoading(true);
      setError("");

      // Request camera access with optimal settings for face capture
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user", // Use front-facing camera
        },
      });

      streamRef.current = mediaStream;

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        videoRef.current.onloadedmetadata = () => {
          setIsLoading(false);
          setHasPermission(true);
        };
      }
    } catch (err) {
      setIsLoading(false);
      setHasPermission(false);
      setError(
        "Camera access denied or not available. Please enable camera permissions."
      );
    }
  }, []);

  /**
   * Captures the current video frame as an image blob
   */
  const captureImage = useCallback(() => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");

    if (context) {
      // Set canvas dimensions to match video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      // Draw current video frame to canvas
      context.drawImage(video, 0, 0);

      // Convert canvas to blob and pass to parent component
      canvas.toBlob(
        (blob) => {
          if (blob) {
            onCapture(blob);
          }
        },
        "image/jpeg",
        0.8
      );
    }
  }, [onCapture]);

  // Initialize camera on component mount
  useEffect(() => {
    startCamera();

    // Cleanup on unmount
    return () => {
      stopCamera();
    };
  }, []); // Empty dependency array ensures this runs only once

  // Show error state if camera permission denied
  if (hasPermission === false) {
    return (
      <div className="glass-card rounded-2xl p-8 text-center transition-all duration-500 hover:scale-[1.02]">
        <CameraOff className="w-16 h-16 text-white/60 mx-auto mb-4 animate-pulse" />
        <h3 className="text-white font-semibold mb-2">Camera Required</h3>
        <p className="text-white/80 text-sm mb-4">
          This application requires camera access for face authentication.
          Please enable camera permissions and refresh the page.
        </p>
        <AppButton
          onClick={startCamera}
          className="glass-card text-white hover:bg-white/20 hover:text-white transition-all duration-400 font-semibold rounded-2xl px-6 hover:scale-105"
        >
          Try Again
        </AppButton>
      </div>
    );
  }

  return (
    <div className="glass-card p-6">
      <div className="relative">
        {/* Loading state */}
        {isLoading && (
          <div className="w-full h-64 bg-black/20 rounded-3xl flex items-center justify-center">
            <div className="text-center">
              <Loader2 className="w-8 h-8 text-white/60 mx-auto mb-2 animate-spin" />
              <p className="text-white/80 text-sm">Initializing camera...</p>
            </div>
          </div>
        )}

        {/* Video preview container */}
        <div className="rounded-3xl border border-white/20 shadow-md overflow-hidden w-full h-80 bg-black/20 relative transition-all duration-500 hover:border-white/30">
          <video
            ref={videoRef}
            autoPlay
            muted
            playsInline
            className={`w-full h-full object-cover transition-all duration-500 ${
              isLoading ? "hidden" : "block"
            }`}
          />
        </div>

        {/* Hidden canvas for image capture */}
        <canvas ref={canvasRef} className="hidden" />

        {/* Face detection overlay corners */}
        {!isLoading && (
          <div className="absolute inset-0 rounded-3xl border-2 border-white/30 pointer-events-none transition-all duration-500">
            <div className="absolute top-4 left-4 w-8 h-8 border-l-2 border-t-2 border-white/60 rounded-tl-3xl animate-pulse"></div>
            <div
              className="absolute top-4 right-4 w-8 h-8 border-r-2 border-t-2 border-white/60 rounded-tr-3xl animate-pulse"
              style={{ animationDelay: "0.5s" }}
            ></div>
            <div
              className="absolute bottom-4 left-4 w-8 h-8 border-l-2 border-b-2 border-white/60 rounded-bl-3xl animate-pulse"
              style={{ animationDelay: "1s" }}
            ></div>
            <div
              className="absolute bottom-4 right-4 w-8 h-8 border-r-2 border-b-2 border-white/60 rounded-br-3xl animate-pulse"
              style={{ animationDelay: "1.5s" }}
            ></div>
          </div>
        )}
      </div>

      {/* Capture controls */}
      <div className="mt-4 text-center">
        <p className="text-white/80 text-sm mb-3">
          Position your face within the frame
        </p>
        <AppButton
          onClick={captureImage}
          disabled={isCapturing || isLoading}
          color="primary"
          className="px-6 transition-all duration-400 hover:scale-105"
        >
          <Camera className="w-4 h-4 mr-2" />
          {buttonText}
        </AppButton>
      </div>

      {/* Error message */}
      {error && (
        <p className="text-red-300 text-sm mt-2 text-center animate-fade-in">
          {error}
        </p>
      )}
    </div>
  );
};

export default WebcamCapture;
