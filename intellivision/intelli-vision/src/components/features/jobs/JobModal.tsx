import React, { useMemo, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { X } from "lucide-react";
import { Job } from "@/types/job";
import JobResultRenderer from "@/components/features/jobs/JobResults/JobResultRenderer";
import CSVDownloadButton from "@/components/features/current/CSVDownloadButton";

interface JobModalProps {
  job: Job | null;
  onClose: () => void;
}

const JobModal: React.FC<JobModalProps> = React.memo(
  ({ job, onClose }) => {
    const handleClose = useCallback(() => {
      onClose();
    }, [onClose]);

    const mediaInfo = useMemo(() => {
      if (!job) return { mediaUrl: "", isVideo: false, hasMedia: false };

      const ensureAbsoluteUrl = (path: string) => {
        if (!path) return "";
        if (path.startsWith("http")) return path;
        if (path.startsWith("/media/")) {
          return `https://medicines-instruction-helmet-young.trycloudflare.com${path}`;
        }
        return path;
      };

      let mediaUrl = "";

      // Use new standardized structure
      if (job.output_url) {
        mediaUrl = ensureAbsoluteUrl(job.output_url);
      } else if (job.output_video) {
        mediaUrl = ensureAbsoluteUrl(job.output_video);
      } else if (job.output_image) {
        mediaUrl = ensureAbsoluteUrl(job.output_image);
      }

      // Use media_type for display decision, with fallback to logic
      let isVideo = false;

      if (job.results?.media_type) {
        isVideo = job.results.media_type === "video";
      } else if (
        job.output_video ||
        job.results?.output_video_path ||
        job.results?.annotated_video_path
      ) {
        isVideo = true;
      } else if (job.output_image) {
        isVideo = false;
      } else {
        // Fallback to extension checking
        const videoExtensions = [".mp4", ".avi", ".mov", ".webm", ".mkv"];
        const imageExtensions = [
          ".jpg",
          ".jpeg",
          ".png",
          ".gif",
          ".webp",
          ".bmp",
        ];
        const urlLower = mediaUrl.toLowerCase();
        if (videoExtensions.some((ext) => urlLower.includes(ext))) {
          isVideo = true;
        } else if (imageExtensions.some((ext) => urlLower.includes(ext))) {
          isVideo = false;
        } else {
          // Default based on job type
          const jobType = job.job_type?.replace("-", "_");
          isVideo = ![
            "food_waste_estimation",
            "food_waste",
            "room_readiness",
          ].includes(jobType || "");
        }
      }

      console.log("JobModal media info:", {
        mediaUrl,
        isVideo,
        hasMedia: !!mediaUrl,
        jobType: job.job_type,
      });
      return { mediaUrl, isVideo, hasMedia: !!mediaUrl };
    }, [
      job?.output_video,
      job?.output_image,
      job?.output_url,
      job?.results,
      job?.id,
      job?.job_type,
    ]);

    const videoKey = useMemo(() => {
      if (!job || !mediaInfo.mediaUrl) {
        return "modal-video-empty";
      }
      const key = `modal-video-${job.id}-${mediaInfo.mediaUrl}`;
      return key;
    }, [job?.id, mediaInfo.mediaUrl]);

    const csvPath = useMemo(() => {
      return job?.csv_report || job?.results?.csv_report_path || null;
    }, [job?.csv_report, job?.results]);

    if (!job) return null;

    return (
      <div className="fixed inset-0 z-50 bg-black/30 backdrop-blur-sm flex items-center justify-center p-4 animate-fade-in">
        <div className="bg-navy-gradient rounded-5xl p-8 max-w-4xl w-full max-h-[90vh] h-full border border-white/20 shadow-2xl backdrop-blur-md animate-scale-in flex flex-col">
          {/* Sticky header */}
          <div
            className="flex justify-between items-center mb-6 sticky top-0 z-10 bg-navy-gradient bg-opacity-95 backdrop-blur-md rounded-t-5xl p-2"
            style={{ backgroundClip: "padding-box" }}
          >
            <h3 className="text-2xl font-bold text-white tracking-tight">
              {mediaInfo.isVideo
                ? `Processed Video (Job #${job.id})`
                : `Analysis Results (Job #${job.id})`}
            </h3>
            <div className="flex items-center gap-3">
              {csvPath && (
                <CSVDownloadButton
                  csvPath={csvPath}
                  className="border-white/30 bg-white/10 backdrop-blur-sm text-white hover:bg-white/20 hover:border-white/40"
                />
              )}
              <Button
                onClick={handleClose}
                variant="outline"
                size="sm"
                className="border-white/30 bg-white/10 backdrop-blur-sm text-white hover:bg-white/20 hover:border-white/40 transition-all duration-300 rounded-2xl"
              >
                <X className="w-4 h-4" />
              </Button>
            </div>
          </div>

          {/* Scrollable content */}
          <div
            className="flex-1 min-h-0 overflow-y-auto pr-2"
            style={{ maxHeight: "calc(90vh - 64px)" }}
          >
            <div className="flex flex-col gap-6">
              {mediaInfo.hasMedia && mediaInfo.mediaUrl && (
                <div className="relative">
                  {mediaInfo.isVideo ? (
                    <video
                      src={mediaInfo.mediaUrl}
                      controls
                      autoPlay
                      muted
                      loop
                      className="w-full h-auto max-h-[70vh] rounded-3xl shadow-lg border border-white/10"
                      key={videoKey}
                      onError={(e) => {
                        console.error(
                          "Video failed to load:",
                          mediaInfo.mediaUrl
                        );
                      }}
                    >
                      Your browser does not support the video tag.
                    </video>
                  ) : (
                    <img
                      src={mediaInfo.mediaUrl}
                      alt="Analysis result"
                      className="w-full h-auto max-h-[70vh] rounded-3xl shadow-lg border border-white/10 object-contain"
                      key={`modal-image-${job.id}-${mediaInfo.mediaUrl}`}
                      onError={(e) => {
                        console.error(
                          "Image failed to load:",
                          mediaInfo.mediaUrl
                        );
                      }}
                    />
                  )}
                </div>
              )}
              <JobResultRenderer
                job={job}
                mediaInfo={mediaInfo}
                showMedia={false}
              />
            </div>
          </div>
        </div>
      </div>
    );
  },
  (prevProps, nextProps) => {
    const same =
      prevProps.job?.id === nextProps.job?.id &&
      prevProps.job?.status === nextProps.job?.status &&
      prevProps.job?.output_video === nextProps.job?.output_video &&
      prevProps.job?.output_image === nextProps.job?.output_image &&
      prevProps.job?.output_url === nextProps.job?.output_url &&
      JSON.stringify(prevProps.job?.results) ===
        JSON.stringify(nextProps.job?.results) &&
      prevProps.onClose === nextProps.onClose;
    return same;
  }
);

JobModal.displayName = "JobModal";

export default JobModal;
