import React, { useState, useMemo, useCallback } from "react";
import { File, Calendar, Clock, Users, Eye, EyeOff } from "lucide-react";
import AppButton from "@/components/ui/app-button";
import { Job } from "@/types/job";
import { formatJobDate, formatDuration, getJobFilename } from "@/lib/jobUtils";
import JobResultRenderer from "@/components/features/jobs/JobResults/JobResultRenderer";

const DoneView: React.FC<{ job: Job }> = React.memo(
  ({ job }) => {
    const [isCollapsed, setIsCollapsed] = useState(true);

    const jobInfo = useMemo(() => {
      const filename = getJobFilename(job.input_video);
      const formattedDate = formatJobDate(job.created_at);
      const peopleCount = (() => {
        if (job.results?.person_count !== undefined) {
          return job.results.person_count;
        }
        if (typeof job.person_count === "number") {
          return job.person_count;
        }
        return null;
      })();
      return { filename, formattedDate, peopleCount };
    }, [job.input_video, job.created_at, job.results, job.person_count]);

    const handleToggleDetails = useCallback(
      () => setIsCollapsed((prev) => !prev),
      []
    );

    const mediaInfo = useMemo(() => {
      const ensureAbsoluteUrl = (path: string) => {
        if (!path) return "";
        if (path.startsWith("http")) return path;
        if (path.startsWith("/media/")) {
          return `https://medicines-instruction-helmet-young.trycloudflare.com${path}`;
        }
        return path;
      };

      // Determine media URL using new standardized structure
      let mediaUrl = "";

      // Priority 1: New standardized output_url
      if (job.output_url) {
        mediaUrl = ensureAbsoluteUrl(job.output_url);
      }
      // Priority 2: New standardized output_video
      else if (job.output_video) {
        mediaUrl = ensureAbsoluteUrl(job.output_video);
      }
      // Priority 3: New standardized output_image
      else if (job.output_image) {
        mediaUrl = ensureAbsoluteUrl(job.output_image);
      }
      // Priority 4: Legacy paths in results
      else if (job.results?.output_path) {
        mediaUrl = ensureAbsoluteUrl(job.results.output_path);
      } else if (job.results?.annotated_video_path) {
        mediaUrl = ensureAbsoluteUrl(job.results.annotated_video_path);
      } else if (job.results?.output_video_path) {
        mediaUrl = ensureAbsoluteUrl(job.results.output_video_path);
      }

      // Determine if it's video or image
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
        // Fallback: check file extension
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

      console.log("DoneView media info:", {
        mediaUrl,
        isVideo,
        hasMedia: !!mediaUrl,
        jobType: job.job_type,
      });
      return { mediaUrl, isVideo, hasMedia: !!mediaUrl };
    }, [
      job.results,
      job.output_video,
      job.output_image,
      job.output_url,
      job.job_type,
    ]);

    const JobInfoBlock = useMemo(
      () => (
        <div className="bg-white/5 backdrop-blur-sm rounded-3xl border border-white/10 p-5 flex items-center gap-x-5 overflow-hidden">
          <File className="w-8 h-8 text-cyan-400 flex-shrink-0" />
          <div className="overflow-hidden flex-1">
            <p
              className="font-semibold text-white truncate"
              title={jobInfo.filename}
            >
              {jobInfo.filename}
            </p>
            <div className="flex items-center flex-wrap gap-x-4 gap-y-1 text-sm text-white/70 mt-1">
              <span className="flex items-center">
                <Calendar className="w-4 h-4 mr-1.5" />
                {jobInfo.formattedDate}
              </span>
              {jobInfo.peopleCount !== null && (
                <>
                  <span className="text-white/30">&middot;</span>
                  <span className="flex items-center">
                    <Users className="w-4 h-4 mr-1.5" />
                    {jobInfo.peopleCount} people
                  </span>
                </>
              )}
            </div>
          </div>
        </div>
      ),
      [jobInfo]
    );

    const videoKey = useMemo(() => {
      return `done-view-${job.id}-${mediaInfo.mediaUrl}`;
    }, [job.id, mediaInfo.mediaUrl]);

    return (
      <div className="space-y-6">
        <div className="flex flex-col lg:grid lg:grid-cols-4 items-stretch gap-6 w-full">
          <div className="lg:col-span-3 w-full">{JobInfoBlock}</div>
          <div className="lg:col-span-1 flex flex-col gap-3">
            <AppButton onClick={handleToggleDetails} color="secondary">
              {isCollapsed ? (
                <Eye className="w-5 h-5 mr-3" />
              ) : (
                <EyeOff className="w-5 h-5 mr-3" />
              )}
              {isCollapsed ? "Show Details" : "Hide Details"}
            </AppButton>
          </div>
        </div>

        <JobResultRenderer
          job={job}
          mediaInfo={mediaInfo}
          videoKey={videoKey}
          isCollapsed={isCollapsed}
        />
      </div>
    );
  },
  (prevProps, nextProps) => {
    // Custom comparison to prevent unnecessary re-renders
    const same =
      prevProps.job.id === nextProps.job.id &&
      prevProps.job.status === nextProps.job.status &&
      JSON.stringify(prevProps.job.results) ===
        JSON.stringify(nextProps.job.results);
    return same;
  }
);

export default DoneView;
