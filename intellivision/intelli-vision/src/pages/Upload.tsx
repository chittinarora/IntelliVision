import React, { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import UploadHeader from "@/components/features/upload/UploadHeader";
import { useJobs } from "@/hooks/useJobs";
import { DJANGO_API_BASE } from "@/constants/api";
import UploadPageHeader from "@/components/features/upload/UploadPageHeader";
import VideoUploadSection from "@/components/features/upload/video/VideoUploadSection";
import ImageUploadSection from "@/components/features/upload/image/ImageUploadSection";
import UnifiedUploadSection from "@/components/features/upload/UnifiedUploadSection";
import LastUploadPreview from "@/components/features/current/LastUploadPreview";
import TaskTypeSelector from "@/components/common/TaskTypeSelector";

const Upload = () => {
  const [searchParams] = useSearchParams();
  const type = searchParams.get("type"); // 'people_count', 'emergency', 'food_waste', 'food_waste_estimation', 'pothole_detection', 'car_count', 'parking_analysis','pest_detection', 'room_readiness', 'lobby_crowd_detection'
  const { jobs, fetchJobs } = useJobs();

  const [selectedTaskType, setSelectedTaskType] = useState(
    type || "people_count"
  );

  console.log("Current route type:", type);

  const mostRecentJob = useMemo(() => {
    if (!jobs || jobs.length === 0) {
      return null;
    }
    return [...jobs].sort(
      (a, b) =>
        new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    )[0];
  }, [jobs]);

  // Types that support both image and video uploads
  const supportsBothUploads = [
    "pothole_detection",
    "car_count",
    "parking_analysis",
    "food_waste",
    "pest_detection",
    "pest_monitoring",
    "room_readiness",
  ];

  // Types that only support image upload (remove room_readiness from here)
  const isImageOnlyUpload = type === "food_waste_estimation";

  // Use selectedTaskType for analysis when type is 'number_plate' or vehicle-related
  const analysisType = selectedTaskType;

  return (
    <div className="min-h-screen bg-navy-gradient relative overflow-hidden flex flex-col">
      <UploadHeader />

      {/* Enhanced animated background elements */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-16 left-8 w-36 h-36 bg-white/3 rounded-full blur-2xl animate-pulse floating"></div>
        <div
          className="absolute bottom-32 right-1/4 w-52 h-52 bg-cyan-400/2 rounded-full blur-3xl animate-pulse floating"
          style={{ animationDelay: "2s" }}
        ></div>
        <div
          className="absolute top-1/2 left-1/4 w-28 h-28 bg-blue-400/3 rounded-full blur-xl animate-pulse floating"
          style={{ animationDelay: "1s" }}
        ></div>
        <div
          className="absolute top-1/3 right-12 w-20 h-20 bg-purple-400/2 rounded-full blur-lg animate-pulse floating"
          style={{ animationDelay: "3s" }}
        ></div>
        <div
          className="absolute bottom-1/4 left-12 w-32 h-32 bg-emerald-400/2 rounded-full blur-2xl animate-pulse floating"
          style={{ animationDelay: "4s" }}
        ></div>
        <div
          className="absolute top-3/4 right-1/3 w-24 h-24 bg-pink-400/2 rounded-full blur-xl animate-pulse floating"
          style={{ animationDelay: "5s" }}
        ></div>
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 pt-28 py-12 w-full flex-1 fade-in-up">
        <UploadPageHeader type={type} />

        <div className="w-full space-y-12">
          <div className="animate-fade-in" style={{ animationDelay: "0.2s" }}>
            {/* Show unified upload section for types that support both */}
            {supportsBothUploads.includes(type || "") ? (
              <UnifiedUploadSection
                apiBase={DJANGO_API_BASE}
                onUploadSuccess={() => fetchJobs(true)}
                analysisType={analysisType}
              />
            ) : isImageOnlyUpload ? (
              <ImageUploadSection
                onUploadSuccess={() => fetchJobs(true)}
                analysisType={analysisType}
              />
            ) : (
              <VideoUploadSection
                apiBase={DJANGO_API_BASE}
                onUploadSuccess={() => fetchJobs(true)}
                analysisType={analysisType}
              />
            )}
          </div>

          {mostRecentJob && (
            <div className="animate-fade-in" style={{ animationDelay: "0.4s" }}>
              <LastUploadPreview job={mostRecentJob} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Upload;
