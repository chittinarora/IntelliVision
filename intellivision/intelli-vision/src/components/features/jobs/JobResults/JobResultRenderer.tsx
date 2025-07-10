import React from "react";
import { Job } from "@/types/job";
import {
  Users,
  Car,
  Activity,
  Construction,
  Bug,
  Bed,
  Shield,
  Hotel,
  AlertTriangle,
} from "lucide-react";
import FoodWasteResult from "./FoodWasteResult";
import RoomReadinessResult from "./RoomReadinessResult";
import { MEDIA_BASE_URL } from "@/constants/api";

interface JobResultRendererProps {
  job: Job;
  mediaInfo?: {
    mediaUrl: string;
    isVideo: boolean;
    hasMedia: boolean;
  };
  videoKey?: string;
  showMedia?: boolean; // Control whether to show media preview
  isCollapsed?: boolean; // Add isCollapsed prop
}

const iconMap: Record<string, React.ReactNode> = {
  people_count: <Users className="w-5 h-5 text-cyan-400" />,
  car_count: <Car className="w-5 h-5 text-blue-400" />,
  parking_analysis: <Car className="w-5 h-5 text-blue-400" />,
  emergency_count: <Activity className="w-5 h-5 text-cyan-400" />,
  pothole_detection: <Construction className="w-5 h-5 text-orange-400" />,
  food_waste_estimation: <Bed className="w-5 h-5 text-green-400" />,
  pest_monitoring: <Bug className="w-5 h-5 text-red-400" />,
  room_readiness: <Hotel className="w-5 h-5 text-indigo-400" />,
  lobby_detection: <Shield className="w-5 h-5 text-indigo-400" />,
  wildlife_detection: <Bug className="w-5 h-5 text-green-400" />,
};

const JobResultRenderer: React.FC<JobResultRendererProps> = ({
  job,
  mediaInfo,
  videoKey,
  showMedia = true, // Default to showing media
  isCollapsed = false, // Default to not collapsed
}) => {
  // Console logs for testing
  // console.log("JobResultRenderer job:", job);
  // console.log("JobResultRenderer results:", job.results);

  if (!job.results) return null;

  // Support both array and object (batch and single)
  const resultsArray = Array.isArray(job.results)
    ? job.results.map((r) => r.data)
    : job.results && "data" in job.results && job.results.data
    ? [job.results.data]
    : [];

  // Helper function to ensure consistent media URL
  const ensureAbsoluteUrl = (path: string) => {
    if (!path) return "";
    if (path.startsWith("http")) return path;
    if (path.startsWith("/media/")) {
      return `${MEDIA_BASE_URL}${path}`;
    }
    return path;
  };

  // Generate media info if not provided
  const finalMediaInfo =
    mediaInfo ||
    (() => {
      let mediaUrl = "";

      // Priority order for media URLs
      if (job.output_url) {
        mediaUrl = ensureAbsoluteUrl(job.output_url);
      } else if (job.output_video) {
        mediaUrl = ensureAbsoluteUrl(job.output_video);
      } else if (job.output_image) {
        mediaUrl = ensureAbsoluteUrl(job.output_image);
      }

      // Determine if it's video or image
      let isVideo = false;
      if (resultsArray[0]?.media_type) {
        isVideo = resultsArray[0].media_type === "video";
      } else if (
        job.output_video ||
        resultsArray[0]?.output_video_path ||
        resultsArray[0]?.annotated_video_path
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
          // Default based on job type - food waste is typically image-based
          isVideo = !["food_waste_estimation", "room_readiness"].includes(
            job.job_type || ""
          );
        }
      }

      return { mediaUrl, isVideo, hasMedia: !!mediaUrl };
    })();

  // Handle job types - support multiple naming conventions
  const jobType = job.job_type;

  // Normalize job type for consistent handling
  let normalizedJobType: string = jobType;
  const normalizationMap: Record<string, string> = {
    "food-waste-estimation": "food_waste_estimation",
    "car-count": "car_count",
    "people-count": "people_count",
    "lobby-detection": "lobby_detection",
  };
  if (
    jobType &&
    Object.prototype.hasOwnProperty.call(normalizationMap, jobType)
  ) {
    normalizedJobType = normalizationMap[jobType];
  }

  // --- Modern card-based result blocks for each job type ---
  const renderResults = () => {
    const cardBase =
      "flex flex-col items-center justify-center p-5 rounded-2xl border border-white/10 bg-white/10 shadow-lg min-w-[160px] min-h-[120px]";
    const labelBase =
      "flex items-center gap-2 text-lg font-bold text-white mb-2";
    const valueBase = "text-4xl font-extrabold text-white mb-1 drop-shadow";
    switch (normalizedJobType) {
      case "people_count": {
        const count = resultsArray[0]?.person_count ?? job.person_count;
        const rawTracks = resultsArray[0]?.raw_track_count;
        return (
          <div className="flex flex-col gap-4 w-full">
            <div className={cardBase}>
              <div className={labelBase}>
                {iconMap[normalizedJobType]}
                People Counted
              </div>
              <span className={valueBase}>{count ?? "N/A"}</span>
              {rawTracks !== undefined && (
                <span className="text-base font-semibold text-cyan-200 mt-1">
                  Raw Tracks: {rawTracks}
                </span>
              )}
            </div>
          </div>
        );
      }
      case "car_count": {
        const count =
          resultsArray[0]?.car_count ?? resultsArray[0]?.vehicle_count;
        const plateCount = resultsArray[0]?.plate_count;
        const plates =
          resultsArray[0]?.plates_detected ||
          resultsArray[0]?.recognized_plates ||
          [];
        const plateTimes = resultsArray[0]?.plate_detection_times || {};
        return (
          <div className="flex flex-col gap-4 w-full">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 bg-cyan-500/20 rounded-full flex items-center justify-center">
                {iconMap[normalizedJobType]}
              </div>
              <div>
                <h3 className="text-xl font-bold text-white">
                  Number Plate Detection
                </h3>
              </div>
            </div>
            <div className={cardBase}>
              <div className={labelBase}>Number Plates</div>
              <span className={valueBase}>{plateCount ?? "N/A"}</span>
              {plateCount !== undefined && (
                <span className="text-base font-semibold text-cyan-200 mt-1">
                  Cars: {count}
                </span>
              )}
            </div>
            {/* Number Plates Card Grid */}
            {plates.length > 0 && (
              <div className="mt-2">
                <h5 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                  Recognized Number Plates
                  <span className="ml-2 px-2 py-0.5 rounded-full bg-purple-500/20 text-cyan-200 text-md font-bold">
                    {plates.length}
                  </span>
                </h5>
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
                  {plates.map((plate: string, idx: number) => (
                    <div
                      key={plate}
                      className="flex flex-row items-center gap-2 bg-cyan-400/30 rounded-2xl px-2 py-2"
                    >
                      <span className="font-mono text-base font-bold text-white tracking-wider">
                        {plate}
                      </span>
                      <span className="text-xs text-white/70 font-semibold ml-2">
                        {plateTimes[plate] ? `${plateTimes[plate]}` : "@ -"}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        );
      }

      case "parking_analysis": {
        // Extract all relevant fields from the new API results
        const entries = resultsArray[0]?.entries;
        const exits = resultsArray[0]?.exits;
        const maxOccupancy = resultsArray[0]?.max_occupancy;
        const finalOccupancy = resultsArray[0]?.final_occupancy;
        const recognizedPlates = resultsArray[0]?.recognized_plates || [];
        const processingFps = resultsArray[0]?.processing_fps;
        const totalFrames = resultsArray[0]?.total_frames;
        const processingTime = resultsArray[0]?.processing_time;
        const plateTimes = resultsArray[0]?.plate_detection_times || {};
        return (
          <div className="flex flex-col gap-4 w-full">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 bg-blue-500/20 rounded-full flex items-center justify-center">
                {iconMap[normalizedJobType]}
              </div>
              <div>
                <h3 className="text-xl font-bold text-white">
                  Parking Analysis
                </h3>
                <p className="text-blue-200/80 text-sm">
                  Entries, Exits, Occupancy, and Plate Recognition
                </p>
              </div>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <div className={cardBase + " bg-blue-400/20"}>
                <div className={labelBase}>Entries</div>
                <span className={valueBase}>{entries ?? "N/A"}</span>
              </div>
              <div className={cardBase + " bg-blue-400/20"}>
                <div className={labelBase}>Exits</div>
                <span className={valueBase}>{exits ?? "N/A"}</span>
              </div>
              <div className={cardBase + " bg-blue-400/20"}>
                <div className={labelBase}>Max Occupancy</div>
                <span className={valueBase}>{maxOccupancy ?? "N/A"}</span>
              </div>
              <div className={cardBase + " bg-blue-400/20"}>
                <div className={labelBase}>Final Occupancy</div>
                <span className={valueBase}>{finalOccupancy ?? "N/A"}</span>
              </div>
              <div className={cardBase + " bg-blue-400/20"}>
                <div className={labelBase}>Unique Vehicles Detected</div>
                <span className={valueBase}>
                  {resultsArray[0]?.vehicle_count ?? "N/A"}
                </span>
              </div>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mt-2">
              <div className={cardBase + " bg-blue-900/10"}>
                <div className={labelBase}>Processing FPS</div>
                <span className={valueBase}>
                  {processingFps ? processingFps.toFixed(2) : "N/A"}
                </span>
              </div>
              <div className={cardBase + " bg-blue-900/10"}>
                <div className={labelBase}>Total Frames</div>
                <span className={valueBase}>{totalFrames ?? "N/A"}</span>
              </div>
              <div className={cardBase + " bg-blue-900/10"}>
                <div className={labelBase}>Processing Time (s)</div>
                <span className={valueBase}>
                  {processingTime ? processingTime.toFixed(2) : "N/A"}
                </span>
              </div>
            </div>
            {/* Recognized Plates Card Grid */}
            {recognizedPlates.length > 0 && (
              <div className="mt-2">
                <h5 className="text-md font-bold text-white mb-2 flex items-center gap-2">
                  <Car className="w-4 h-4 text-purple-300" />
                  Recognized Number Plates
                  <span className="ml-2 px-2 py-0.5 rounded-full bg-purple-500/20 text-purple-200 text-xs font-bold">
                    {recognizedPlates.length}
                  </span>
                </h5>
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
                  {recognizedPlates.map((plate: string, idx: number) => (
                    <div
                      key={plate}
                      className="flex flex-row items-center gap-2 bg-gradient-to-br from-purple-900/40 to-blue-900/20 rounded-xl px-2 py-1 border border-purple-400/20 shadow group hover:scale-[1.02] hover:shadow-lg transition-transform duration-150"
                    >
                      <Car className="w-4 h-4 text-purple-300 group-hover:text-blue-300 transition-colors flex-shrink-0" />
                      <span className="font-mono text-base font-bold text-white tracking-wider">
                        {plate}
                      </span>
                      <span className="text-xs text-white/70 font-semibold ml-2">
                        {plateTimes[plate] ? `@ ${plateTimes[plate]}` : "@ -"}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        );
      }

      case "emergency_count": {
        const inCount = resultsArray[0]?.fast_in_count ?? resultsArray[0]?.in;
        const outCount =
          resultsArray[0]?.fast_out_count ?? resultsArray[0]?.out;

        return (
          <div className="flex flex-col gap-4 w-full">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 bg-cyan-500/20 rounded-full flex items-center justify-center">
                <Activity className="w-5 h-5 text-cyan-400" />
              </div>
              <div>
                <h3 className="text-lg font-bold text-white">Movement Count</h3>
                <p className="text-cyan-200/80 text-sm">In/Out Tracking</p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="bg-cyan-500/30 rounded-xl p-4 border border-cyan-400/20">
                <span className="text-cyan-200 font-semibold text-sm">In</span>
                <div className="text-3xl font-bold text-cyan-100 mt-1">
                  {inCount ?? "N/A"}
                </div>
              </div>

              <div className="bg-emerald-400/30 rounded-xl p-4 border border-emerald-400/20">
                <span className="text-emerald-200 font-semibold text-sm">
                  Out
                </span>
                <div className="text-3xl font-bold text-emerald-100 mt-1">
                  {outCount ?? "N/A"}
                </div>
              </div>
            </div>
          </div>
        );
      }

      case "pothole_detection": {
        const potholeCount =
          resultsArray[0]?.total_potholes ??
          resultsArray[0]?.pothole_count ??
          0;
        return (
          <div className="flex flex-col gap-4 w-full">
            <div className={cardBase}>
              <div className={labelBase}>
                {iconMap[normalizedJobType]}
                Potholes Detected
              </div>
              <span className={valueBase + " text-orange-100"}>
                {potholeCount}
              </span>
            </div>
          </div>
        );
      }
      case "food_waste_estimation": {
        return (
          <div className="flex flex-col gap-4 w-full">
            <FoodWasteResult job={job} isCollapsed={isCollapsed} />
          </div>
        );
      }
      case "pest_monitoring": {
        const snakeCount =
          resultsArray[0]?.detected_snakes ?? resultsArray[0]?.pest_count ?? 0;
        return (
          <div className="flex flex-col gap-4 w-full">
            <div className={cardBase}>
              <div className={labelBase}>
                {iconMap[normalizedJobType]}
                Snakes Detected
              </div>
              <span className={valueBase + " text-red-100"}>{snakeCount}</span>
            </div>
          </div>
        );
      }
      case "room_readiness": {
        return (
          <div className="flex flex-col gap-4 w-full">
            <div className={labelBase}>
              {iconMap[normalizedJobType]}
              Room Readiness
            </div>
            <RoomReadinessResult job={job} />
          </div>
        );
      }
      case "lobby_detection": {
        const zoneCounts = resultsArray[0]?.zone_counts;
        const zoneThresholds = resultsArray[0]?.zone_thresholds || {};
        const alerts = resultsArray[0]?.alerts || [];
        const occupancyCount = resultsArray[0]?.occupancy_count;
        const crowdLevel = resultsArray[0]?.crowd_level;
        const zoneNames = zoneCounts ? Object.keys(zoneCounts) : [];
        const exceededZones = zoneNames.filter(
          (zone) =>
            zoneThresholds[zone] !== undefined &&
            zoneCounts[zone] >= zoneThresholds[zone]
        );
        return (
          <div className="flex flex-col gap-4 w-full">
            <div className={cardBase + " items-start bg-red-400/30"}>
              <div className="flex items-start gap-2 mb-1 w-full">
                <span className="text-lg font-bold text-white text-left">
                  Crowd Alert Triggered
                </span>
              </div>
              <ul className="list-disc ml-3 space-y-1 w-full text-left">
                {alerts.map((alert, idx) => (
                  <li
                    key={idx}
                    className="flex items-start gap-1 text-md text-white"
                  >
                    <span>{alert}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {zoneNames.map((zone) => {
                const count = zoneCounts[zone];
                const threshold = zoneThresholds[zone];
                const exceeded = threshold !== undefined && count >= threshold;
                return (
                  <div key={zone} className={cardBase}>
                    <span className="text-lg font-bold text-white mb-1">
                      {zone}
                    </span>
                    <span
                      className={
                        `mb-2 px-3 py-0.5 rounded-full text-xs font-bold ` +
                        (exceeded
                          ? "bg-red-500/60 text-white animate-pulse"
                          : "bg-cyan-500/60 text-white")
                      }
                    >
                      {exceeded ? "Over Threshold" : "Normal"}
                    </span>
                    <span className="text-3xl font-extrabold text-white drop-shadow-lg">
                      {count}
                    </span>
                    {threshold !== undefined && (
                      <span className="text-xs text-cyan-100/80 mt-0.5">
                        (threshold: {threshold})
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
            {(occupancyCount !== undefined || crowdLevel) && (
              <div
                className={
                  cardBase +
                  " bg-indigo-900/10 border-indigo-400/10 items-start"
                }
              >
                <span className="text-base font-bold text-indigo-100">
                  Occupancy: {occupancyCount ?? "N/A"}
                </span>
                <span className="text-sm text-white/80">
                  Crowd Level: {crowdLevel || "Unknown"}
                </span>
              </div>
            )}
          </div>
        );
      }
      case "wildlife_detection": {
        const wildlifeDetected = resultsArray[0]?.wildlife_detected;
        const wildlifeCount = resultsArray[0]?.wildlife_count ?? 0;
        const wildlifeTypes = resultsArray[0]?.wildlife_types || [];
        return (
          <div className="flex flex-col gap-4 w-full">
            <div className={cardBase}>
              <div className={labelBase}>
                {iconMap[normalizedJobType]}
                Wildlife Detected
              </div>
              <span className={valueBase + " text-green-100"}>
                {wildlifeDetected ? "Yes" : "No"}
              </span>
              <span className="text-base text-orange-200">
                Count: {wildlifeCount}
              </span>
              {wildlifeTypes.length > 0 && (
                <div className="flex flex-wrap gap-2 mt-2">
                  {wildlifeTypes.map((type, idx) => (
                    <span
                      key={idx}
                      className="bg-green-500/20 text-green-300 px-2 py-1 rounded-lg text-xs font-medium"
                    >
                      {type}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        );
      }
      default:
        return <p className="text-white/70">No results available.</p>;
    }
  };

  if (!finalMediaInfo?.hasMedia && !job.results) return null;

  // --- New unified layout: everything in one full-width card ---
  return (
    <div className="space-y-6 scale-in">
      <div className="w-full">
        <div className="p-6 bg-white/10 rounded-3xl border border-white/10 shadow-2xl flex flex-col gap-8">
          <h4 className="text-xl font-extrabold text-white mb-4 tracking-tight">
            Analysis Results
          </h4>

          {/* Media Preview (top) */}
          {showMedia &&
            normalizedJobType !== "room_readiness" &&
            finalMediaInfo?.hasMedia && (
              <div className="rounded-3xl shadow-2xl fade-in-up flex items-center justify-center overflow-hidden w-full mb-4">
                {finalMediaInfo.isVideo ? (
                  <video
                    src={finalMediaInfo.mediaUrl}
                    controls
                    muted
                    loop
                    className="object-contain rounded-3xl max-w-full max-h-[480px]"
                    key={videoKey || `result-${job.id}`}
                    onError={(e) => {
                      console.error(
                        "Video failed to load:",
                        finalMediaInfo.mediaUrl
                      );
                    }}
                  >
                    Your browser does not support the video tag.
                  </video>
                ) : (
                  <img
                    src={finalMediaInfo.mediaUrl}
                    alt="Analysis result"
                    className="object-contain rounded-3xl max-w-full max-h-[480px]"
                    onError={(e) => {
                      console.error(
                        "Image failed to load:",
                        finalMediaInfo.mediaUrl
                      );
                      e.currentTarget.style.display = "none";
                    }}
                  />
                )}
              </div>
            )}
          {/* Results (bottom, full width) */}
          <div className="w-full fade-in-up">{renderResults()}</div>
        </div>
      </div>
    </div>
  );
};

export default JobResultRenderer;
