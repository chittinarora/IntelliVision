import React from "react";
import { Job } from "@/types/job";
import {
  CheckCircle,
  XCircle,
  AlertTriangle,
  Bed,
  Hotel,
  ClipboardList,
} from "lucide-react";
import { MEDIA_BASE_URL } from "@/constants/api";

interface RoomReadinessResultProps {
  job: Job;
}

const RoomReadinessResult: React.FC<RoomReadinessResultProps> = ({ job }) => {
  if (!job || !job.results || !job.results.data) return null;

  // --- New backend format ---
  const data = job.results.data;
  const summary = {
    readiness_score: data.readiness_score,
    status: data.status,
    overall_status: data.overall_status,
    rooms_analyzed: data.rooms_analyzed,
    rooms_ready: data.rooms_ready,
    total_issues: data.total_issues,
  };
  const rooms = data.rooms || {};

  // Get video or image preview (if any)
  let mediaUrl = job.output_video || job.output_image || job.output_url;
  if (mediaUrl && typeof mediaUrl === "string" && mediaUrl.startsWith("/")) {
    mediaUrl = `${MEDIA_BASE_URL}${mediaUrl}`;
  }
  const isVideo = !!job.output_video;

  // Helper for status display
  const getStatusDisplay = (score?: number, status?: string) => {
    const isReady = status === "Guest Ready" || score === 100;
    return {
      isReady,
      statusText: isReady ? "Guest Ready" : "Not Guest Ready",
      statusColorClass: isReady ? "text-emerald-400" : "text-red-400",
      statusBgColorClass: isReady
        ? "bg-emerald-500/20 border-emerald-500/40"
        : "bg-red-500/20 border-red-500/40",
      scoreTextColorClass: isReady ? "text-emerald-400" : "text-red-400",
    };
  };

  // Helper for checklist item status
  const getChecklistItemStatus = (item: any) => {
    const status = item.status?.toLowerCase();
    switch (status) {
      case "ready":
        return {
          isOk: true,
          icon: CheckCircle,
          colorClass: "text-emerald-400",
          bgClass: "bg-emerald-500/10 border-emerald-500/20",
          statusText: "Ready",
        };
      case "needs_attention":
        return {
          isOk: false,
          icon: AlertTriangle,
          colorClass: "text-amber-400",
          bgClass: "bg-amber-200/10 border-amber-500/20",
          statusText: "Needs Attention",
        };
      default:
        return {
          isOk: false,
          icon: XCircle,
          colorClass: "text-red-400",
          bgClass: "bg-red-500/30 border-red-500/20",
          statusText: "Issue Found",
        };
    }
  };

  // --- Render ---
  return (
    <div className="space-y-8 w-full">
      {/* Summary Card */}
      <div>
        <div className="text-lg font-bold text-cyan-200 mb-2">Summary</div>
        <div
          className={`rounded-2xl ${
            getStatusDisplay(summary.readiness_score, summary.status)
              .statusBgColorClass
          } border p-6 flex flex-col items-start shadow-lg w-full mx-auto`}
        >
          <div className="grid grid-cols-2 gap-6 items-center mb-1 w-full">
            <span
              className={`text-xl font-extrabold ${
                getStatusDisplay(summary.readiness_score, summary.status)
                  .statusColorClass
              }`}
            >
              {
                getStatusDisplay(summary.readiness_score, summary.status)
                  .statusText
              }
            </span>
            <span
              className={`text-xl font-extrabold ${
                getStatusDisplay(summary.readiness_score, summary.status)
                  .scoreTextColorClass
              } text-right`}
            >
              {summary.readiness_score}/100
            </span>
          </div>
          <span className="text-white/70 text-base">
            {summary.overall_status} &middot; Rooms analyzed:{" "}
            {summary.rooms_analyzed}, Ready: {summary.rooms_ready}, Issues:{" "}
            {summary.total_issues}
          </span>
        </div>
      </div>

      {/* Media Preview (video or image) */}
      {mediaUrl && (
        <div className="rounded-3xl shadow-2xl fade-in-up flex items-center justify-center overflow-hidden w-full mb-4">
          {isVideo ? (
            <video
              src={mediaUrl}
              controls
              muted
              loop
              className="object-contain rounded-3xl max-w-full max-h-[480px]"
            >
              Your browser does not support the video tag.
            </video>
          ) : (
            <img
              src={mediaUrl}
              alt="Room readiness result"
              className="object-contain rounded-3xl max-w-full max-h-[480px]"
            />
          )}
        </div>
      )}

      {/* Room-wise Results */}
      <div className="space-y-8">
        {Object.entries(rooms).map(([roomType, roomInfo]: [string, any]) => (
          <div
            key={roomType}
            className="rounded-2xl border border-cyan-400/30 bg-cyan-900/10 shadow-lg p-6 flex flex-col gap-6 items-start"
          >
            <div className="flex items-center gap-3 mb-2">
              <Bed className="w-5 h-5 text-cyan-400" />
              <span className="text-lg font-bold text-white capitalize">
                {roomType.replace("_", " ")}
              </span>
              {/* Combined status and score */}
              <span
                className={`ml-4 flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold border-2 ${
                  getStatusDisplay(roomInfo.score, roomInfo.status)
                    .statusBgColorClass
                }`}
                style={{ minWidth: 120 }}
              >
                {getStatusDisplay(roomInfo.score, roomInfo.status).isReady ? (
                  <CheckCircle className="w-4 h-4 text-emerald-400 mr-1" />
                ) : (
                  <XCircle className="w-4 h-4 text-red-400 mr-1" />
                )}
                <span
                  className={
                    getStatusDisplay(roomInfo.score, roomInfo.status)
                      .statusColorClass
                  }
                >
                  {roomInfo.status}
                </span>
                <span className="mx-2 text-white/40">|</span>
                <span
                  className={
                    getStatusDisplay(roomInfo.score, roomInfo.status)
                      .scoreTextColorClass
                  }
                >
                  {roomInfo.score}/100
                </span>
              </span>
            </div>
            {/* Checklist */}
            {roomInfo.checklist && roomInfo.checklist.length > 0 && (
              <div className="rounded-xl bg-white/10 border border-blue-400/20 p-4 shadow flex flex-col mb-2 w-full">
                <span className="text-md font-bold text-blue-300 mb-2 flex items-center gap-2">
                  <ClipboardList className="w-4 h-4 text-blue-400" /> Room
                  Checklist
                </span>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {roomInfo.checklist.map((item: any, cidx: number) => {
                    const itemStatus = getChecklistItemStatus(item);
                    const IconComponent = itemStatus.icon;
                    return (
                      <div
                        key={cidx}
                        className={`flex items-center justify-between p-2 rounded-lg border ${itemStatus.bgClass}`}
                      >
                        <div className="flex items-center gap-2">
                          <IconComponent
                            className={`w-4 h-4 ${itemStatus.colorClass}`}
                          />
                          <span className="text-white/90 font-medium text-sm">
                            {item.item || item.parameter}
                          </span>
                        </div>
                        <span
                          className={`text-xs font-semibold ${itemStatus.colorClass}`}
                        >
                          {itemStatus.statusText}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
            {/* Issues and Fixes */}
            {roomInfo.issues && roomInfo.issues.length > 0 && (
              <div className="rounded-xl bg-white/10 border border-amber-400/20 p-4 shadow flex flex-col w-full">
                <span className="text-md font-bold text-amber-300 mb-4 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-amber-400" /> Issues
                  and Fixes
                </span>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {roomInfo.issues.map((issue: any, i_idx: number) => (
                    <div
                      key={i_idx}
                      className="rounded-xl bg-white/5 border border-amber-400/10 p-4 shadow flex flex-col gap-1 mx-1 min-h-[140px]"
                    >
                      <div>
                        <span className="text-xs font-bold text-red-400 uppercase tracking-wide">
                          Problem
                        </span>
                        <div className="text-white/90 font-medium text-sm mt-1 mb-2">
                          {issue.issue}
                        </div>
                      </div>
                      <div className="border-t border-white/10 pt-2 mt-2">
                        <span className="text-xs font-bold text-cyan-400 uppercase tracking-wide">
                          Solution
                        </span>
                        <div className="text-white/90 font-medium text-sm mt-1">
                          {issue.fix}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default RoomReadinessResult;
