import React from "react";
import { Button } from "@/components/ui/button";
import { Download, Eye, Clock, Hash, Activity, Tag, Hotel } from "lucide-react";
import { Job } from "@/types/job";
import { ensureHttpsUrl } from "@/lib/utils";
import { toast } from "@/components/ui/use-toast";
import { Badge } from "@/components/ui/badge";
import { downloadVideo } from "@/lib/videoUtils";

interface JobCardProps {
  job: Job;
  onView: (job: Job) => void;
}

const getStatusInfo = (status: Job["status"]) => {
  switch (status) {
    case "done":
      return {
        color: "bg-green-500/20 text-green-400 border-green-500/30",
        icon: <Download className="w-4 h-4" />,
      };
    case "processing":
      return {
        color: "bg-blue-500/20 text-blue-400 border-blue-500/30",
        icon: <Activity className="w-4 h-4 animate-spin" />,
      };
    case "failed":
      return {
        color: "bg-red-500/20 text-red-400 border-red-500/30",
        icon: <Activity className="w-4 h-4" />,
      };
    default:
      return {
        color: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
        icon: <Activity className="w-4 h-4" />,
      };
  }
};

const isHotelJob = (jobType?: string) => {
  return [
    "room_readiness",
    "lobby_crowd_detection",
    "wildlife_detection",
  ].includes(jobType || "");
};

const formatJobType = (jobType?: string) => {
  if (!jobType) return "Unknown";

  switch (jobType) {
    case "people_count":
      return "People Counting";
    case "car_count":
      return "Number Plate Detection";
    case "parking_analysis":
      return "Parking Analysis";
    case "in_out":
      return "In/Out Counting";
    case "pothole_detection":
      return "Pothole Detection";
    case "food_waste_estimation":
      return "Food Waste Estimation";
    case "pest_monitoring":
      return "Pest Monitoring";
    case "room_readiness":
      return "Room Readiness";
    case "lobby_crowd_detection":
      return "Lobby Crowd Detection";
    case "wildlife_detection":
      return "Wildlife Detection";
    default:
      return jobType.replace("_", " ").replace(/\b\w/g, (l) => l.toUpperCase());
  }
};

const JobCard: React.FC<JobCardProps> = ({ job, onView }) => {
  const statusInfo = getStatusInfo(job.status);
  const isHotel = isHotelJob(job.job_type);

  return (
    <div className="grid grid-cols-12 items-center gap-6 px-6 py-4 border-b border-white/10 last:border-b-0 hover:bg-white/5 transition-colors duration-200">
      {/* Job ID - Column 1 */}
      <div className="col-span-1 flex items-center">
        <Hash className="w-4 h-4 mr-2 text-white/70 flex-shrink-0" />
        <span className="font-bold text-white">{job.id}</span>
      </div>

      {/* Job Type - Column 2-4 */}
      <div className="col-span-3 flex items-center">
        <Tag className="w-3 h-3 mr-2 text-white/70 flex-shrink-0" />
        <div
          className={`px-2 py-1 rounded-xl flex items-center gap-1 ${
            isHotel ? "bg-purple-500/20" : "bg-white/10"
          }`}
        >
          {isHotel && <Hotel className="w-3 h-3 text-purple-300" />}
          <span className="text-sm text-white/90">
            {formatJobType(job.job_type)}
          </span>
        </div>
      </div>

      {/* Submitted Date - Column 5-6 */}
      <div className="col-span-2 flex items-center">
        <Clock className="w-4 h-4 mr-2 text-white/70 flex-shrink-0" />
        <span className="text-sm text-white/80">
          {new Date(job.created_at).toLocaleDateString()}
        </span>
      </div>

      {/* Status - Column 7-9 */}
      <div className="col-span-3 flex justify-center">
        <Badge
          variant="outline"
          className={`capitalize font-semibold text-sm ${statusInfo.color}`}
        >
          {statusInfo.icon}
          <span className="ml-2">{job.status}</span>
        </Badge>
      </div>

      {/* Actions - Column 10-12 */}
      <div className="col-span-3 flex gap-2 justify-center">
        {job.status === "done" ? (
          <div className="flex gap-2">
            <Button
              onClick={() => onView(job)}
              variant="outline"
              size="sm"
              className="bg-cyan-500/60 backdrop-blur-xl border border-cyan-400/70 text-white hover:bg-cyan-500/70 hover:text-white transition-all duration-300 font-semibold rounded-2xl px-3 py-2 h-8 min-w-[80px] shadow-lg button-hover"
            >
              <Eye className="w-3 h-3 mr-1" />
              View
            </Button>
            {job.output_video && (
              <Button
                onClick={() => downloadVideo(job.output_video)}
                size="sm"
                className="bg-green-500/60 backdrop-blur-xl border border-green-400/70 text-white hover:bg-green-500/70 hover:text-white transition-all duration-300 font-semibold rounded-2xl px-3 py-2 h-8 min-w-[80px] shadow-lg button-hover"
              >
                <Download className="w-3 h-3 mr-1" />
                Download
              </Button>
            )}
          </div>
        ) : (
          <Button
            variant="outline"
            size="sm"
            disabled
            className="bg-white/10 backdrop-blur-xl border border-white/20 text-white/50 transition-all duration-300 font-semibold rounded-2xl px-3 py-2 h-8 min-w-[80px] shadow-lg cursor-not-allowed opacity-50"
          >
            <Clock className="w-3 h-3 mr-1" />
            Pending
          </Button>
        )}
      </div>
    </div>
  );
};

export default JobCard;
