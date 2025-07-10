import React, { useMemo, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { ChevronRight } from "lucide-react";
import { Job } from "@/types/job";
import {
  Users,
  Car,
  Activity,
  Construction,
  Trash2,
  Bug,
  Bed,
  Shield,
  Hotel,
  Star,
  CheckCircle,
  BarChart3,
  AlertTriangle,
  XCircle,
  CalendarCheck,
  ListChecks,
  Award,
} from "lucide-react";

interface StatsOverviewProps {
  jobs: Job[];
}

const StatsOverview = ({ jobs }: StatsOverviewProps) => {
  const [generalStatsOpen, setGeneralStatsOpen] = useState(true);
  const [hotelStatsOpen, setHotelStatsOpen] = useState(true);
  const [miscStatsOpen, setMiscStatsOpen] = useState(true);

  const stats = useMemo(() => {
    const totalAnalyses = jobs.length;

    // People counting - use new results structure first, fallback to legacy
    const completedPeopleCountJobs = jobs.filter(
      (job) =>
        job.job_type === "people_count" &&
        job.status === "done" &&
        ((!Array.isArray(job.results) &&
          job.results?.data?.person_count !== undefined) ||
          job.person_count !== undefined)
    );

    const totalPeopleCounted = completedPeopleCountJobs.reduce(
      (sum, job) =>
        sum +
        ((!Array.isArray(job.results) && job.results?.data?.person_count) ??
          job.person_count ??
          0),
      0
    );

    // Car counting
    const completedCarCountJobs = jobs.filter(
      (job) =>
        job.job_type === "car_count" &&
        job.status === "done" &&
        !Array.isArray(job.results) &&
        job.results?.data?.car_count !== undefined
    );

    const totalCarsCounted = completedCarCountJobs.reduce(
      (sum, job) =>
        sum +
        ((!Array.isArray(job.results) && job.results?.data?.car_count) ?? 0),
      0
    );

    // In/Out counting - include emergency_count jobs
    const completedInOutJobs = jobs.filter(
      (job) =>
        (job.job_type === "in_out" || job.job_type === "emergency_count") &&
        job.status === "done" &&
        !Array.isArray(job.results) &&
        (job.results?.data?.in !== undefined ||
          job.results?.data?.out !== undefined)
    );

    const totalInCount = completedInOutJobs.reduce(
      (sum, job) =>
        sum + ((!Array.isArray(job.results) && job.results?.data?.in) ?? 0),
      0
    );

    const totalOutCount = completedInOutJobs.reduce(
      (sum, job) =>
        sum + ((!Array.isArray(job.results) && job.results?.data?.out) ?? 0),
      0
    );

    // Pothole detection - use total_potholes from API response
    const completedPotholeJobs = jobs.filter(
      (job) =>
        job.job_type === "pothole_detection" &&
        job.status === "done" &&
        !Array.isArray(job.results) &&
        job.results?.data?.total_potholes !== undefined
    );

    const totalPotholes = completedPotholeJobs.reduce(
      (sum, job) =>
        sum +
        ((!Array.isArray(job.results) && job.results?.data?.total_potholes) ??
          0),
      0
    );

    // Food waste estimation - calculate total calories
    const completedFoodWasteJobs = jobs.filter(
      (job) =>
        job.job_type === "food_waste_estimation" &&
        job.status === "done" &&
        job.results
    );

    const totalFoodWasteCalories = completedFoodWasteJobs.reduce((sum, job) => {
      if (Array.isArray(job.results)) {
        return (
          sum +
          job.results.reduce(
            (jobSum, result) =>
              jobSum +
              (result.data?.total_calories || result.total_calories || 0),
            0
          )
        );
      } else if (job.results && !Array.isArray(job.results)) {
        return (
          sum +
          (job.results.data?.total_calories || job.results.total_calories || 0)
        );
      }
      return sum;
    }, 0);

    // Pest monitoring - use detected_snakes from API response
    const completedPestJobs = jobs.filter(
      (job) =>
        job.job_type === "pest_monitoring" &&
        job.status === "done" &&
        !Array.isArray(job.results) &&
        job.results?.data?.detected_snakes !== undefined
    );

    const totalPests = completedPestJobs.reduce(
      (sum, job) =>
        sum +
        ((!Array.isArray(job.results) && job.results?.data?.detected_snakes) ??
          0),
      0
    );

    // Hotel job stats - Room Readiness
    const completedRoomReadinessJobs = jobs.filter(
      (job) =>
        job.job_type === "room_readiness" &&
        job.status === "done" &&
        job.results &&
        !Array.isArray(job.results)
    );

    const averageRoomScore =
      completedRoomReadinessJobs.length > 0
        ? Math.round(
            completedRoomReadinessJobs.reduce((sum, job) => {
              const score = !Array.isArray(job.results)
                ? job.results.data?.room_score ||
                  job.results.data?.readiness_score ||
                  job.results?.room_score ||
                  job.results?.readiness_score ||
                  0
                : 0;
              return sum + score;
            }, 0) / completedRoomReadinessJobs.length
          )
        : 0;

    const readyRooms = completedRoomReadinessJobs.filter((job) => {
      if (!Array.isArray(job.results)) {
        const d = job.results.data || job.results;
        return (
          d.readiness_status === "ready" ||
          d.readiness_status === "Guest Ready" ||
          d.readiness_score === 100 ||
          d.room_score === 100
        );
      }
      return false;
    }).length;

    // Hotel job stats - Lobby Crowd Detection (FIXED)
    const completedLobbyJobs = jobs.filter(
      (job) =>
        job.job_type === "lobby_crowd_detection" &&
        job.status === "done" &&
        job.results &&
        !Array.isArray(job.results)
    );

    const totalLobbyAlerts = completedLobbyJobs.filter((job) => {
      if (!Array.isArray(job.results) && job.results) {
        const d = job.results.data || job.results;
        return (
          d.alert_triggered === true ||
          ("crowd_alert" in d && d.crowd_alert === true) ||
          ("occupancy_alert" in d && d.occupancy_alert === true)
        );
      }
      return false;
    }).length;

    // Hotel job stats - Wildlife Detection (FIXED)
    const completedWildlifeJobs = jobs.filter(
      (job) =>
        job.job_type === "wildlife_detection" &&
        job.status === "done" &&
        job.results &&
        !Array.isArray(job.results)
    );

    const wildlifeDetections = completedWildlifeJobs.filter((job) => {
      if (!Array.isArray(job.results) && job.results) {
        const d = job.results.data || job.results;
        return (
          d.wildlife_detected === true ||
          ("detection_count" in d &&
            typeof d.detection_count === "number" &&
            d.detection_count > 0)
        );
      }
      return false;
    }).length;

    const totalWildlifeCount = completedWildlifeJobs.reduce((sum, job) => {
      if (!Array.isArray(job.results) && job.results) {
        const d = job.results.data || job.results;
        return (
          sum +
          (d.wildlife_count ||
            ("detection_count" in d && typeof d.detection_count === "number"
              ? d.detection_count
              : 0) ||
            0)
        );
      }
      return sum;
    }, 0);

    const jobsThisMonth = jobs.filter((job) => {
      const jobDate = new Date(job.created_at);
      const currentDate = new Date();
      return (
        jobDate.getFullYear() === currentDate.getFullYear() &&
        jobDate.getMonth() === currentDate.getMonth()
      );
    }).length;

    const pendingJobs = jobs.filter(
      (job) => job.status === "pending" || job.status === "processing"
    ).length;

    const completedJobs = jobs.filter((job) => job.status === "done").length;
    const failedJobs = jobs.filter((job) => job.status === "failed").length;

    return {
      totalAnalyses,
      totalPeopleCounted,
      totalCarsCounted,
      totalInCount,
      totalOutCount,
      totalPotholes,
      totalFoodWasteCalories,
      totalPests,
      averageRoomScore,
      readyRooms,
      totalLobbyAlerts,
      wildlifeDetections,
      totalWildlifeCount,
      jobsThisMonth,
      pendingJobs,
      completedJobs,
      failedJobs,
    };
  }, [jobs]);

  // Helper to get the main data object (handles array or object)
  const getData = (job: any) => {
    if (!job.results) return null;
    if (Array.isArray(job.results)) {
      return job.results[0]?.data || null;
    }
    return job.results.data || null;
  };

  // StatCard component for compact, consistent stat display
  const StatCard = ({
    icon,
    value,
    label,
    color,
    title,
  }: {
    icon: React.ReactNode;
    value: React.ReactNode;
    label: string;
    color: string;
    title?: string;
  }) => (
    <div
      className={`glass-card flex items-center p-3 gap-3 border-l-4 ${color} bg-gradient-to-r from-[${color.replace(
        "border-",
        ""
      )}/10] to-transparent`}
      title={title || label}
    >
      {icon}
      <div className="ml-3 flex flex-col">
        <span
          className={`text-xl font-extrabold ${color.replace(
            "border-",
            "text-"
          )} leading-tight`}
        >
          {value}
        </span>
        <span className="text-xs text-white/60">{label}</span>
      </div>
    </div>
  );

  // Configuration for all stats to display
  const statsConfig = [
    // Top summary
    {
      key: "totalAnalyses",
      icon: <BarChart3 className="w-7 h-7 text-cyan-400 flex-shrink-0" />,
      color: "border-cyan-400",
      label: "Total Analyses",
      title: "Total number of analysis jobs you've run.",
    },
    {
      key: "completedJobs",
      icon: <CheckCircle className="w-7 h-7 text-green-400 flex-shrink-0" />,
      color: "border-green-400",
      label: "Completed",
      title: "Jobs that finished successfully.",
    },
    {
      key: "pendingJobs",
      icon: <AlertTriangle className="w-7 h-7 text-yellow-400 flex-shrink-0" />,
      color: "border-yellow-400",
      label: "Pending",
      title: "Jobs that are currently processing or waiting.",
    },
    {
      key: "failedJobs",
      icon: <XCircle className="w-7 h-7 text-red-400 flex-shrink-0" />,
      color: "border-red-400",
      label: "Failed",
      title: "Jobs that failed to complete.",
    },
    {
      key: "jobsThisMonth",
      icon: <CalendarCheck className="w-7 h-7 text-blue-400 flex-shrink-0" />,
      color: "border-blue-400",
      label: "Jobs This Month",
      title: "Jobs started this calendar month.",
    },
    // General Analytics
    {
      key: "totalPeopleCounted",
      icon: <Users className="w-7 h-7 text-blue-400 flex-shrink-0" />,
      color: "border-blue-400",
      label: "People Counted",
      title: "Sum of all people detected in completed jobs.",
    },
    {
      key: "totalCarsCounted",
      icon: <Car className="w-7 h-7 text-purple-400 flex-shrink-0" />,
      color: "border-purple-400",
      label: "Cars Detected",
      title: "Sum of all cars detected in completed jobs.",
    },
    {
      key: "totalInCount",
      icon: <Activity className="w-7 h-7 text-green-400 flex-shrink-0" />,
      color: "border-green-400",
      label: "In Count",
      title: "Total number of people counted as entering (IN) across all jobs.",
    },
    {
      key: "totalOutCount",
      icon: <Activity className="w-7 h-7 text-red-400 flex-shrink-0" />,
      color: "border-red-400",
      label: "Out Count",
      title: "Total number of people counted as exiting (OUT) across all jobs.",
    },
    {
      key: "totalPotholes",
      icon: <Construction className="w-7 h-7 text-orange-400 flex-shrink-0" />,
      color: "border-orange-400",
      label: "Potholes Found",
      title: "Total potholes detected in completed jobs.",
    },
    {
      key: "totalPests",
      icon: <Bug className="w-7 h-7 text-red-400 flex-shrink-0" />,
      color: "border-red-400",
      label: "Pests Found",
      title: "Total pests detected in completed jobs.",
    },
    {
      key: "totalFoodWasteCalories",
      icon: <Trash2 className="w-7 h-7 text-yellow-400 flex-shrink-0" />,
      color: "border-yellow-400",
      label: "Food Waste Calories",
      title: "Total estimated calories of food waste detected.",
      format: (v: number) => v.toLocaleString(),
    },
    // Hotel Analytics
    {
      key: "averageRoomScore",
      icon: <Bed className="w-7 h-7 text-cyan-400 flex-shrink-0" />,
      color: "border-cyan-400",
      label: "Avg Room Score",
      title: "Average room readiness score (0-100) for completed jobs.",
      format: (v: number) => `${v}/100`,
    },
    {
      key: "readyRooms",
      icon: <CheckCircle className="w-7 h-7 text-green-400 flex-shrink-0" />,
      color: "border-green-400",
      label: "Ready Rooms",
      title: "Number of rooms marked as ready in completed jobs.",
    },
    {
      key: "totalLobbyAlerts",
      icon: <Shield className="w-7 h-7 text-indigo-400 flex-shrink-0" />,
      color: "border-indigo-400",
      label: "Lobby Alerts",
      title: "Number of lobby crowd alerts triggered in completed jobs.",
    },
    {
      key: "wildlifeDetections",
      icon: <Bug className="w-7 h-7 text-emerald-400 flex-shrink-0" />,
      color: "border-emerald-400",
      label: "Wildlife Found",
      title: "Number of wildlife detections in completed jobs.",
    },
    {
      key: "totalWildlifeCount",
      icon: <Star className="w-7 h-7 text-yellow-400 flex-shrink-0" />,
      color: "border-yellow-400",
      label: "Total Wildlife Count",
      title: "Total number of wildlife detected in completed jobs.",
    },
  ];

  return (
    <Card className="glass-intense backdrop-blur-3xl border border-white/15 shadow-2xl mb-16 rounded-4xl overflow-hidden">
      <CardHeader className="bg-gradient-to-r from-white/5 to-white/2 border-b border-white/10 py-4">
        <CardTitle className="text-2xl font-bold text-white tracking-tight mb-1">
          Analytics Overview
        </CardTitle>
        <CardDescription className="text-white/60 text-base">
          Your key analysis statistics at a glance.
        </CardDescription>
      </CardHeader>
      <CardContent className="p-6">
        {/* Top Summary Row */}
        <h3 className="text-lg font-bold text-white mb-2 mt-2">Summary</h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 mb-6">
          {statsConfig.slice(0, 5).map((cfg) => (
            <StatCard
              key={cfg.key}
              icon={cfg.icon}
              value={cfg.format ? cfg.format(stats[cfg.key]) : stats[cfg.key]}
              label={cfg.label}
              color={cfg.color}
              title={cfg.title}
            />
          ))}
        </div>
        <h3 className="text-lg font-bold text-white mb-2 mt-2">
          General Analytics
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 mb-6">
          {statsConfig.slice(5, 12).map((cfg) => (
            <StatCard
              key={cfg.key}
              icon={cfg.icon}
              value={cfg.format ? cfg.format(stats[cfg.key]) : stats[cfg.key]}
              label={cfg.label}
              color={cfg.color}
              title={cfg.title}
            />
          ))}
        </div>
        <h3 className="text-lg font-bold text-white mb-2 mt-2">
          Hotel Analytics
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 mb-6">
          {statsConfig.slice(12).map((cfg) => (
            <StatCard
              key={cfg.key}
              icon={cfg.icon}
              value={cfg.format ? cfg.format(stats[cfg.key]) : stats[cfg.key]}
              label={cfg.label}
              color={cfg.color}
              title={cfg.title}
            />
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

export default StatsOverview;
