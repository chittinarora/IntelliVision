import React, { useState } from "react";
import { Link } from "react-router-dom";
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
import {
  BarChart3,
  Users,
  Car,
  Activity,
  Construction,
  Trash2,
  Bug,
  Hotel,
  Bed,
  Shield,
  ChevronRight,
} from "lucide-react";

const GENERAL_TASKS = [
  {
    key: "people_count",
    label: "People Counting",
    description:
      "Analyze footfall and monitor exits for safety and compliance.",
    icon: <Users className="w-8 h-8" />,
    color: "bg-blue-500/20 hover:bg-blue-500/30 border-blue-400/30",
  },
  {
    key: "car_count",
    label: "Number Plate Detection",
    description: "Detect and extract number plates from vehicles.",
    icon: <Car className="w-8 h-8" />,
    color: "bg-purple-500/20 hover:bg-purple-500/30 border-purple-400/30",
  },
  {
    key: "parking_analysis",
    label: "Parking Analysis",
    description: "Analyze the movement of vehicles in parking zones.",
    icon: <Car className="w-8 h-8" />,
    color: "bg-purple-500/20 hover:bg-purple-500/30 border-purple-400/30",
  },
  {
    key: "in_out",
    label: "In/Out Counting",
    description: "Track people entering and exiting areas.",
    icon: <Activity className="w-8 h-8" />,
    color: "bg-green-500/20 hover:bg-green-500/30 border-green-400/30",
  },
  {
    key: "pothole_detection",
    label: "Pothole Detection",
    description: "Detect and locate potholes in road videos.",
    icon: <Construction className="w-8 h-8" />,
    color: "bg-orange-500/20 hover:bg-orange-500/30 border-orange-400/30",
  },
  {
    key: "food_waste_estimation",
    label: "Food Waste Estimation",
    description: "Estimate food waste from video analysis.",
    icon: <Trash2 className="w-8 h-8" />,
    color: "bg-yellow-500/20 hover:bg-yellow-500/30 border-yellow-400/30",
  },
  {
    key: "pest_monitoring",
    label: "Pest Monitoring",
    description: "Monitor and identify pests in your environment.",
    icon: <Bug className="w-8 h-8" />,
    color: "bg-red-500/20 hover:bg-red-500/30 border-red-400/30",
  },
];

const HOTEL_TASKS = [
  {
    key: "room_readiness",
    label: "Room Readiness Analysis",
    description: "Analyze room cleanliness and readiness for guests.",
    icon: <Bed className="w-8 h-8" />,
    color: "bg-cyan-500/20 hover:bg-cyan-500/30 border-cyan-400/30",
  },
  {
    key: "lobby_crowd_detection",
    label: "Lobby Crowd Detection",
    description: "Monitor lobby occupancy and detect crowd alerts.",
    icon: <Shield className="w-8 h-8" />,
    color: "bg-indigo-500/20 hover:bg-indigo-500/30 border-indigo-400/30",
  },
  {
    key: "wildlife_detection",
    label: "Wildlife Detection",
    description: "Detect wildlife presence in hotel premises.",
    icon: <Bug className="w-8 h-8" />,
    color: "bg-emerald-500/20 hover:bg-emerald-500/30 border-emerald-400/30",
  },
];

const TaskSelector = () => {
  const [generalJobsOpen, setGeneralJobsOpen] = useState(true);
  const [hotelJobsOpen, setHotelJobsOpen] = useState(true);

  return (
    <Card
      id="task-selector"
      className="glass-intense backdrop-blur-3xl border border-white/15 shadow-2xl mb-8 rounded-4xl overflow-hidden"
    >
      <CardHeader className="bg-gradient-to-r from-white/5 to-white/2 border-b border-white/10 py-4">
        <CardTitle className="text-2xl font-bold text-white tracking-tight mb-1">
          What do you want to analyze?
        </CardTitle>
        <CardDescription className="text-white/60 text-sm">
          Select a job type to begin a new analysis:
        </CardDescription>
      </CardHeader>
      <CardContent className="p-4">
        {/* General Jobs Section */}
        <Collapsible open={generalJobsOpen} onOpenChange={setGeneralJobsOpen}>
          <CollapsibleTrigger className="flex items-center justify-between w-full p-3 rounded-2xl hover:bg-white/5 transition-colors duration-200 group">
            <div className="flex items-center gap-3">
              <div className="bg-gradient-to-br from-blue-500/20 to-cyan-500/20 backdrop-blur-sm p-2 rounded-xl shadow-lg border border-white/10">
                <BarChart3 className="w-5 h-5 text-cyan-300" />
              </div>
              <h3 className="text-lg font-bold text-white">General Jobs</h3>
            </div>
            <ChevronRight
              className={`w-5 h-5 text-white/70 transition-transform duration-200 ${
                generalJobsOpen ? "rotate-90" : ""
              }`}
            />
          </CollapsibleTrigger>
          <CollapsibleContent className="pt-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {GENERAL_TASKS.map((task) => (
                <Link
                  key={task.key}
                  to={`/upload?type=${task.key}`}
                  className={`${task.color} backdrop-blur-2xl border-2 text-white font-bold rounded-3xl px-4 py-5 shadow-xl flex flex-col items-center gap-3 text-base min-h-[160px] transition-all duration-300 hover:scale-105 hover:shadow-2xl group`}
                  style={{ textDecoration: "none" }}
                >
                  <div className="transition-transform duration-300 group-hover:scale-110">
                    {task.icon}
                  </div>
                  <span className="text-lg font-extrabold text-center leading-tight">
                    {task.label}
                  </span>
                  <span className="text-white/80 text-sm font-normal text-center leading-relaxed px-2">
                    {task.description}
                  </span>
                </Link>
              ))}
            </div>
          </CollapsibleContent>
        </Collapsible>

        {/* Hotel Jobs Section */}
        <Collapsible
          open={hotelJobsOpen}
          onOpenChange={setHotelJobsOpen}
          className="mt-6"
        >
          <CollapsibleTrigger className="flex items-center justify-between w-full p-3 rounded-2xl hover:bg-white/5 transition-colors duration-200 group">
            <div className="flex items-center gap-3">
              <div className="bg-gradient-to-br from-purple-500/20 to-pink-500/20 backdrop-blur-sm p-2 rounded-xl shadow-lg border border-white/10">
                <Hotel className="w-5 h-5 text-pink-300" />
              </div>
              <h3 className="text-lg font-bold text-white">Hotel Jobs</h3>
            </div>
            <ChevronRight
              className={`w-5 h-5 text-white/70 transition-transform duration-200 ${
                hotelJobsOpen ? "rotate-90" : ""
              }`}
            />
          </CollapsibleTrigger>
          <CollapsibleContent className="pt-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {HOTEL_TASKS.map((task) => (
                <Link
                  key={task.key}
                  to={`/upload?type=${task.key}`}
                  className={`${task.color} backdrop-blur-2xl border-2 text-white font-bold rounded-3xl px-4 py-5 shadow-xl flex flex-col items-center gap-3 text-base min-h-[160px] transition-all duration-300 hover:scale-105 hover:shadow-2xl group relative`}
                  style={{ textDecoration: "none" }}
                >
                  <div className="absolute top-3 right-3">
                    <Hotel className="w-4 h-4 text-white/60" />
                  </div>
                  <div className="transition-transform duration-300 group-hover:scale-110">
                    {task.icon}
                  </div>
                  <span className="text-lg font-extrabold text-center leading-tight">
                    {task.label}
                  </span>
                  <span className="text-white/80 text-sm font-normal text-center leading-relaxed px-2">
                    {task.description}
                  </span>
                </Link>
              ))}
            </div>
          </CollapsibleContent>
        </Collapsible>
      </CardContent>
    </Card>
  );
};

export default TaskSelector;
