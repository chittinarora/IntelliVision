import React from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Car,
  Camera,
  Users,
  Activity,
  Construction,
  Trash2,
  Bug,
  Hotel,
  Bed,
  Shield,
} from "lucide-react";

interface TaskTypeSelectorProps {
  selectedType: string;
  onTypeChange: (type: string) => void;
}

const GENERAL_TASK_OPTIONS = [
  {
    value: "people_count",
    label: "People Counting",
    description: "Analyze footfall and monitor exits for safety",
    icon: <Users className="w-4 h-4" />,
  },
  {
    value: "car_count",
    label: "Number Plate Detection",
    description: "Detect and extract vehicle number plates",
    icon: <Car className="w-4 h-4" />,
  },
  {
    value: "parking_analysis",
    label: "Parking Analysis",
    description: "Detect and analyze car movement in parking zones",
    icon: <Camera className="w-4 h-4" />,
  },
  {
    value: "in_out",
    label: "In/Out Counting",
    description: "Track people entering and exiting areas",
    icon: <Activity className="w-4 h-4" />,
  },
  {
    value: "pothole_detection",
    label: "Pothole Detection",
    description: "Detect and locate potholes in road videos",
    icon: <Construction className="w-4 h-4" />,
  },
  {
    value: "food_waste_estimation",
    label: "Food Waste Estimation",
    description: "Estimate food waste from video analysis",
    icon: <Trash2 className="w-4 h-4" />,
  },
  {
    value: "pest_monitoring",
    label: "Pest Monitoring",
    description: "Monitor and identify pests in environment",
    icon: <Bug className="w-4 h-4" />,
  },
];

const HOTEL_TASK_OPTIONS = [
  {
    value: "room_readiness",
    label: "Room Readiness Analysis",
    description: "Analyze room cleanliness and readiness",
    icon: <Bed className="w-4 h-4" />,
  },
  {
    value: "lobby_crowd_detection",
    label: "Lobby Crowd Detection",
    description: "Monitor lobby occupancy and crowd alerts",
    icon: <Shield className="w-4 h-4" />,
  },
  {
    value: "wildlife_detection",
    label: "Wildlife Detection",
    description: "Detect wildlife presence in premises",
    icon: <Bug className="w-4 h-4" />,
  },
];

const ALL_TASK_OPTIONS = [...GENERAL_TASK_OPTIONS, ...HOTEL_TASK_OPTIONS];

const TaskTypeSelector: React.FC<TaskTypeSelectorProps> = ({
  selectedType,
  onTypeChange,
}) => {
  const selectedOption = ALL_TASK_OPTIONS.find(
    (option) => option.value === selectedType
  );

  const isHotelTask = HOTEL_TASK_OPTIONS.some(
    (option) => option.value === selectedType
  );

  return (
    <div className="mb-8">
      <div className="flex items-center gap-8">
        {/* Left column - Heading (30%) */}
        <div className="flex-shrink-0 w-1.5/5">
          <h3 className="text-xl font-bold text-white mb-2 tracking-tight">
            Select Analysis Type
          </h3>
        </div>

        {/* Right column - Selector (70%) */}
        <div className="flex-1">
          <Select value={selectedType} onValueChange={onTypeChange}>
            <SelectTrigger className="h-16 bg-white/10 backdrop-blur-xl border border-white/20 text-white shadow-lg transition-all duration-300 font-semibold rounded-3xl px-6 text-lg focus:ring-0 focus:ring-offset-0 focus:outline-none hover:bg-white/15 hover:shadow-xl">
              <SelectValue className="text-white font-semibold text-lg">
                {selectedOption ? (
                  <div className="flex items-center gap-4 justify-center">
                    <div className="text-white">{selectedOption.icon}</div>
                    {isHotelTask && (
                      <Hotel className="w-4 h-4 text-purple-300" />
                    )}
                    <span className="text-white text-lg">
                      {selectedOption.label}
                    </span>
                  </div>
                ) : (
                  "Select analysis type"
                )}
              </SelectValue>
            </SelectTrigger>
            <SelectContent className="bg-white/60 backdrop-blur-2xl border border-white/20 shadow-2xl rounded-2xl py-2 z-50 min-w-[400px] ring-1 ring-white/30">
              <div className="px-4 py-2 text-xs font-semibold text-slate-600 uppercase tracking-wider">
                General Tasks
              </div>
              {GENERAL_TASK_OPTIONS.map((option) => (
                <SelectItem
                  key={option.value}
                  value={option.value}
                  className="px-6 py-3 rounded-xl hover:bg-white/80 hover:backdrop-blur-xl font-semibold text-slate-900/80 transition-all duration-150 cursor-pointer focus:bg-white/70"
                >
                  <div className="flex items-center gap-4 w-full">
                    <div className="text-slate-700 p-2 rounded-xl transition-colors">
                      {option.icon}
                    </div>
                    <div className="flex-1">
                      <div className="font-bold text-slate-900 text-base">
                        {option.label}
                      </div>
                      <div className="text-sm text-slate-600 mt-1">
                        {option.description}
                      </div>
                    </div>
                  </div>
                </SelectItem>
              ))}

              <div className="px-4 py-2 text-xs font-semibold text-slate-600 uppercase tracking-wider border-t border-slate-300 mt-2">
                Hotel Tasks
              </div>
              {HOTEL_TASK_OPTIONS.map((option) => (
                <SelectItem
                  key={option.value}
                  value={option.value}
                  className="px-6 py-3 rounded-xl hover:bg-white/80 hover:backdrop-blur-xl font-semibold text-slate-900/80 transition-all duration-150 cursor-pointer focus:bg-white/70"
                >
                  <div className="flex items-center gap-4 w-full">
                    <div className="text-slate-700 p-2 rounded-xl transition-colors flex items-center gap-1">
                      {option.icon}
                      <Hotel className="w-3 h-3 text-purple-600" />
                    </div>
                    <div className="flex-1">
                      <div className="font-bold text-slate-900 text-base">
                        {option.label}
                      </div>
                      <div className="text-sm text-slate-600 mt-1">
                        {option.description}
                      </div>
                    </div>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
    </div>
  );
};

export default TaskTypeSelector;
