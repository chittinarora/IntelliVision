import React from "react";
import {
  Users,
  AlarmCheck,
  Car,
  Utensils,
  Camera,
  Construction,
  Bug,
  Bed,
} from "lucide-react";

const TASK_INFO = {
  people_count: {
    label: "People Counting",
    description:
      "Count people entering or exiting for safety compliance, occupancy, and crowd management.",
    icon: <Users className="w-8 h-8 text-white/90 mr-2" />,
  },
  emergency: {
    label: "Emergency Scenario",
    description:
      "Analyze emergency evacuation videos for real-time incident management and post-event review.",
    icon: <AlarmCheck className="w-8 h-8 text-white/90 mr-2" />,
  },
  parking_analysis: {
    label: "Parking Analysis",
    description:
      "Monitor parking spaces, detect occupancy, and analyze parking patterns in real-time.",
    icon: <Car className="w-8 h-8 text-white/90 mr-2" />,
  },
  car_count: {
    label: "Number Plate Detection",
    description:
      "Automatically detect and extract vehicle number plates from your surveillance footage.",
    icon: <Car className="w-8 h-8 text-white/90 mr-2" />,
  },
  food_waste: {
    label: "Food Waste Detection",
    description:
      "Upload food images to analyze visible items, estimate portions, calories, and get detailed nutritional insights.",
    icon: <Utensils className="w-8 h-8 text-white/90 mr-2" />,
  },
  food_waste_estimation: {
    label: "Food Waste Estimation",
    description:
      "Upload food images to analyze visible items, estimate portions, calories, and get detailed nutritional insights.",
    icon: <Utensils className="w-8 h-8 text-white/90 mr-2" />,
  },
  pothole_detection: {
    label: "Pothole Detection",
    description:
      "Detect and analyze road potholes in images or videos for infrastructure monitoring.",
    icon: <Construction className="w-8 h-8 text-white/90 mr-2" />,
  },
  pest_detection: {
    label: "Pest Detection",
    description:
      "Identify and analyze pest infestations in agricultural or residential settings.",
    icon: <Bug className="w-8 h-8 text-white/90 mr-2" />,
  },
  pest_monitoring: {
    label: "Pest Monitoring",
    description:
      "Detect and monitor snake presence in images or videos for safety and wildlife management.",
    icon: <Bug className="w-8 h-8 text-white/90 mr-2" />,
  },
  room_readiness: {
    label: "Room Readiness Analysis",
    description:
      "Upload hotel room photos for automated AI inspection. Analyze cleanliness, amenities, and overall guest readiness with detailed checklist results.",
    icon: <Bed className="w-8 h-8 text-white/90 mr-2" />,
  },
};

const UploadPageHeader: React.FC<{
  type: string | null;
  selectedTaskType?: string;
}> = ({ type, selectedTaskType }) => {
  const task = type && TASK_INFO[type as keyof typeof TASK_INFO];

  // Use task type info if available, otherwise use main task info
  const displayInfo = task;

  return (
    <div className="mb-10 text-center">
      <div className="flex items-center justify-center mb-3">
        {displayInfo?.icon}
        <h2 className="text-4xl font-extrabold text-white tracking-tight drop-shadow-2xl">
          {displayInfo
            ? `Upload Media for ${displayInfo.label}`
            : "Upload Media"}
        </h2>
      </div>
      <p className="text-white/80 text-lg max-w-2xl mx-auto font-semibold">
        {displayInfo
          ? displayInfo.description
          : "Select your media for analysis using AI-powered computer vision."}
      </p>
    </div>
  );
};

export default UploadPageHeader;
