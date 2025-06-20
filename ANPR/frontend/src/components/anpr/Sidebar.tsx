
import { Camera, Upload, BarChart3, History, Download, MapPin } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { ActiveSection } from "@/pages/Index";

interface SidebarProps {
  activeSection: ActiveSection;
  setActiveSection: (section: ActiveSection) => void;
}

const sidebarItems = [
  { id: "upload" as ActiveSection, label: "Upload Video", icon: Upload, description: "Analyze video files" },
  { id: "camera" as ActiveSection, label: "Live Camera", icon: Camera, description: "Real-time detection" },
  { id: "stats" as ActiveSection, label: "Parking Stats", icon: BarChart3, description: "Occupancy metrics" },
  { id: "history" as ActiveSection, label: "Detection History", icon: History, description: "Past records" },
  { id: "export" as ActiveSection, label: "Export Logs", icon: Download, description: "Data export" },
  { id: "boundaries" as ActiveSection, label: "Boundary Lines", icon: MapPin, description: "Zone config" },
];

export function Sidebar({ activeSection, setActiveSection }: SidebarProps) {
  return (
    <div className="w-72 bg-white/80 backdrop-blur-lg border-r border-blue-100 shadow-xl">
      <div className="p-6 border-b border-blue-100 bg-gradient-to-r from-blue-600 to-indigo-600">
        <div className="flex items-center gap-3 mb-3">
          <div className="w-10 h-10 bg-white rounded-lg flex items-center justify-center shadow-lg">
            <Camera className="h-6 w-6 text-blue-600" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white">ANPR System</h2>
            <p className="text-blue-100 text-sm">
              License Plate Recognition
            </p>
          </div>
        </div>
      </div>

      <nav className="p-4 space-y-2">
        {sidebarItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeSection === item.id;
          return (
            <Button
              key={item.id}
              variant="ghost"
              className={cn(
                "w-full justify-start gap-3 h-auto p-4 rounded-xl transition-all duration-200",
                "hover:bg-blue-50 hover:shadow-md hover:scale-[1.02]",
                isActive && "bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg scale-[1.02]",
                !isActive && "text-slate-700 hover:text-blue-700"
              )}
              onClick={() => setActiveSection(item.id)}
            >
              <div className={cn(
                "w-10 h-10 rounded-lg flex items-center justify-center transition-colors",
                isActive ? "bg-white/20" : "bg-blue-100"
              )}>
                <Icon className={cn(
                  "h-5 w-5",
                  isActive ? "text-white" : "text-blue-600"
                )} />
              </div>
              <div className="text-left flex-1">
                <div className="font-medium">{item.label}</div>
                <div className={cn(
                  "text-xs opacity-80",
                  isActive ? "text-blue-100" : "text-slate-500"
                )}>
                  {item.description}
                </div>
              </div>
            </Button>
          );
        })}
      </nav>

      <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-blue-50 to-transparent">
        <div className="text-center text-xs text-slate-500">
          © 2024 ANPR System v2.1
        </div>
      </div>
    </div>
  );
}
