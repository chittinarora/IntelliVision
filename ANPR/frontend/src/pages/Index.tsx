
import { useState } from "react";
import { Sidebar } from "@/components/anpr/Sidebar";
import { VideoUpload } from "@/components/anpr/VideoUpload";
import { LiveCamera } from "@/components/anpr/LiveCamera";
import { ParkingStats } from "@/components/anpr/ParkingStats";
import { DetectionHistory } from "@/components/anpr/DetectionHistory";
import { ExportLogs } from "@/components/anpr/ExportLogs";
import { BoundaryLines } from "@/components/anpr/BoundaryLines";

export type ActiveSection = 
  | "upload"
  | "camera" 
  | "stats"
  | "history"
  | "export"
  | "boundaries";

const Index = () => {
  const [activeSection, setActiveSection] = useState<ActiveSection>("upload");

  const renderMainContent = () => {
    switch (activeSection) {
      case "upload":
        return <VideoUpload />;
      case "camera":
        return <LiveCamera />;
      case "stats":
        return <ParkingStats />;
      case "history":
        return <DetectionHistory />;
      case "export":
        return <ExportLogs />;
      case "boundaries":
        return <BoundaryLines />;
      default:
        return <VideoUpload />;
    }
  };

  const getSectionTitle = () => {
    switch (activeSection) {
      case "upload": return "Video Upload & Analysis";
      case "camera": return "Live Camera Detection";
      case "stats": return "Parking Statistics";
      case "history": return "Detection History";
      case "export": return "Export & Reports";
      case "boundaries": return "Boundary Configuration";
      default: return "ANPR Dashboard";
    }
  };

  const getSectionDescription = () => {
    switch (activeSection) {
      case "upload": return "Upload and analyze video files for license plate detection";
      case "camera": return "Real-time license plate recognition from live camera feed";
      case "stats": return "Monitor parking occupancy and vehicle statistics";
      case "history": return "Review past detections and search records";
      case "export": return "Download reports and export detection data";
      case "boundaries": return "Configure detection zones and parking boundaries";
      default: return "Automatic Number Plate Recognition System";
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 flex">
      <Sidebar activeSection={activeSection} setActiveSection={setActiveSection} />
      <main className="flex-1 p-8">
        <div className="max-w-7xl mx-auto">
          <header className="mb-8 animate-fade-in">
            <div className="flex items-center gap-4 mb-4">
              <div className="w-12 h-12 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg">
                <div className="w-6 h-6 bg-white rounded-sm opacity-90"></div>
              </div>
              <div>
                <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-900 via-blue-700 to-indigo-600 bg-clip-text text-transparent">
                  {getSectionTitle()}
                </h1>
                <p className="text-slate-600 text-lg mt-1">
                  {getSectionDescription()}
                </p>
              </div>
            </div>
            <div className="h-1 bg-gradient-to-r from-blue-600 via-indigo-500 to-purple-600 rounded-full opacity-20"></div>
          </header>
          <div className="animate-fade-in">
            {renderMainContent()}
          </div>
        </div>
      </main>
    </div>
  );
};

export default Index;
