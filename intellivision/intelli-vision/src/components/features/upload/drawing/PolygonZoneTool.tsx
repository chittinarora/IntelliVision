import React, { useState, useRef, useEffect } from "react";
import {
  Trash2,
  Plus,
  CheckCircle,
  AlertCircle,
  RotateCcw,
  Info,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import SharedDrawingLayout from "./SharedDrawingLayout";

// PolygonZoneTool.tsx
// This component allows users to draw polygonal detection zones on a video frame for crowd density or area-based analytics.
// It handles video frame extraction, polygon drawing, zone management, and communicates zone data to the parent.

// Point: Represents a point in normalized coordinates (0-1)
interface Point {
  x: number;
  y: number;
}

// Zone: Represents a detection zone with a name, points, threshold, and completion state
interface Zone {
  id: string;
  name: string;
  points: Point[];
  threshold: number;
  isComplete: boolean;
}

// Props for the PolygonZoneTool
interface PolygonZoneToolProps {
  videoFile: File; // The video file to extract a frame from
  onZonesChange: (zones: Zone[]) => void; // Callback when zones are updated
}

// SNAP_STEP: Used to snap coordinates to a grid for easier point placement
const SNAP_STEP = 10;
// snap: Helper function to snap a value to the nearest grid step
const snap = (val: number): number =>
  (Math.round((val * 1000) / SNAP_STEP) * SNAP_STEP) / 1000;

const PolygonZoneTool: React.FC<PolygonZoneToolProps> = ({
  videoFile,
  onZonesChange,
}) => {
  // State for the extracted image frame
  const [imageSrc, setImageSrc] = useState<string>("");
  // State for all zones
  const [zones, setZones] = useState<Zone[]>([]);
  // State for the currently active zone (being drawn)
  const [activeZoneId, setActiveZoneId] = useState<string | null>(null);
  // State for drawing mode
  const [isDrawing, setIsDrawing] = useState(false);
  // State for dragging a point handle
  const [dragTarget, setDragTarget] = useState<{
    zoneId: string;
    pointIdx: number;
  } | null>(null);
  // State for hovered handle (for UI feedback)
  const [hoveredHandle, setHoveredHandle] = useState<{
    zoneId: string;
    pointIdx: number;
  } | null>(null);

  // Refs for the image and container
  const imageRef = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Extract a frame from the video file when it changes
  useEffect(() => {
    if (!videoFile) return;

    const video = document.createElement("video");
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    video.src = URL.createObjectURL(videoFile);
    video.currentTime = 1;

    video.addEventListener("loadeddata", () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      if (ctx) {
        ctx.drawImage(video, 0, 0);
        setImageSrc(canvas.toDataURL("image/jpeg", 0.8));
      }
      URL.revokeObjectURL(video.src);
    });
  }, [videoFile]);

  // Convert mouse coordinates to normalized image coordinates (0-1 relative to image)
  const getCoordinates = (clientX: number, clientY: number) => {
    if (!imageRef.current) return { x: 0, y: 0 };
    const imageRect = imageRef.current.getBoundingClientRect();
    let x = (clientX - imageRect.left) / imageRect.width;
    let y = (clientY - imageRect.top) / imageRect.height;
    // Snap to grid for easier placement
    x = snap(x);
    y = snap(y);
    // Clamp between 0 and 1
    return {
      x: Math.max(0, Math.min(1, x)),
      y: Math.max(0, Math.min(1, y)),
    };
  };

  // Create a new zone and enter drawing mode
  const createNewZone = () => {
    const newZone: Zone = {
      id: `zone-${Date.now()}`,
      name: `Zone ${zones.length + 1}`,
      points: [],
      threshold: 5,
      isComplete: false,
    };
    setZones((prev) => [...prev, newZone]);
    setActiveZoneId(newZone.id);
    setIsDrawing(true);
  };

  // Handle click (mousedown) on image to add a point to the active zone
  const handleImageClick = (e: React.MouseEvent) => {
    // Only add a point on left mouse button, and if not dragging
    if (e.button !== 0) return;
    if (dragTarget) return;
    if (!isDrawing || !activeZoneId) return;

    const coords = getCoordinates(e.clientX, e.clientY);

    setZones((prev) =>
      prev.map((zone) => {
        if (zone.id === activeZoneId) {
          const newPoints = [...zone.points, coords];
          return { ...zone, points: newPoints };
        }
        return zone;
      })
    );
  };

  // Complete the current zone if it has enough points
  const completeZone = () => {
    if (!activeZoneId) return;

    setZones((prev) =>
      prev.map((zone) => {
        if (zone.id === activeZoneId && zone.points.length >= 3) {
          return { ...zone, isComplete: true };
        }
        return zone;
      })
    );

    setIsDrawing(false);
    setActiveZoneId(null);
  };

  // Delete a zone by id
  const deleteZone = (zoneId: string) => {
    setZones((prev) => prev.filter((zone) => zone.id !== zoneId));
    if (activeZoneId === zoneId) {
      setActiveZoneId(null);
      setIsDrawing(false);
    }
  };

  // Undo the last point added to the active zone
  const handleUndo = () => {
    if (!activeZoneId) return;

    setZones((prev) =>
      prev.map((zone) => {
        if (zone.id === activeZoneId && zone.points.length > 0) {
          const newPoints = zone.points.slice(0, -1);
          return { ...zone, points: newPoints };
        }
        return zone;
      })
    );
  };

  // Reset the active zone's points
  const resetActiveZone = () => {
    if (!activeZoneId) return;

    setZones((prev) =>
      prev.map((zone) => {
        if (zone.id === activeZoneId) {
          return { ...zone, points: [] };
        }
        return zone;
      })
    );
  };

  // Update the name of a zone
  const updateZoneName = (zoneId: string, name: string) => {
    setZones((prev) =>
      prev.map((zone) => (zone.id === zoneId ? { ...zone, name } : zone))
    );
  };

  // Update the alert threshold of a zone
  const updateZoneThreshold = (zoneId: string, threshold: number) => {
    setZones((prev) =>
      prev.map((zone) => (zone.id === zoneId ? { ...zone, threshold } : zone))
    );
  };

  // Handle mouse down on a point handle to start dragging
  const handleHandleMouseDown =
    (zoneId: string, pointIdx: number) => (e: React.MouseEvent) => {
      e.stopPropagation(); // Prevent adding a new point
      e.preventDefault(); // Prevent parent onClick from firing
      setDragTarget({ zoneId, pointIdx });
    };

  // Handle mouse move event to update dragging
  const handleMouseMove = (e: React.MouseEvent) => {
    if (!dragTarget) return;
    const coords = getCoordinates(e.clientX, e.clientY);
    setZones((prev) =>
      prev.map((zone) => {
        if (zone.id === dragTarget.zoneId) {
          const newPoints = [...zone.points];
          newPoints[dragTarget.pointIdx] = coords;
          return { ...zone, points: newPoints };
        }
        return zone;
      })
    );
  };

  // Handle mouse up event to stop dragging
  const handleMouseUp = () => {
    if (dragTarget) {
      setDragTarget(null);
    }
  };

  // Notify parent when zones change
  useEffect(() => {
    onZonesChange(zones);
  }, [zones, onZonesChange]);

  // Get a color for each zone (cycled from a palette)
  const getZoneColor = (index: number) => {
    const colors = [
      "#ef4444",
      "#3b82f6",
      "#10b981",
      "#f59e0b",
      "#8b5cf6",
      "#ec4899",
    ];
    return colors[index % colors.length];
  };

  // Derived state for active, completed, and incomplete zones
  const activeZone = zones.find((z) => z.id === activeZoneId);
  const completedZones = zones.filter((z) => z.isComplete);
  const incompleteZones = zones.filter((z) => !z.isComplete);

  // Show loading UI if image frame is not ready
  if (!imageSrc) {
    return (
      <div className="flex items-center justify-center h-64 glass-card rounded-3xl">
        <div className="text-white/60 animate-pulse">
          Extracting video frame...
        </div>
      </div>
    );
  }

  // Status card UI (shows number of complete/in-progress zones)
  const statusCard = (
    <div
      className={`p-6 rounded-2xl backdrop-blur-sm border transition-all flex items-center gap-6 justify-center ${
        completedZones.length > 0
          ? "bg-emerald-500/10 border-emerald-400/30"
          : "bg-amber-500/10 border-amber-400/30"
      }`}
    >
      <div className="flex items-center gap-3">
        <CheckCircle className="w-6 h-6 text-emerald-400" />
        <span className="text-emerald-300 font-semibold text-md">
          {completedZones.length} Complete
        </span>
      </div>
      {incompleteZones.length > 0 && (
        <div className="flex items-center gap-3">
          <AlertCircle className="w-6 h-6 text-amber-400" />
          <span className="text-amber-300 font-semibold text-md">
            {incompleteZones.length} In Progress
          </span>
        </div>
      )}
    </div>
  );

  // Controls UI (add zone, drawing info, undo, reset, complete, cancel)
  const controls = (
    <div className="space-y-3">
      <div className="flex gap-2">
        <Button
          onClick={createNewZone}
          disabled={isDrawing}
          className="flex-1 glass-button bg-cyan-500/80 hover:bg-cyan-500/90 text-white shadow-lg h-8 text-sm font-bold rounded-lg border-0 transition-all disabled:opacity-50 disabled:cursor-not-allowed backdrop-blur-sm px-2 rounded-xl"
          size="sm"
        >
          <Plus className="w-4 h-4 mr-1" />
          Add Zone
        </Button>
      </div>
      {/* Drawing Mode Instructions */}
      {isDrawing && activeZone && (
        <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-xl backdrop-blur-sm space-y-2 mt-2">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-amber-400 rounded-full animate-pulse"></div>
            <span className="text-amber-300 font-semibold text-sm">
              Drawing Mode Active
            </span>
          </div>
          <p className="text-amber-200/90 text-xs leading-relaxed">
            Click on the video frame to add points. You need at least 3 points
            to create a valid detection zone.
          </p>
          <div className="flex items-center gap-2 text-xs text-amber-200/80">
            <span className="font-medium">
              Points: {activeZone.points.length}/3+
            </span>
          </div>
          <div className="flex gap-2">
            <Button
              onClick={handleUndo}
              disabled={activeZone.points.length === 0}
              variant="ghost"
              size="sm"
              className="glass-button text-white hover:bg-white/15 rounded-lg font-medium disabled:opacity-50 h-8 px-2"
            >
              Undo
            </Button>
            <Button
              onClick={resetActiveZone}
              variant="ghost"
              size="sm"
              className="glass-button text-white hover:bg-white/15 rounded-lg font-medium h-8 px-2"
            >
              <RotateCcw className="w-4 h-4 mr-1" />
              Reset
            </Button>
          </div>
          <div className="flex gap-2 pt-2">
            <Button
              onClick={completeZone}
              disabled={activeZone.points.length < 3}
              className="flex-1 bg-green-600/80 hover:bg-green-600/90 text-white rounded-lg border-0 transition-all disabled:opacity-50 font-medium h-8 text-xs"
              size="sm"
            >
              <CheckCircle className="w-4 h-4 mr-1" />
              Complete
            </Button>
            <Button
              onClick={() => deleteZone(activeZoneId!)}
              variant="ghost"
              className="bg-red-500/20 hover:bg-red-500/30 text-red-300 border border-red-500/30 rounded-lg font-medium h-8 text-xs px-2"
              size="sm"
            >
              Cancel
            </Button>
          </div>
        </div>
      )}
    </div>
  );

  // List section UI (shows all zones and their properties)
  const listSection = (
    <div className="w-full">
      <div className="glass-card rounded-2xl border-white/10 p-6 shadow-lg">
        <h4 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
          Detection Zones
        </h4>
        {zones.length === 0 ? (
          <div className="p-8 bg-white/5 rounded-xl border border-white/10 text-center">
            <p className="text-white/60 font-medium">No zones configured yet</p>
            <p className="text-white/40 text-sm mt-2">
              Click "Add Zone" to start
            </p>
          </div>
        ) : (
          // Scrollable grid: 3 columns on lg+, 2 on md, 1 on mobile
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 max-h-80 overflow-y-auto pr-2">
            {zones.map((zone, index) => (
              <div
                key={zone.id}
                className={`relative p-5 rounded-2xl border flex flex-col gap-2 transition-all shadow-md group
                  ${
                    zone.id === activeZoneId
                      ? "bg-gradient-to-br from-cyan-900/30 to-cyan-700/5 border-cyan-400/40 shadow-cyan-400/10"
                      : "bg-gradient-to-br from-white/10 to-white/5 border-white/10 hover:shadow-lg hover:shadow-cyan-400/10"
                  }
                  hover:scale-[1.025] duration-150 ease-in-out
                `}
                style={{
                  borderLeftWidth: 3,
                  borderLeftColor: getZoneColor(index) + "33", // subtle, semi-transparent
                  boxShadow: `0 2px 12px 0 ${getZoneColor(index)}22`, // faint colored glow
                }}
              >
                {/* Header: Color, Status, Delete */}
                <div className="flex items-center gap-2 min-w-[90px] mb-2 pb-2 border-b border-white/10">
                  <div
                    className="w-3 h-3 rounded-full border border-white/20"
                    style={{
                      backgroundColor: getZoneColor(index),
                      opacity: 0.7,
                    }}
                  />
                  {zone.isComplete ? (
                    <CheckCircle className="w-4 h-4 text-green-400" />
                  ) : (
                    <div className="w-4 h-4 rounded-full border border-amber-400 flex items-center justify-center">
                      <span className="text-amber-400 text-xs font-bold">
                        {zone.points.length}
                      </span>
                    </div>
                  )}
                  <span className="text-white font-semibold text-xs md:text-sm">
                    {zone.isComplete
                      ? "Complete"
                      : `${zone.points.length}/3+ pts`}
                  </span>
                  <div className="flex-1" />
                  <Button
                    onClick={() => deleteZone(zone.id)}
                    variant="ghost"
                    size="sm"
                    className="text-red-400 hover:text-red-300 hover:bg-red-500/20 rounded-xl p-2 opacity-80 group-hover:opacity-100 transition-opacity"
                    title="Delete zone"
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
                {/* Zone Name on its own row */}
                <div className="flex flex-col mb-1">
                  <Label className="text-white/70 text-xs mb-1 block font-medium">
                    Name
                  </Label>
                  <Input
                    value={zone.name}
                    onChange={(e) => updateZoneName(zone.id, e.target.value)}
                    className="bg-white/10 border-white/20 text-white rounded-xl focus:border-cyan-400/50 focus:ring-1 focus:ring-cyan-400/50 font-medium backdrop-blur-sm h-8 text-xs md:w-48"
                    placeholder="Zone name"
                  />
                </div>
                {/* Threshold on its own row */}
                <div className="flex flex-col">
                  <Label className="text-white/70 text-xs mb-1 block font-medium">
                    Threshold
                  </Label>
                  <div className="flex items-center gap-1">
                    <Input
                      type="number"
                      min="1"
                      value={zone.threshold}
                      onChange={(e) =>
                        updateZoneThreshold(
                          zone.id,
                          parseInt(e.target.value) || 1
                        )
                      }
                      className="bg-white/10 border-white/20 text-white rounded-xl focus:border-cyan-400/50 focus:ring-1 focus:ring-cyan-400/50 font-medium backdrop-blur-sm h-8 text-xs w-20"
                      placeholder="5"
                    />
                    <span className="text-white/60 text-xs font-medium whitespace-nowrap">
                      people
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );

  // Preview UI (image, SVG overlay, instructions)
  const preview = (
    <div className="space-y-6 h-full">
      {/* Drawing Canvas without card wrapper */}
      <div
        ref={containerRef}
        className="relative w-full overflow-hidden rounded-2xl bg-black/20 backdrop-blur-sm border border-white/10"
        style={{
          userSelect: "none",
          touchAction: "none",
          cursor: dragTarget ? "grabbing" : isDrawing ? "crosshair" : "default",
        }}
        onMouseDown={handleImageClick}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => {
          setDragTarget(null);
          setHoveredHandle(null);
        }}
      >
        <img
          ref={imageRef}
          src={imageSrc}
          alt="Video frame for zone configuration"
          className="w-full h-auto block select-none rounded-2xl"
          style={{
            display: "block",
            width: "100%",
            height: "auto",
            maxHeight: 500,
            objectFit: "contain",
          }}
          draggable={false}
        />
        {/* SVG Overlay for zones */}
        <svg
          className="absolute inset-0 w-full h-full"
          style={{ pointerEvents: "auto", borderRadius: "1rem" }}
        >
          {zones.map((zone, index) => {
            const color = getZoneColor(index);
            const points = zone.points;
            if (points.length === 0) return null;
            return (
              <g key={zone.id}>
                {zone.isComplete && points.length >= 3 && (
                  <polygon
                    points={points
                      .map((p) => `${p.x * 100}%,${p.y * 100}%`)
                      .join(" ")}
                    fill={color}
                    fillOpacity="0.2"
                    stroke={color}
                    strokeWidth="3"
                    className="transition-all duration-300"
                    style={{ filter: "drop-shadow(0 0 8px rgba(0,0,0,0.4))" }}
                  />
                )}
                {!zone.isComplete &&
                  points.length > 1 &&
                  points.map((point, i) => {
                    if (i === points.length - 1) return null;
                    const nextPoint = points[i + 1];
                    return (
                      <line
                        key={i}
                        x1={`${point.x * 100}%`}
                        y1={`${point.y * 100}%`}
                        x2={`${nextPoint.x * 100}%`}
                        y2={`${nextPoint.y * 100}%`}
                        stroke={color}
                        strokeWidth="3"
                        strokeDasharray="8,4"
                        style={{
                          filter: "drop-shadow(0 0 4px rgba(0,0,0,0.4))",
                        }}
                      />
                    );
                  })}
                {points.map((point, i) => (
                  <g key={i}>
                    <circle
                      cx={`${point.x * 100}%`}
                      cy={`${point.y * 100}%`}
                      r={
                        dragTarget?.zoneId === zone.id &&
                        dragTarget?.pointIdx === i
                          ? 18
                          : hoveredHandle?.zoneId === zone.id &&
                            hoveredHandle?.pointIdx === i
                          ? 16
                          : 14
                      }
                      fill={color}
                      stroke="#fff"
                      strokeWidth="4"
                      style={{
                        filter: "drop-shadow(0 0 8px rgba(0,0,0,0.6))",
                        pointerEvents: "auto",
                        cursor: "grab",
                        transition: "r 0.15s cubic-bezier(.4,2,.6,1)",
                      }}
                      onMouseDown={handleHandleMouseDown(zone.id, i)}
                      onMouseEnter={() =>
                        setHoveredHandle({ zoneId: zone.id, pointIdx: i })
                      }
                      onMouseLeave={() => setHoveredHandle(null)}
                    />
                    <text
                      x={`${point.x * 100}%`}
                      y={`${point.y * 100}%`}
                      fontSize="14"
                      fontWeight="bold"
                      fill="#fff"
                      textAnchor="middle"
                      dominantBaseline="central"
                      style={{
                        fontFamily: "Inter, sans-serif",
                        filter: "drop-shadow(0 0 3px #000)",
                        pointerEvents: "none",
                      }}
                    >
                      {i + 1}
                    </text>
                  </g>
                ))}
                {zone.isComplete && points.length >= 3 && (
                  <g>
                    <rect
                      x={`${
                        (points.reduce((sum, p) => sum + p.x, 0) /
                          points.length) *
                          100 -
                        10
                      }%`}
                      y={`${
                        (points.reduce((sum, p) => sum + p.y, 0) /
                          points.length) *
                          100 -
                        3
                      }%`}
                      width="20%"
                      height="6%"
                      fill="rgba(0, 0, 0, 0.8)"
                      rx="8"
                      style={{
                        filter: "drop-shadow(0 2px 8px rgba(0,0,0,0.4))",
                      }}
                    />
                    <text
                      x={`${
                        (points.reduce((sum, p) => sum + p.x, 0) /
                          points.length) *
                        100
                      }%`}
                      y={`${
                        (points.reduce((sum, p) => sum + p.y, 0) /
                          points.length) *
                        100
                      }%`}
                      fontSize="14"
                      fontWeight="bold"
                      fill="#fff"
                      textAnchor="middle"
                      dominantBaseline="central"
                      style={{
                        fontFamily: "Inter, sans-serif",
                        pointerEvents: "none",
                      }}
                    >
                      {zone.name}
                    </text>
                  </g>
                )}
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );

  // Render the shared layout with controls, status, list, and preview
  return (
    <div className="w-full space-y-8">
      {/* Header Section */}
      <div className="text-center space-y-4">
        <h2 className="text-3xl font-bold text-white">
          Configure Detection Zones
        </h2>
        <p className="text-white/70 text-lg max-w-3xl mx-auto leading-relaxed">
          Draw polygonal zones on the video frame where you want to monitor
          crowd density. Click to add points and create detection areas with
          customizable alert thresholds.
        </p>
      </div>
      {/* Main Content - Controls and Preview side by side */}
      <div className="grid grid-cols-1 xl:grid-cols-5 gap-8">
        {/* Left Column: Controls & Status */}
        <div className="xl:col-span-2 space-y-6">
          <div className="glass-card rounded-3xl border-white/10 p-6 h-fit">
            {/* Status Card */}
            {statusCard && <div className="mb-6">{statusCard}</div>}
            {/* Controls */}
            <div className="space-y-6">{controls}</div>
          </div>
        </div>
        {/* Right Column: Preview */}
        <div className="xl:col-span-3">
          <div className="h-full flex flex-col">{preview}</div>
        </div>
      </div>
      {/* List Section - Full width below grid */}
      <div className="w-full">{listSection}</div>
    </div>
  );
};

export default PolygonZoneTool;
// End of PolygonZoneTool.tsx
