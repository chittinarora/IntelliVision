import React, { useState, useRef, useEffect, useCallback } from "react";
import { RotateCcw, CheckCircle, AlertTriangle, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import SharedDrawingLayout from "./SharedDrawingLayout";

// LineDrawingTool.tsx
// This component allows users to draw two counting lines on a video frame for people counting analytics.
// It handles video frame extraction, line drawing, dragging endpoints, and communicates line data to the parent.

// SNAP_STEP: Used to snap coordinates to a grid for easier line placement
const SNAP_STEP = 10;
// snap: Helper function to snap a value to the nearest grid step
const snap = (val: number): number => Math.round(val / SNAP_STEP) * SNAP_STEP;

// LineData: Represents a line with start and end coordinates (normalized 0-1)
interface LineData {
  startX: number;
  startY: number;
  endX: number;
  endY: number;
  inDirection?: "UP" | "DOWN" | "LR" | "RL";
}

// Add direction options with backend-compatible values
const DIRECTION_OPTIONS = [
  { value: "UP", label: "Up" },
  { value: "DOWN", label: "Down" },
  { value: "RL", label: "Left" },
  { value: "LR", label: "Right" },
];

// Props for the LineDrawingTool
interface LineDrawingToolProps {
  videoFile: File; // The video file to extract a frame from
  onLinesChange: (lines: [LineData, LineData] | null) => void; // Callback when lines are updated
  selectedLines: [LineData, LineData] | null; // Pre-selected lines (optional)
}

const LineDrawingTool: React.FC<LineDrawingToolProps> = ({
  videoFile,
  onLinesChange,
}) => {
  // State for the extracted image frame
  const [imageSrc, setImageSrc] = useState<string>("");
  // State for drawn lines
  const [lines, setLines] = useState<LineData[]>([]);
  // State for drawing mode
  const [isDrawing, setIsDrawing] = useState(false);
  // State for the line currently being drawn
  const [currentLine, setCurrentLine] = useState<Partial<LineData> | null>(
    null
  );
  // State for dragging a line endpoint
  const [dragTarget, setDragTarget] = useState<{
    lineIdx: number;
    pt: "start" | "end";
  } | null>(null);
  // State for hovered handle (for UI feedback)
  const [hoveredHandle, setHoveredHandle] = useState<{
    lineIdx: number;
    pt: "start" | "end";
  } | null>(null);
  // Add a toggle for 'both lines as in/out'
  const [bothInOut, setBothInOut] = useState(false);

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
    // Calculate normalized coordinates
    let x = (clientX - imageRect.left) / imageRect.width;
    let y = (clientY - imageRect.top) / imageRect.height;
    // Snap to grid for easier placement
    x = (Math.round((x * 1000) / SNAP_STEP) * SNAP_STEP) / 1000;
    y = (Math.round((y * 1000) / SNAP_STEP) * SNAP_STEP) / 1000;
    // Clamp between 0 and 1
    return { x: Math.max(0, Math.min(1, x)), y: Math.max(0, Math.min(1, y)) };
  };

  // Handle mouse down event to start drawing a new line
  const handleMouseDown = (e: React.MouseEvent) => {
    // Prevent drawing more than 2 lines or starting a new line while dragging
    if (lines.length >= 2 || dragTarget) return;

    const coords = getCoordinates(e.clientX, e.clientY);
    // Ignore if click is outside image
    if (coords.x === 0 && coords.y === 0) return;

    setIsDrawing(true);
    // Start a new line at the clicked position
    setCurrentLine({
      startX: coords.x,
      startY: coords.y,
      endX: coords.x,
      endY: coords.y,
    });
  };

  // Handle mouse move event to update drawing or drag endpoints
  const handleMouseMove = (e: React.MouseEvent) => {
    const coords = getCoordinates(e.clientX, e.clientY);

    if (dragTarget) {
      // If dragging an endpoint, update the corresponding line's endpoint
      setLines((prev) => {
        const updated = [...prev];
        if (updated[dragTarget.lineIdx]) {
          if (dragTarget.pt === "start") {
            updated[dragTarget.lineIdx] = {
              ...updated[dragTarget.lineIdx],
              startX: coords.x,
              startY: coords.y,
            };
          } else {
            updated[dragTarget.lineIdx] = {
              ...updated[dragTarget.lineIdx],
              endX: coords.x,
              endY: coords.y,
            };
          }
          // Notify parent if both lines are present
          if (updated.length === 2) {
            onLinesChange([updated[0], updated[1]]);
          }
        }
        return updated;
      });
    } else if (isDrawing && currentLine) {
      // If drawing a new line, update its end point as the mouse moves
      setCurrentLine({ ...currentLine, endX: coords.x, endY: coords.y });
    }
  };

  // Handle mouse up event to finish drawing or dragging
  const handleMouseUp = () => {
    if (dragTarget) {
      // Stop dragging endpoint
      setDragTarget(null);
      return;
    }

    // If finished drawing a line, add it to the lines array
    if (
      isDrawing &&
      currentLine?.startX !== undefined &&
      currentLine?.endX !== undefined
    ) {
      const newLine: LineData = {
        startX: currentLine.startX,
        startY: currentLine.startY!,
        endX: currentLine.endX,
        endY: currentLine.endY!,
        inDirection: "UP", // default to 'UP' for new lines
      };
      const newLines = [...lines, newLine];
      setLines(newLines);
      setIsDrawing(false);
      setCurrentLine(null);

      // Notify parent if both lines are present
      if (newLines.length === 2) {
        onLinesChange([newLines[0], newLines[1]]);
      }
    }
  };

  // Handle mouse down on a line endpoint to start dragging
  const handleHandleMouseDown =
    (lineIdx: number, pt: "start" | "end") => (e: React.MouseEvent) => {
      e.stopPropagation(); // Prevent starting a new line
      setDragTarget({ lineIdx, pt });
    };

  // Reset all lines and drawing state
  const resetLines = () => {
    setLines([]);
    setCurrentLine(null);
    setIsDrawing(false);
    setDragTarget(null);
    setHoveredHandle(null);
    onLinesChange(null);
  };

  // Undo the last drawn line
  const handleUndo = () => {
    if (lines.length > 0) {
      const newLines = lines.slice(0, -1);
      setLines(newLines);
      onLinesChange(newLines.length === 2 ? [newLines[0], newLines[1]] : null);
    }
  };

  // Whether both lines are drawn
  const isComplete = lines.length === 2;

  // Status card UI (shows progress/completion)
  const statusCard = (
    <div
      className={`p-6 rounded-2xl backdrop-blur-sm border transition-all ${
        isComplete
          ? "bg-emerald-500/10 border-emerald-400/30"
          : "bg-amber-500/10 border-amber-400/30"
      }`}
    >
      <div className="flex items-center gap-4">
        <div className="flex-1">
          <h3
            className={`font-bold text-lg mb-2 flex items-center gap-3 ${
              isComplete ? "text-emerald-300" : "text-amber-300"
            }`}
          >
            {isComplete ? (
              <CheckCircle className="w-5 h-5 text-emerald-400" />
            ) : (
              <AlertTriangle className="w-5 h-5 text-amber-400" />
            )}
            {lines.length}/2 Lines Drawn
          </h3>
          <p
            className={`text-sm ${
              isComplete ? "text-emerald-200/80" : "text-amber-200/80"
            }`}
          >
            {isComplete
              ? "Both counting lines have been set up successfully"
              : `Please draw ${2 - lines.length} more line${
                  2 - lines.length > 1 ? "s" : ""
                }`}
          </p>
        </div>
      </div>
    </div>
  );

  // Controls UI (undo, reset)
  const controls = (
    <div className="space-y-4">
      <div className="flex gap-3">
        <Button
          variant="ghost"
          onClick={handleUndo}
          disabled={lines.length === 0}
          className="flex-1 glass-button"
          size="sm"
        >
          Undo
        </Button>
        <Button
          variant="ghost"
          onClick={resetLines}
          className="flex-1 glass-button"
          size="sm"
        >
          <RotateCcw className="w-4 h-4 mr-1" />
          Reset
        </Button>
      </div>
      {/* Per-line direction segmented buttons (up/down/left/right) */}
      {lines.length > 0 && (
        <div className="space-y-3">
          {lines.map((line, idx) => (
            <div key={idx} className="flex items-center gap-3">
              <span className="text-white text-md font-semibold min-w-max">
                Line {idx + 1}:
              </span>
              <div className="flex flex-1 rounded-xl overflow-hidden backdrop-blur-xl bg-white/10 shadow-lg">
                {/* Direction selector as segmented buttons (styled like previous In/Out) */}

                {DIRECTION_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    type="button"
                    className={`flex-1 px-4 py-2 text-sm border-[0.1px] font-bold transition focus:outline-none focus:ring-1 focus:ring-cyan-400 focus:z-10
                      ${
                        line.inDirection === opt.value
                          ? "bg-cyan-500/60 backdrop-blur-lg text-white shadow-inner border-cyan-500"
                          : "bg-transparent text-cyan-200 hover:bg-cyan-400/20 border-white/20"
                      }
                    `}
                    onClick={() => {
                      const updated = lines.map((l, i) =>
                        i === idx
                          ? {
                              ...l,
                              inDirection: opt.value as
                                | "UP"
                                | "DOWN"
                                | "LR"
                                | "RL",
                            }
                          : l
                      );
                      setLines(updated);
                      if (updated.length === 2)
                        onLinesChange([
                          {
                            ...updated[0],
                            inDirection: updated[0].inDirection,
                          },
                          {
                            ...updated[1],
                            inDirection: updated[1].inDirection,
                          },
                        ]);
                    }}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
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
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => {
          setIsDrawing(false);
          setCurrentLine(null);
          setDragTarget(null);
          setHoveredHandle(null);
        }}
      >
        <img
          ref={imageRef}
          src={imageSrc}
          alt="Video frame for line drawing"
          className="w-full h-auto rounded-2xl block select-none"
          style={{
            display: "block",
            width: "100%",
            height: "auto",
            maxHeight: 500,
            objectFit: "contain",
          }}
          draggable={false}
        />

        {/* SVG Overlay for lines */}
        <svg
          className="absolute inset-0 w-full h-full"
          style={{ borderRadius: "1rem", pointerEvents: "none" }}
        >
          {/* ... keep existing code (SVG line rendering logic) */}
          {lines.map((line, index) => {
            const color = index === 0 ? "#06b6d4" : "#3b82f6";
            return (
              <g key={index}>
                <line
                  x1={`${line.startX * 100}%`}
                  y1={`${line.startY * 100}%`}
                  x2={`${line.endX * 100}%`}
                  y2={`${line.endY * 100}%`}
                  stroke={color}
                  strokeWidth="8"
                  strokeLinecap="round"
                  style={{ filter: "drop-shadow(0 0 6px rgba(0,0,0,0.4))" }}
                />
                <line
                  x1={`${line.startX * 100}%`}
                  y1={`${line.startY * 100}%`}
                  x2={`${line.endX * 100}%`}
                  y2={`${line.endY * 100}%`}
                  stroke="#fff"
                  strokeWidth="4"
                  strokeLinecap="round"
                  opacity="0.9"
                />
                {[
                  ["start", line.startX, line.startY],
                  ["end", line.endX, line.endY],
                ].map(([pt, x, y]) => (
                  <circle
                    key={pt as string}
                    cx={`${(x as number) * 100}%`}
                    cy={`${(y as number) * 100}%`}
                    r={
                      dragTarget?.lineIdx === index && dragTarget?.pt === pt
                        ? 16
                        : 12
                    }
                    fill={color}
                    stroke="#fff"
                    strokeWidth="3"
                    style={{
                      pointerEvents: "all",
                      cursor: "grab",
                      filter: "drop-shadow(0 0 4px rgba(0,0,0,0.5))",
                    }}
                    onMouseDown={handleHandleMouseDown(
                      index,
                      pt as "start" | "end"
                    )}
                    onMouseEnter={() =>
                      setHoveredHandle({
                        lineIdx: index,
                        pt: pt as "start" | "end",
                      })
                    }
                    onMouseLeave={() => setHoveredHandle(null)}
                  />
                ))}
                <text
                  x={`${line.startX * 100 + 2}%`}
                  y={`${line.startY * 100 - 2}%`}
                  fontSize="18"
                  fontWeight="bold"
                  fill="#fff"
                  style={{
                    fontFamily: "Inter, sans-serif",
                    pointerEvents: "none",
                    filter: "drop-shadow(0 0 3px #000)",
                  }}
                >
                  Line {index + 1}
                </text>
              </g>
            );
          })}
          {currentLine &&
            currentLine.startX !== undefined &&
            currentLine.endX !== undefined && (
              <line
                x1={`${currentLine.startX * 100}%`}
                y1={`${currentLine.startY! * 100}%`}
                x2={`${currentLine.endX * 100}%`}
                y2={`${currentLine.endY! * 100}%`}
                stroke={lines.length === 0 ? "#06b6d4" : "#3b82f6"}
                strokeWidth="8"
                strokeDasharray="12,6"
                strokeLinecap="round"
                opacity="0.8"
              />
            )}
        </svg>
      </div>
    </div>
  );

  // Render the shared layout with controls, status, and preview
  return (
    <SharedDrawingLayout
      title="Draw Counting Lines"
      description="Draw two lines across which people will move. You can drag endpoints after drawing. After drawing the lines, choose the movement direction which counts as in for each line."
      statusCard={statusCard}
      controls={controls}
      preview={preview}
    />
  );
};

export default LineDrawingTool;
// End of LineDrawingTool.tsx
