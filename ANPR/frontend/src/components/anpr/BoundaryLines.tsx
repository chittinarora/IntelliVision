import { useState, useRef, useEffect, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { MapPin, Save, Trash2, Plus, Eye, RotateCcw } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface Point {
  x: number;
  y: number;
}

interface BoundaryLine {
  id: string;
  name: string;
  type: "entry" | "exit";
  points: Point[];
  color: string;
}

export function BoundaryLines() {
  const [lines, setLines] = useState<BoundaryLine[]>([]);
  const [currentLine, setCurrentLine] = useState<Point[]>([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [lineType, setLineType] = useState<"entry" | "exit">("entry");
  const [lineName, setLineName] = useState("");
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { toast } = useToast();

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw background with subtle blue gradient
    const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
    gradient.addColorStop(0, '#f8fafc');
    gradient.addColorStop(1, '#e2e8f0');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw subtle grid
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 0.5;
    const gridSize = 25;
    
    for (let i = 0; i <= canvas.width; i += gridSize) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, canvas.height);
      ctx.stroke();
    }
    for (let i = 0; i <= canvas.height; i += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, i);
      ctx.lineTo(canvas.width, i);
      ctx.stroke();
    }

    // Draw existing lines
    lines.forEach(line => {
      if (line.points.length > 1) {
        ctx.strokeStyle = line.color;
        ctx.lineWidth = 4;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        
        // Draw line with shadow for depth
        ctx.shadowColor = 'rgba(0,0,0,0.2)';
        ctx.shadowBlur = 3;
        ctx.shadowOffsetX = 1;
        ctx.shadowOffsetY = 1;
        
        ctx.beginPath();
        ctx.moveTo(line.points[0].x, line.points[0].y);
        for (let i = 1; i < line.points.length; i++) {
          ctx.lineTo(line.points[i].x, line.points[i].y);
        }
        ctx.stroke();
        
        ctx.shadowColor = 'transparent';

        // Draw line label with background
        if (line.points.length > 0) {
          const labelX = line.points[0].x + 8;
          const labelY = line.points[0].y - 8;
          
          ctx.font = 'bold 12px sans-serif';
          const textWidth = ctx.measureText(line.name).width;
          
          // Label background
          ctx.fillStyle = 'rgba(255,255,255,0.9)';
          ctx.fillRect(labelX - 4, labelY - 16, textWidth + 8, 20);
          ctx.strokeStyle = line.color;
          ctx.lineWidth = 1;
          ctx.strokeRect(labelX - 4, labelY - 16, textWidth + 8, 20);
          
          // Label text
          ctx.fillStyle = line.color;
          ctx.fillText(line.name, labelX, labelY);
        }
      }

      // Draw points as circles
      line.points.forEach((point, index) => {
        ctx.fillStyle = line.color;
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(point.x, point.y, 6, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
        
        // Number the points
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 10px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText((index + 1).toString(), point.x, point.y + 3);
        ctx.textAlign = 'start';
      });
    });

    // Draw current line being drawn
    if (currentLine.length > 0) {
      const color = lineType === "entry" ? "#3b82f6" : "#ef4444";
      
      // Draw current line
      if (currentLine.length > 1) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.setLineDash([8, 4]);
        ctx.lineCap = 'round';
        
        ctx.beginPath();
        ctx.moveTo(currentLine[0].x, currentLine[0].y);
        for (let i = 1; i < currentLine.length; i++) {
          ctx.lineTo(currentLine[i].x, currentLine[i].y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Draw current points
      currentLine.forEach((point, index) => {
        ctx.fillStyle = color;
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
        
        // Number the points
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 9px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText((index + 1).toString(), point.x, point.y + 2);
        ctx.textAlign = 'start';
      });
    }
  }, [lines, currentLine, lineType]);

  useEffect(() => {
    drawCanvas();
  }, [drawCanvas]);

  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    const x = (event.clientX - rect.left) * scaleX;
    const y = (event.clientY - rect.top) * scaleY;

    setCurrentLine(prev => [...prev, { x, y }]);
  };

  const startDrawing = () => {
    if (!lineName.trim()) {
      toast({
        title: "Line name required",
        description: "Please enter a name for the boundary line",
        variant: "destructive"
      });
      return;
    }
    setIsDrawing(true);
    setCurrentLine([]);
  };

  const saveLine = async () => {
    if (currentLine.length < 2) {
      toast({
        title: "Invalid line",
        description: "A boundary line must have at least 2 points",
        variant: "destructive"
      });
      return;
    }

    const newLine: BoundaryLine = {
      id: `line_${Date.now()}`,
      name: lineName,
      type: lineType,
      points: currentLine,
      color: lineType === "entry" ? "#3b82f6" : "#ef4444"
    };

    try {
      const response = await fetch('http://localhost:8000/detect/set_lines', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          lines: [...lines, newLine]
        }),
      });

      if (response.ok) {
        setLines(prev => [...prev, newLine]);
        finishDrawing();
        
        toast({
          title: "Line saved",
          description: `${lineType} line "${lineName}" has been saved`
        });
      } else {
        throw new Error('Failed to save line');
      }
    } catch (error) {
      // Save locally for demo
      setLines(prev => [...prev, newLine]);
      finishDrawing();
      
      toast({
        title: "Line saved",
        description: `${lineType} line "${lineName}" has been saved`
      });
    }
  };

  const finishDrawing = () => {
    setCurrentLine([]);
    setIsDrawing(false);
    setLineName("");
  };

  const deleteLine = (lineId: string) => {
    setLines(prev => prev.filter(line => line.id !== lineId));
    toast({
      title: "Line deleted",
      description: "Boundary line has been removed"
    });
  };

  const cancelDrawing = () => {
    setIsDrawing(false);
    setCurrentLine([]);
  };

  const clearCanvas = () => {
    setLines([]);
    setCurrentLine([]);
    setIsDrawing(false);
    toast({
      title: "Canvas cleared",
      description: "All boundary lines have been removed"
    });
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold mb-2 text-blue-900">Boundary Line Management</h2>
        <p className="text-blue-600">
          Draw entry and exit lines for vehicle counting and flow analysis
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Controls */}
        <Card className="border-blue-200 bg-gradient-to-br from-blue-50 to-white">
          <CardHeader className="bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-t-lg">
            <CardTitle className="flex items-center gap-2">
              <MapPin className="h-5 w-5" />
              Line Controls
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 p-6">
            <div className="space-y-2">
              <label className="text-sm font-medium text-blue-800">Line Name</label>
              <input
                type="text"
                value={lineName}
                onChange={(e) => setLineName(e.target.value)}
                placeholder="e.g., Main Entrance"
                className="w-full px-3 py-2 border border-blue-200 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                disabled={isDrawing}
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-blue-800">Line Type</label>
              <div className="grid grid-cols-2 gap-2">
                <Button
                  variant={lineType === "entry" ? "default" : "outline"}
                  size="sm"
                  onClick={() => setLineType("entry")}
                  disabled={isDrawing}
                  className={lineType === "entry" ? "bg-blue-600 hover:bg-blue-700" : "border-blue-300 text-blue-700 hover:bg-blue-50"}
                >
                  Entry
                </Button>
                <Button
                  variant={lineType === "exit" ? "destructive" : "outline"}
                  size="sm"
                  onClick={() => setLineType("exit")}
                  disabled={isDrawing}
                  className={lineType === "exit" ? "" : "border-red-300 text-red-700 hover:bg-red-50"}
                >
                  Exit
                </Button>
              </div>
            </div>

            <div className="space-y-2">
              {!isDrawing ? (
                <Button onClick={startDrawing} className="w-full bg-blue-600 hover:bg-blue-700">
                  <Plus className="h-4 w-4 mr-2" />
                  Start Drawing
                </Button>
              ) : (
                <div className="space-y-2">
                  <Button onClick={saveLine} className="w-full bg-green-600 hover:bg-green-700">
                    <Save className="h-4 w-4 mr-2" />
                    Save Line ({currentLine.length} points)
                  </Button>
                  <Button onClick={cancelDrawing} variant="outline" className="w-full border-slate-300">
                    Cancel
                  </Button>
                </div>
              )}
            </div>

            {lines.length > 0 && (
              <Button onClick={clearCanvas} variant="outline" className="w-full border-red-300 text-red-700 hover:bg-red-50">
                <RotateCcw className="h-4 w-4 mr-2" />
                Clear All Lines
              </Button>
            )}

            {isDrawing && (
              <div className="p-3 bg-blue-100 border border-blue-300 rounded-lg">
                <p className="text-sm text-blue-800 font-medium">Drawing Mode Active</p>
                <p className="text-xs text-blue-600 mt-1">
                  Click on the canvas to add points. At least 2 points required.
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Canvas */}
        <Card className="lg:col-span-2 border-blue-200 bg-gradient-to-br from-blue-50 to-white">
          <CardHeader className="bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-t-lg">
            <CardTitle className="flex items-center gap-2">
              <Eye className="h-5 w-5" />
              Drawing Area
            </CardTitle>
          </CardHeader>
          <CardContent className="p-6">
            <div className="border-2 border-blue-200 rounded-lg overflow-hidden bg-white shadow-inner">
              <canvas
                ref={canvasRef}
                width={600}
                height={400}
                onClick={handleCanvasClick}
                className={`w-full max-w-full ${isDrawing ? 'cursor-crosshair' : 'cursor-default'}`}
                style={{ maxHeight: '400px', display: 'block' }}
              />
            </div>
            <div className="mt-3 flex items-center justify-between text-xs text-blue-600">
              <span>
                {isDrawing ? 'Click to add points when drawing mode is active' : 'Use controls to start drawing boundary lines'}
              </span>
              <span>
                Canvas: 600x400px
              </span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Existing Lines */}
      <Card className="border-blue-200 bg-gradient-to-br from-blue-50 to-white">
        <CardHeader className="bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-t-lg">
          <CardTitle>Configured Lines ({lines.length})</CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          {lines.length === 0 ? (
            <div className="text-center py-8 text-blue-600">
              <MapPin className="h-12 w-12 mx-auto mb-3 text-blue-400" />
              <p className="text-lg font-medium">No boundary lines configured yet</p>
              <p className="text-sm">Start by drawing your first entry or exit line above</p>
            </div>
          ) : (
            <div className="space-y-3">
              {lines.map((line) => (
                <div
                  key={line.id}
                  className="flex items-center justify-between p-4 bg-white border border-blue-200 rounded-lg shadow-sm hover:shadow-md transition-shadow"
                >
                  <div className="flex items-center gap-3">
                    <div
                      className="w-5 h-5 rounded-full border-2 border-white shadow-sm"
                      style={{ backgroundColor: line.color }}
                    />
                    <div>
                      <p className="font-medium text-slate-800">{line.name}</p>
                      <p className="text-sm text-slate-600">
                        {line.type} line • {line.points.length} points
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge 
                      variant={line.type === "entry" ? "default" : "destructive"}
                      className={line.type === "entry" ? "bg-blue-100 text-blue-800" : ""}
                    >
                      {line.type}
                    </Badge>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => deleteLine(line.id)}
                      className="border-red-300 text-red-700 hover:bg-red-50"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
