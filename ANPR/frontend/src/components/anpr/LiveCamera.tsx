import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Camera, Square, Play, AlertCircle, Zap, Activity } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface Detection {
  plate: string;
  confidence: number;
  timestamp: string;
  bbox: [number, number, number, number];
}

export function LiveCamera() {
  const [isActive, setIsActive] = useState(false);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [cameraStatus, setCameraStatus] = useState<'idle' | 'connecting' | 'connected' | 'error'>('idle');
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const { toast } = useToast();

  const startCamera = async () => {
    setCameraStatus('connecting');
    
    try {
      const response = await fetch('http://localhost:8000/detect/start_camera', {
        method: 'POST',
      });

      if (response.ok) {
        setIsActive(true);
        setCameraStatus('connected');
        toast({
          title: "Camera started",
          description: "Live detection is now active"
        });
        
        startDetectionPolling();
      } else {
        throw new Error('Failed to start camera');
      }
    } catch (error) {
      setCameraStatus('error');
      toast({
        title: "Camera error",
        description: "Failed to start camera. Please check connection.",
        variant: "destructive"
      });
    }
  };

  const stopCamera = async () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    
    setIsActive(false);
    setCameraStatus('idle');
    setDetections([]);
    
    toast({
      title: "Camera stopped",
      description: "Live detection has been stopped"
    });
  };

  const startDetectionPolling = () => {
    intervalRef.current = setInterval(() => {
      if (isActive && cameraStatus === 'connected') {
        if (Math.random() > 0.85) {
          const mockDetection: Detection = {
            plate: `${['ABC', 'XYZ', 'DEF', 'GHI'][Math.floor(Math.random() * 4)]}${Math.floor(Math.random() * 900) + 100}`,
            confidence: 0.75 + Math.random() * 0.2,
            timestamp: new Date().toISOString(),
            bbox: [100 + Math.random() * 100, 100 + Math.random() * 50, 200, 150]
          };
          
          setDetections(prev => [mockDetection, ...prev.slice(0, 9)]);
        }
      }
    }, 3000);
  };

  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  const getCameraStatusColor = () => {
    switch (cameraStatus) {
      case 'connected': return 'text-green-600 bg-green-100';
      case 'connecting': return 'text-yellow-600 bg-yellow-100';
      case 'error': return 'text-red-600 bg-red-100';
      default: return 'text-slate-600 bg-slate-100';
    }
  };

  const getCameraStatusText = () => {
    switch (cameraStatus) {
      case 'connected': return '🟢 Live';
      case 'connecting': return '🟡 Connecting...';
      case 'error': return '🔴 Error';
      default: return '⚫ Inactive';
    }
  };

  return (
    <div className="space-y-8">
      <Card className="glass-effect card-shadow-lg border-0 overflow-hidden">
        <CardHeader className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white">
          <CardTitle className="flex items-center gap-3 text-xl">
            <div className="w-8 h-8 bg-white/20 rounded-lg flex items-center justify-center">
              <Camera className="h-5 w-5" />
            </div>
            Live Camera Detection
            {isActive && (
              <div className="ml-auto flex items-center gap-2">
                <Activity className="h-4 w-4 animate-pulse" />
                <span className="text-sm">LIVE</span>
              </div>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="p-8">
          <div className="flex items-center gap-6 mb-8">
            <Button
              onClick={isActive ? stopCamera : startCamera}
              disabled={cameraStatus === 'connecting'}
              className={`flex items-center gap-3 px-8 py-4 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 ${
                isActive 
                  ? 'bg-red-600 hover:bg-red-700 text-white' 
                  : 'bg-blue-600 hover:bg-blue-700 text-white'
              }`}
              size="lg"
            >
              {isActive ? (
                <>
                  <Square className="h-5 w-5" />
                  Stop Camera
                </>
              ) : (
                <>
                  <Play className="h-5 w-5" />
                  {cameraStatus === 'connecting' ? 'Starting...' : 'Start Camera'}
                </>
              )}
            </Button>
            
            <Badge 
              className={`px-4 py-2 text-sm font-medium rounded-xl ${getCameraStatusColor()} ${isActive ? 'animate-pulse' : ''}`}
            >
              <Zap className="h-4 w-4 mr-2" />
              {getCameraStatusText()}
            </Badge>

            {isActive && (
              <div className="flex items-center gap-2 text-sm text-slate-600">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                Recording & Analyzing
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Camera Feed */}
            <Card className="glass-effect border-blue-200">
              <CardContent className="p-6">
                <h3 className="font-semibold text-slate-800 mb-4 flex items-center gap-2">
                  <Camera className="h-5 w-5 text-blue-600" />
                  Camera Feed
                </h3>
                <div className="aspect-video bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl flex items-center justify-center border-2 border-blue-200 shadow-inner">
                  {cameraStatus === 'connected' ? (
                    <div className="text-center text-white">
                      <div className="w-16 h-16 mx-auto mb-4 bg-green-500/20 rounded-2xl flex items-center justify-center">
                        <Camera className="h-8 w-8 text-green-400" />
                      </div>
                      <p className="text-lg font-medium mb-2">Live camera feed active</p>
                      <p className="text-sm text-slate-300 mb-4">Detecting license plates in real-time...</p>
                      <div className="flex justify-center items-center gap-2">
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" style={{animationDelay: '0.4s'}}></div>
                      </div>
                    </div>
                  ) : cameraStatus === 'connecting' ? (
                    <div className="text-center text-white">
                      <div className="w-16 h-16 mx-auto mb-4 bg-yellow-500/20 rounded-2xl flex items-center justify-center">
                        <Camera className="h-8 w-8 text-yellow-400 animate-pulse" />
                      </div>
                      <p className="text-lg font-medium">Connecting to camera...</p>
                      <p className="text-sm text-slate-300">Please wait while we establish connection</p>
                    </div>
                  ) : cameraStatus === 'error' ? (
                    <div className="text-center text-red-300">
                      <div className="w-16 h-16 mx-auto mb-4 bg-red-500/20 rounded-2xl flex items-center justify-center">
                        <AlertCircle className="h-8 w-8 text-red-400" />
                      </div>
                      <p className="text-lg font-medium">Camera connection failed</p>
                      <p className="text-sm text-slate-300">Please check your camera connection</p>
                    </div>
                  ) : (
                    <div className="text-center text-slate-400">
                      <div className="w-16 h-16 mx-auto mb-4 bg-slate-600/20 rounded-2xl flex items-center justify-center">
                        <Camera className="h-8 w-8" />
                      </div>
                      <p className="text-lg font-medium">Camera not active</p>
                      <p className="text-sm text-slate-300">Click "Start Camera" to begin detection</p>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Recent Detections */}
            <Card className="glass-effect border-blue-200">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold text-slate-800 flex items-center gap-2">
                    <Activity className="h-5 w-5 text-indigo-600" />
                    Recent Detections
                  </h3>
                  <Badge variant="outline" className="text-xs">
                    {detections.length} total
                  </Badge>
                </div>
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {detections.length === 0 ? (
                    <div className="p-6 bg-gradient-to-br from-blue-50 to-indigo-50 border border-blue-200 rounded-xl text-center">
                      <div className="w-12 h-12 mx-auto mb-3 bg-blue-100 rounded-xl flex items-center justify-center">
                        <Zap className="h-6 w-6 text-blue-600" />
                      </div>
                      <p className="font-medium text-blue-800 mb-1">
                        {isActive ? 'Waiting for detections...' : 'No detections yet'}
                      </p>
                      <p className="text-sm text-blue-600">
                        {isActive ? 'System is monitoring for license plates' : 'Start camera to begin detection'}
                      </p>
                    </div>
                  ) : (
                    detections.map((detection, index) => (
                      <Card
                        key={index}
                        className="glass-effect border-slate-200 hover:border-blue-300 transition-all duration-200 animate-scale-in"
                      >
                        <CardContent className="p-4">
                          <div className="flex justify-between items-center mb-2">
                            <span className="font-mono font-bold text-xl text-slate-800">
                              {detection.plate}
                            </span>
                            <Badge 
                              variant="secondary" 
                              className="bg-blue-100 text-blue-800 border-blue-200"
                            >
                              {(detection.confidence * 100).toFixed(1)}%
                            </Badge>
                          </div>
                          <div className="flex items-center gap-2 text-xs text-slate-600">
                            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                            <span>{new Date(detection.timestamp).toLocaleTimeString()}</span>
                          </div>
                        </CardContent>
                      </Card>
                    ))
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
