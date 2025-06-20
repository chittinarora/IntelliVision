
import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Search, Filter, RefreshCw, History, Clock, Camera, Video } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface Detection {
  id: string;
  plate: string;
  timestamp: string;
  source: string;
  confidence: number;
  vehicle_type?: string;
  location?: string;
}

export function DetectionHistory() {
  const [detections, setDetections] = useState<Detection[]>([]);
  const [filteredDetections, setFilteredDetections] = useState<Detection[]>([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  const fetchHistory = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/detect/history');
      
      if (response.ok) {
        const data = await response.json();
        setDetections(data);
        setFilteredDetections(data);
      } else {
        throw new Error('Failed to fetch history');
      }
    } catch (error) {
      toast({
        title: "Failed to load history",
        description: "Please check your connection and try again",
        variant: "destructive"
      });
      
      // Mock data for demo
      const mockData: Detection[] = Array.from({ length: 20 }, (_, i) => ({
        id: `det_${i + 1}`,
        plate: `ABC${Math.floor(Math.random() * 1000).toString().padStart(3, '0')}`,
        timestamp: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(),
        source: Math.random() > 0.5 ? 'camera' : 'video',
        confidence: 0.7 + Math.random() * 0.3,
        vehicle_type: ['car', 'truck', 'motorcycle'][Math.floor(Math.random() * 3)],
        location: ['Entrance', 'Exit', 'Parking Area'][Math.floor(Math.random() * 3)]
      }));
      
      setDetections(mockData);
      setFilteredDetections(mockData);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  useEffect(() => {
    const filtered = detections.filter(detection =>
      detection.plate.toLowerCase().includes(searchTerm.toLowerCase()) ||
      detection.source.toLowerCase().includes(searchTerm.toLowerCase()) ||
      detection.vehicle_type?.toLowerCase().includes(searchTerm.toLowerCase()) ||
      detection.location?.toLowerCase().includes(searchTerm.toLowerCase())
    );
    setFilteredDetections(filtered);
  }, [searchTerm, detections]);

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return "bg-emerald-100 text-emerald-800 border-emerald-200";
    if (confidence >= 0.7) return "bg-amber-100 text-amber-800 border-amber-200";
    return "bg-red-100 text-red-800 border-red-200";
  };

  const getSourceBadge = (source: string) => {
    return source === 'camera' ? 
      <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200 hover:bg-blue-100">
        <Camera className="h-3 w-3 mr-1" />
        Live Camera
      </Badge> :
      <Badge variant="outline" className="bg-purple-50 text-purple-700 border-purple-200 hover:bg-purple-100">
        <Video className="h-3 w-3 mr-1" />
        Video Upload
      </Badge>;
  };

  const getVehicleTypeBadge = (type?: string) => {
    const colors = {
      car: "bg-blue-50 text-blue-700 border-blue-200",
      truck: "bg-orange-50 text-orange-700 border-orange-200",
      motorcycle: "bg-green-50 text-green-700 border-green-200"
    };
    const color = colors[type as keyof typeof colors] || "bg-gray-50 text-gray-700 border-gray-200";
    
    return (
      <Badge variant="outline" className={color}>
        {type || 'Unknown'}
      </Badge>
    );
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-col gap-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg">
              <History className="h-6 w-6 text-white" />
            </div>
            <div>
              <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-900 to-indigo-600 bg-clip-text text-transparent">
                Detection History
              </h2>
              <p className="text-slate-600">Review past license plate detections and search records</p>
            </div>
          </div>
          <Button
            onClick={fetchHistory}
            disabled={isLoading}
            className="bg-blue-600 hover:bg-blue-700 shadow-lg hover:shadow-xl transition-all duration-300"
            size="lg"
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh Data
          </Button>
        </div>
        
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card className="glass-effect border-blue-200">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                  <History className="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <p className="text-sm text-slate-600">Total Detections</p>
                  <p className="text-2xl font-bold text-slate-800">{detections.length}</p>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card className="glass-effect border-green-200">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
                  <Camera className="h-5 w-5 text-green-600" />
                </div>
                <div>
                  <p className="text-sm text-slate-600">Live Camera</p>
                  <p className="text-2xl font-bold text-slate-800">
                    {detections.filter(d => d.source === 'camera').length}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card className="glass-effect border-purple-200">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
                  <Video className="h-5 w-5 text-purple-600" />
                </div>
                <div>
                  <p className="text-sm text-slate-600">Video Upload</p>
                  <p className="text-2xl font-bold text-slate-800">
                    {detections.filter(d => d.source === 'video').length}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card className="glass-effect border-amber-200">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-amber-100 rounded-lg flex items-center justify-center">
                  <Clock className="h-5 w-5 text-amber-600" />
                </div>
                <div>
                  <p className="text-sm text-slate-600">Today</p>
                  <p className="text-2xl font-bold text-slate-800">
                    {detections.filter(d => 
                      new Date(d.timestamp).toDateString() === new Date().toDateString()
                    ).length}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Search and Filter Card */}
      <Card className="glass-effect border-blue-200 shadow-lg">
        <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50 border-b border-blue-100">
          <CardTitle className="flex items-center gap-3">
            <Filter className="h-5 w-5 text-blue-600" />
            Search & Filter Detections
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          <div className="flex items-center gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-slate-400" />
              <Input
                placeholder="Search by license plate, source, vehicle type, or location..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 h-12 rounded-xl border-blue-200 focus:border-blue-400 transition-colors"
              />
            </div>
            <Badge variant="outline" className="px-3 py-2 text-sm">
              {filteredDetections.length} of {detections.length} records
            </Badge>
          </div>
        </CardContent>
      </Card>

      {/* Detection Table */}
      <Card className="glass-effect border-blue-200 shadow-lg">
        <CardHeader className="bg-gradient-to-r from-slate-50 to-blue-50 border-b border-blue-100">
          <CardTitle>Detection Records</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow className="bg-slate-50/50 border-b border-blue-100">
                  <TableHead className="font-semibold text-slate-700">License Plate</TableHead>
                  <TableHead className="font-semibold text-slate-700">Date & Time</TableHead>
                  <TableHead className="font-semibold text-slate-700">Source</TableHead>
                  <TableHead className="font-semibold text-slate-700">Confidence</TableHead>
                  <TableHead className="font-semibold text-slate-700">Vehicle Type</TableHead>
                  <TableHead className="font-semibold text-slate-700">Location</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredDetections.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={6} className="text-center py-12">
                      <div className="flex flex-col items-center gap-3">
                        <div className="w-16 h-16 bg-slate-100 rounded-full flex items-center justify-center">
                          <Search className="h-8 w-8 text-slate-400" />
                        </div>
                        <div className="text-slate-500">
                          {searchTerm ? 'No detections match your search criteria' : 'No detection records found'}
                        </div>
                      </div>
                    </TableCell>
                  </TableRow>
                ) : (
                  filteredDetections.map((detection) => (
                    <TableRow key={detection.id} className="hover:bg-blue-50/50 transition-colors">
                      <TableCell>
                        <div className="font-mono font-bold text-lg bg-slate-100 px-3 py-1 rounded-lg inline-block">
                          {detection.plate}
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="space-y-1">
                          <div className="font-medium text-slate-800">
                            {new Date(detection.timestamp).toLocaleDateString('en-US', {
                              month: 'short',
                              day: 'numeric',
                              year: 'numeric'
                            })}
                          </div>
                          <div className="text-sm text-slate-500">
                            {new Date(detection.timestamp).toLocaleTimeString('en-US', {
                              hour: '2-digit',
                              minute: '2-digit'
                            })}
                          </div>
                        </div>
                      </TableCell>
                      <TableCell>
                        {getSourceBadge(detection.source)}
                      </TableCell>
                      <TableCell>
                        <Badge className={`${getConfidenceColor(detection.confidence)} font-medium`}>
                          {(detection.confidence * 100).toFixed(1)}%
                        </Badge>
                      </TableCell>
                      <TableCell>
                        {getVehicleTypeBadge(detection.vehicle_type)}
                      </TableCell>
                      <TableCell>
                        <span className="text-slate-600 font-medium">
                          {detection.location || 'Unknown'}
                        </span>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
