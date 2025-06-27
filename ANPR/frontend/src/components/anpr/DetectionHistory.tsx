
import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Search, Filter, RefreshCw } from "lucide-react";
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
    if (confidence >= 0.9) return "bg-green-100 text-green-800";
    if (confidence >= 0.7) return "bg-yellow-100 text-yellow-800";
    return "bg-red-100 text-red-800";
  };

  const getSourceBadge = (source: string) => {
    return source === 'camera' ? 
      <Badge variant="outline" className="bg-blue-50 text-blue-700">📹 Camera</Badge> :
      <Badge variant="outline" className="bg-purple-50 text-purple-700">🎬 Video</Badge>;
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row gap-4 justify-between">
        <h2 className="text-2xl font-bold">Detection History</h2>
        <Button
          onClick={fetchHistory}
          disabled={isLoading}
          variant="outline"
          size="sm"
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      <Card>
        <CardHeader>
          <div className="flex flex-col sm:flex-row gap-4 justify-between">
            <CardTitle className="flex items-center gap-2">
              <Filter className="h-5 w-5" />
              Search & Filter
            </CardTitle>
            <div className="flex items-center gap-2">
              <Search className="h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search by plate, source, type, or location..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full sm:w-80"
              />
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="mb-4 flex justify-between items-center">
            <p className="text-sm text-muted-foreground">
              Showing {filteredDetections.length} of {detections.length} detections
            </p>
          </div>

          <div className="rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>License Plate</TableHead>
                  <TableHead>Timestamp</TableHead>
                  <TableHead>Source</TableHead>
                  <TableHead>Confidence</TableHead>
                  <TableHead>Vehicle Type</TableHead>
                  <TableHead>Location</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredDetections.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={6} className="text-center py-8">
                      <div className="text-muted-foreground">
                        {searchTerm ? 'No detections match your search' : 'No detections found'}
                      </div>
                    </TableCell>
                  </TableRow>
                ) : (
                  filteredDetections.map((detection) => (
                    <TableRow key={detection.id}>
                      <TableCell className="font-mono font-bold">
                        {detection.plate}
                      </TableCell>
                      <TableCell>
                        <div className="space-y-1">
                          <div className="text-sm">
                            {new Date(detection.timestamp).toLocaleDateString()}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {new Date(detection.timestamp).toLocaleTimeString()}
                          </div>
                        </div>
                      </TableCell>
                      <TableCell>
                        {getSourceBadge(detection.source)}
                      </TableCell>
                      <TableCell>
                        <Badge className={getConfidenceColor(detection.confidence)}>
                          {(detection.confidence * 100).toFixed(1)}%
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <Badge variant="secondary">
                          {detection.vehicle_type || 'Unknown'}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        {detection.location || 'Unknown'}
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
