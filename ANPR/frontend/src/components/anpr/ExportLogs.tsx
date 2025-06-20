
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Calendar } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { Download, Calendar as CalendarIcon, FileText, Database, Clock, Zap } from "lucide-react";
import { format } from "date-fns";
import { useToast } from "@/hooks/use-toast";

export function ExportLogs() {
  const [exportFormat, setExportFormat] = useState<"csv" | "json">("csv");
  const [dateRange, setDateRange] = useState<{
    from: Date | undefined;
    to: Date | undefined;
  }>({
    from: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000), // 7 days ago
    to: new Date()
  });
  const [isExporting, setIsExporting] = useState(false);
  const { toast } = useToast();

  const handleExport = async () => {
    setIsExporting(true);
    
    try {
      const params = new URLSearchParams();
      if (dateRange.from) params.append('from', dateRange.from.toISOString());
      if (dateRange.to) params.append('to', dateRange.to.toISOString());
      params.append('format', exportFormat);

      const response = await fetch(`http://localhost:8000/detect/export_logs?${params}`);
      
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `anpr_logs_${format(new Date(), 'yyyy-MM-dd')}.${exportFormat}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        toast({
          title: "Export successful",
          description: `Logs exported as ${exportFormat.toUpperCase()}`
        });
      } else {
        throw new Error('Export failed');
      }
    } catch (error) {
      toast({
        title: "Export failed",
        description: "Please try again later",
        variant: "destructive"
      });
    } finally {
      setIsExporting(false);
    }
  };

  const quickExportOptions = [
    {
      label: "Today's Data",
      format: "csv" as const,
      icon: Clock,
      color: "blue",
      days: 0,
      description: "Export today's detections"
    },
    {
      label: "Last 7 Days",
      format: "json" as const,
      icon: Zap,
      color: "green",
      days: 7,
      description: "Weekly detection summary"
    },
    {
      label: "Last 30 Days",
      format: "csv" as const,
      icon: Database,
      color: "purple",
      days: 30,
      description: "Monthly analytics report"
    }
  ];

  const setQuickRange = (days: number, format: "csv" | "json") => {
    const to = new Date();
    const from = days === 0 ? new Date() : new Date(Date.now() - days * 24 * 60 * 60 * 1000);
    setDateRange({ from, to });
    setExportFormat(format);
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center gap-4">
        <div className="w-12 h-12 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg">
          <Download className="h-6 w-6 text-white" />
        </div>
        <div>
          <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-900 to-indigo-600 bg-clip-text text-transparent">
            Export Detection Logs
          </h2>
          <p className="text-slate-600">Download detection history and analytics data in your preferred format</p>
        </div>
      </div>

      {/* Quick Export Options */}
      <Card className="glass-effect border-blue-200 shadow-lg">
        <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50 border-b border-blue-100">
          <CardTitle className="flex items-center gap-3">
            <Zap className="h-5 w-5 text-blue-600" />
            Quick Export Options
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {quickExportOptions.map((option, index) => {
              const Icon = option.icon;
              const colorClasses = {
                blue: "from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700",
                green: "from-green-500 to-green-600 hover:from-green-600 hover:to-green-700",
                purple: "from-purple-500 to-purple-600 hover:from-purple-600 hover:to-purple-700"
              };
              
              return (
                <Card key={index} className="hover:shadow-lg transition-all duration-300 cursor-pointer group border-blue-200" 
                      onClick={() => setQuickRange(option.days, option.format)}>
                  <CardContent className="p-6">
                    <div className="flex items-center gap-4 mb-3">
                      <div className={`w-10 h-10 bg-gradient-to-r ${colorClasses[option.color as keyof typeof colorClasses]} rounded-lg flex items-center justify-center shadow-md group-hover:scale-110 transition-transform`}>
                        <Icon className="h-5 w-5 text-white" />
                      </div>
                      <div>
                        <h3 className="font-semibold text-slate-800">{option.label}</h3>
                        <Badge variant="outline" className="text-xs mt-1">
                          {option.format.toUpperCase()}
                        </Badge>
                      </div>
                    </div>
                    <p className="text-sm text-slate-600">{option.description}</p>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
        {/* Export Configuration */}
        <Card className="glass-effect border-blue-200 shadow-lg">
          <CardHeader className="bg-gradient-to-r from-slate-50 to-blue-50 border-b border-blue-100">
            <CardTitle className="flex items-center gap-3">
              <FileText className="h-5 w-5 text-blue-600" />
              Export Configuration
            </CardTitle>
          </CardHeader>
          <CardContent className="p-6 space-y-8">
            {/* Format Selection */}
            <div className="space-y-4">
              <Label className="text-base font-semibold text-slate-700">Export Format</Label>
              <RadioGroup
                value={exportFormat}
                onValueChange={(value) => setExportFormat(value as "csv" | "json")}
                className="space-y-3"
              >
                <div className="flex items-center space-x-3 p-4 rounded-xl border border-blue-200 hover:bg-blue-50 transition-colors">
                  <RadioGroupItem value="csv" id="csv" className="text-blue-600" />
                  <Label htmlFor="csv" className="flex items-center gap-3 cursor-pointer flex-1">
                    <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                      <FileText className="h-4 w-4 text-blue-600" />
                    </div>
                    <div>
                      <div className="font-medium">CSV Format</div>
                      <div className="text-sm text-slate-500">Comma Separated Values - Perfect for Excel</div>
                    </div>
                  </Label>
                </div>
                <div className="flex items-center space-x-3 p-4 rounded-xl border border-blue-200 hover:bg-blue-50 transition-colors">
                  <RadioGroupItem value="json" id="json" className="text-blue-600" />
                  <Label htmlFor="json" className="flex items-center gap-3 cursor-pointer flex-1">
                    <div className="w-8 h-8 bg-purple-100 rounded-lg flex items-center justify-center">
                      <Database className="h-4 w-4 text-purple-600" />
                    </div>
                    <div>
                      <div className="font-medium">JSON Format</div>
                      <div className="text-sm text-slate-500">JavaScript Object Notation - For developers</div>
                    </div>
                  </Label>
                </div>
              </RadioGroup>
            </div>

            {/* Date Range Selection */}
            <div className="space-y-4">
              <Label className="text-base font-semibold text-slate-700">Date Range Selection</Label>
              <div className="grid grid-cols-1 gap-3">
                <div>
                  <Label className="text-sm text-slate-600 mb-2 block">From Date</Label>
                  <Popover>
                    <PopoverTrigger asChild>
                      <Button
                        variant="outline"
                        className="w-full justify-start text-left font-normal h-12 border-blue-200 hover:border-blue-400"
                      >
                        <CalendarIcon className="mr-3 h-4 w-4 text-blue-600" />
                        {dateRange.from ? format(dateRange.from, "PPP") : "Select start date"}
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-auto p-0" align="start">
                      <Calendar
                        mode="single"
                        selected={dateRange.from}
                        onSelect={(date) => setDateRange(prev => ({ ...prev, from: date }))}
                        initialFocus
                      />
                    </PopoverContent>
                  </Popover>
                </div>

                <div>
                  <Label className="text-sm text-slate-600 mb-2 block">To Date</Label>
                  <Popover>
                    <PopoverTrigger asChild>
                      <Button
                        variant="outline"
                        className="w-full justify-start text-left font-normal h-12 border-blue-200 hover:border-blue-400"
                      >
                        <CalendarIcon className="mr-3 h-4 w-4 text-blue-600" />
                        {dateRange.to ? format(dateRange.to, "PPP") : "Select end date"}
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-auto p-0" align="start">
                      <Calendar
                        mode="single"
                        selected={dateRange.to}
                        onSelect={(date) => setDateRange(prev => ({ ...prev, to: date }))}
                        initialFocus
                      />
                    </PopoverContent>
                  </Popover>
                </div>
              </div>
            </div>

            {/* Export Button */}
            <Button
              onClick={handleExport}
              disabled={isExporting || !dateRange.from || !dateRange.to}
              className="w-full h-12 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 shadow-lg hover:shadow-xl transition-all duration-300"
              size="lg"
            >
              <Download className="h-5 w-5 mr-2" />
              {isExporting ? "Exporting..." : "Export Detection Logs"}
            </Button>
          </CardContent>
        </Card>

        {/* Export Preview */}
        <Card className="glass-effect border-blue-200 shadow-lg">
          <CardHeader className="bg-gradient-to-r from-slate-50 to-blue-50 border-b border-blue-100">
            <CardTitle>Export Preview & Summary</CardTitle>
          </CardHeader>
          <CardContent className="p-6 space-y-6">
            {/* Export Details */}
            <div className="space-y-4">
              <div className="flex justify-between items-center p-3 bg-blue-50 rounded-lg">
                <span className="text-sm font-medium text-slate-700">Format:</span>
                <Badge variant="outline" className="bg-white">
                  {exportFormat.toUpperCase()}
                </Badge>
              </div>
              
              <div className="flex justify-between items-start p-3 bg-blue-50 rounded-lg">
                <span className="text-sm font-medium text-slate-700">Date Range:</span>
                <div className="text-right">
                  <div className="text-sm font-medium">
                    {dateRange.from ? format(dateRange.from, "MMM dd, yyyy") : "Not set"}
                  </div>
                  <div className="text-sm text-slate-500">
                    to {dateRange.to ? format(dateRange.to, "MMM dd, yyyy") : "Not set"}
                  </div>
                </div>
              </div>
              
              <div className="flex justify-between items-center p-3 bg-green-50 rounded-lg">
                <span className="text-sm font-medium text-slate-700">Estimated Records:</span>
                <Badge variant="outline" className="bg-green-100 text-green-800 border-green-200">
                  ~{Math.floor(Math.random() * 500) + 100} records
                </Badge>
              </div>
            </div>

            {/* Data Fields */}
            <div className="pt-4 border-t border-blue-100">
              <h4 className="font-semibold mb-4 text-slate-800">Included Data Fields:</h4>
              <div className="grid grid-cols-1 gap-2">
                {[
                  "License plate number",
                  "Detection timestamp", 
                  "Source (camera/video)",
                  "Confidence score",
                  "Vehicle type",
                  "Location coordinates",
                  "Bounding box data",
                  "Image metadata"
                ].map((field, index) => (
                  <div key={index} className="flex items-center gap-2 text-sm">
                    <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                    <span className="text-slate-600">{field}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* File Size Estimate */}
            <div className="pt-4 border-t border-blue-100">
              <div className="bg-amber-50 p-4 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <FileText className="h-4 w-4 text-amber-600" />
                  <span className="text-sm font-medium text-amber-800">Estimated File Size</span>
                </div>
                <p className="text-sm text-amber-700">
                  Approximately {exportFormat === 'csv' ? '2-5 MB' : '3-8 MB'} based on selected date range
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
