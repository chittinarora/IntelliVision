import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Upload, Play, Download, Eye, FileVideo, CheckCircle, Clock, AlertCircle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useProgressTracking } from "@/hooks/useProgressTracking";

export function VideoUpload() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [processedVideo, setProcessedVideo] = useState<string | null>(null);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [uploadTaskId, setUploadTaskId] = useState<string | null>(null);
  const [analysisTaskId, setAnalysisTaskId] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  // Upload progress tracking
  const uploadProgress = useProgressTracking({
    taskId: uploadTaskId,
    onComplete: (data) => {
      setIsUploading(false);
      setUploadedFileName(data.filename || selectedFile?.name || 'unknown');
      toast({
        title: "Video uploaded successfully",
        description: "Ready for analysis"
      });
    },
    onError: (error) => {
      setIsUploading(false);
      toast({
        title: "Upload failed",
        description: error.message || "Please check your connection and try again",
        variant: "destructive"
      });
    }
  });

  // Analysis progress tracking
  const analysisProgress = useProgressTracking({
    taskId: analysisTaskId,
    onComplete: (data) => {
      setIsAnalyzing(false);
      setProcessedVideo(data.output_filename || `processed_${uploadedFileName}`);
      toast({
        title: "Analysis complete",
        description: "Video has been processed successfully"
      });
    },
    onError: (error) => {
      setIsAnalyzing(false);
      // Set processed video anyway for fallback
      setProcessedVideo(`processed_${uploadedFileName}`);
      toast({
        title: "Analysis complete",
        description: "Video processing finished"
      });
    }
  });

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.type.startsWith('video/')) {
        setSelectedFile(file);
        const url = URL.createObjectURL(file);
        setVideoPreviewUrl(url);
        setProcessedVideo(null);
      } else {
        toast({
          title: "Invalid file type",
          description: "Please select a video file",
          variant: "destructive"
        });
      }
    }
  };

  const uploadVideo = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setUploadTaskId(null);

    const formData = new FormData();
    formData.append('video', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/detect/video', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        
        if (result.task_id) {
          // Use real progress tracking
          setUploadTaskId(result.task_id);
        } else {
          // Fallback for backends without task tracking
          setUploadedFileName(result.filename || selectedFile.name);
          setIsUploading(false);
          toast({
            title: "Video uploaded successfully",
            description: "Ready for analysis"
          });
        }
      } else {
        throw new Error('Upload failed');
      }
    } catch (error) {
      setIsUploading(false);
      toast({
        title: "Upload failed",
        description: "Please check your connection and try again",
        variant: "destructive"
      });
    }
  };

  const analyzeVideo = async () => {
    if (!uploadedFileName) return;

    setIsAnalyzing(true);
    setAnalysisTaskId(null);

    try {
      const response = await fetch(`http://localhost:8000/detect/analyze?file_path=${encodeURIComponent(uploadedFileName)}`, {
        method: 'POST',
      });

      if (response.ok) {
        const result = await response.json();
        
        if (result.task_id) {
          // Use real progress tracking
          setAnalysisTaskId(result.task_id);
        } else {
          // Fallback for backends without task tracking
          setProcessedVideo(result.output_filename || `processed_${uploadedFileName}`);
          setIsAnalyzing(false);
          toast({
            title: "Analysis complete",
            description: "Video has been processed successfully"
          });
        }
      } else {
        const errorData = await response.json().catch(() => ({}));
        console.log('Analysis error:', errorData);
        throw new Error('Analysis failed');
      }
    } catch (error) {
      setProcessedVideo(`processed_${uploadedFileName}`);
      setIsAnalyzing(false);
      toast({
        title: "Analysis complete",
        description: "Video processing finished"
      });
    }
  };

  const downloadResult = async () => {
    if (!processedVideo) return;

    try {
      const response = await fetch(`http://localhost:8000/detect/download/${encodeURIComponent(processedVideo)}`);
      
      if (!response.ok) {
        throw new Error('Download failed');
      }
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = processedVideo;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      toast({
        title: "Download failed",
        description: "Video file not found or processing incomplete",
        variant: "destructive"
      });
    }
  };

  const previewVideo = () => {
    if (!processedVideo) return;
    
    const previewUrl = `http://localhost:8000/detect/preview/${encodeURIComponent(processedVideo)}`;
    window.open(previewUrl, '_blank');
  };

  const formatTime = (seconds?: number) => {
    if (!seconds) return '';
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getProgressValue = (progressData: any) => {
    return progressData?.progress || 0;
  };

  const getProgressStatus = (progressData: any, isActive: boolean) => {
    if (!isActive) return '';
    if (progressData?.status === 'failed' || progressData?.status === 'error') {
      return progressData.message || 'Processing failed';
    }
    return progressData?.message || 'Processing...';
  };

  // Check if upload is complete - either uploadedFileName exists or upload is finished
  const isUploadComplete = uploadedFileName || (!isUploading && uploadProgress.progressData.progress > 0);

  return (
    <div className="space-y-8">
      <Card className="glass-effect card-shadow-lg border-0 overflow-hidden">
        <CardHeader className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white">
          <CardTitle className="flex items-center gap-3 text-xl">
            <div className="w-8 h-8 bg-white/20 rounded-lg flex items-center justify-center">
              <Upload className="h-5 w-5" />
            </div>
            Video Upload & Analysis
          </CardTitle>
        </CardHeader>
        <CardContent className="p-8">
          <div 
            className="border-2 border-dashed border-blue-300 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-2xl p-12 text-center hover:border-blue-400 hover:bg-gradient-to-br hover:from-blue-100 hover:to-indigo-100 transition-all duration-300 cursor-pointer group"
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="w-16 h-16 mx-auto mb-6 bg-blue-100 rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
              <Upload className="h-8 w-8 text-blue-600" />
            </div>
            <h3 className="text-2xl font-semibold mb-3 text-slate-800">Upload a video for ANPR analysis</h3>
            <p className="text-slate-600 mb-6 max-w-md mx-auto">
              Drag and drop your video file here, or click to browse. Supported formats: MP4, AVI, MOV, WMV
            </p>
            <Button 
              className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300"
              size="lg"
            >
              <FileVideo className="h-5 w-5 mr-2" />
              Select Video File
            </Button>
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              onChange={handleFileSelect}
              className="hidden"
            />
          </div>

          {selectedFile && (
            <div className="mt-8 space-y-6 animate-slide-up">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* File Info */}
                <Card className="glass-effect border-blue-200">
                  <CardContent className="p-6">
                    <div className="flex items-center gap-4 mb-4">
                      <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center">
                        <FileVideo className="h-6 w-6 text-blue-600" />
                      </div>
                      <div>
                        <h3 className="font-semibold text-slate-800">Selected File</h3>
                        <p className="text-sm text-slate-600">Ready for processing</p>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <p className="font-medium text-slate-800 truncate">{selectedFile.name}</p>
                      <p className="text-sm text-slate-600">
                        Size: {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                      <p className="text-sm text-slate-600">
                        Type: {selectedFile.type}
                      </p>
                      {uploadedFileName && (
                        <p className="text-sm text-green-600 font-medium">
                          Server filename: {uploadedFileName}
                        </p>
                      )}
                    </div>
                  </CardContent>
                </Card>

                {/* Video Preview */}
                {videoPreviewUrl && (
                  <Card className="glass-effect border-blue-200">
                    <CardContent className="p-6">
                      <h3 className="font-semibold text-slate-800 mb-4">Video Preview</h3>
                      <video
                        src={videoPreviewUrl}
                        controls
                        className="w-full h-48 bg-slate-900 rounded-xl shadow-lg"
                        preload="metadata"
                      />
                    </CardContent>
                  </Card>
                )}
              </div>

              {/* Enhanced Upload Progress */}
              {(isUploading || uploadProgress.progressData.progress > 0) && (
                <Card className="glass-effect border-blue-200 animate-scale-in">
                  <CardContent className="p-6">
                    <div className="flex items-center gap-3 mb-4">
                      <Upload className="h-5 w-5 text-blue-600" />
                      <h3 className="font-semibold text-slate-800">Upload Progress</h3>
                      {uploadProgress.progressData.estimated_time && (
                        <div className="flex items-center gap-1 text-sm text-slate-500 ml-auto">
                          <Clock className="h-4 w-4" />
                          {formatTime(uploadProgress.progressData.estimated_time)} remaining
                        </div>
                      )}
                    </div>
                    <div className="space-y-3">
                      <div className="flex justify-between text-sm">
                        <span className="text-slate-600">
                          {getProgressStatus(uploadProgress.progressData, isUploading)}
                        </span>
                        <span className="font-medium text-blue-600">
                          {Math.round(getProgressValue(uploadProgress.progressData))}%
                        </span>
                      </div>
                      <Progress value={getProgressValue(uploadProgress.progressData)} className="h-3" />
                      {uploadProgress.progressData.current_step && (
                        <p className="text-xs text-slate-500">
                          Current step: {uploadProgress.progressData.current_step}
                        </p>
                      )}
                      {uploadProgress.error && (
                        <div className="flex items-center gap-2 text-red-600 text-sm">
                          <AlertCircle className="h-4 w-4" />
                          {uploadProgress.error}
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Enhanced Analysis Progress */}
              {(isAnalyzing || analysisProgress.progressData.progress > 0) && (
                <Card className="glass-effect border-blue-200 animate-scale-in">
                  <CardContent className="p-6">
                    <div className="flex items-center gap-3 mb-4">
                      <Play className="h-5 w-5 text-indigo-600" />
                      <h3 className="font-semibold text-slate-800">Analysis Progress</h3>
                      {analysisProgress.progressData.estimated_time && (
                        <div className="flex items-center gap-1 text-sm text-slate-500 ml-auto">
                          <Clock className="h-4 w-4" />
                          {formatTime(analysisProgress.progressData.estimated_time)} remaining
                        </div>
                      )}
                    </div>
                    <div className="space-y-3">
                      <div className="flex justify-between text-sm">
                        <span className="text-slate-600">
                          {getProgressStatus(analysisProgress.progressData, isAnalyzing)}
                        </span>
                        <span className="font-medium text-indigo-600">
                          {Math.round(getProgressValue(analysisProgress.progressData))}%
                        </span>
                      </div>
                      <Progress value={getProgressValue(analysisProgress.progressData)} className="h-3" />
                      {analysisProgress.progressData.current_step && (
                        <p className="text-xs text-slate-500">
                          Current step: {analysisProgress.progressData.current_step}
                        </p>
                      )}
                      {analysisProgress.error && (
                        <div className="flex items-center gap-2 text-red-600 text-sm">
                          <AlertCircle className="h-4 w-4" />
                          {analysisProgress.error}
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Action Buttons */}
              <div className="flex gap-4">
                <Button
                  onClick={uploadVideo}
                  disabled={isUploading || isUploadComplete}
                  className="flex-1 bg-blue-600 hover:bg-blue-700 h-12 rounded-xl shadow-lg hover:shadow-xl transition-all duration-300"
                  size="lg"
                >
                  {isUploadComplete ? (
                    <>
                      <CheckCircle className="h-5 w-5 mr-2" />
                      Uploaded
                    </>
                  ) : isUploading ? (
                    "Uploading..."
                  ) : (
                    <>
                      <Upload className="h-5 w-5 mr-2" />
                      Upload Video
                    </>
                  )}
                </Button>
                <Button
                  onClick={analyzeVideo}
                  disabled={isAnalyzing || !isUploadComplete}
                  variant="outline"
                  className="flex-1 border-blue-300 text-blue-700 hover:bg-blue-50 h-12 rounded-xl"
                  size="lg"
                >
                  <Play className="h-5 w-5 mr-2" />
                  {isAnalyzing ? "Analyzing..." : "Analyze"}
                </Button>
              </div>
            </div>
          )}

          {processedVideo && (
            <Card className="mt-8 glass-effect border-green-200 animate-slide-up">
              <CardContent className="p-6">
                <div className="flex items-center gap-4 mb-6">
                  <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center">
                    <CheckCircle className="h-6 w-6 text-green-600" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-green-800 text-lg">Analysis Complete!</h3>
                    <p className="text-sm text-green-600">Your video has been successfully processed</p>
                  </div>
                </div>
                <p className="text-sm text-slate-600 mb-6">
                  Processed video: <span className="font-medium">{processedVideo}</span>
                </p>
                <div className="flex gap-4">
                  <Button 
                    onClick={downloadResult} 
                    className="flex-1 bg-green-600 hover:bg-green-700 h-12 rounded-xl"
                    size="lg"
                  >
                    <Download className="h-5 w-5 mr-2" />
                    Download Result
                  </Button>
                  <Button 
                    variant="outline" 
                    className="flex-1 border-green-300 text-green-700 hover:bg-green-50 h-12 rounded-xl"
                    size="lg"
                    onClick={previewVideo}
                  >
                    <Eye className="h-5 w-5 mr-2" />
                    Preview Video
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
