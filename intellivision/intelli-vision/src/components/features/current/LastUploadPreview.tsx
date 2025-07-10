import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Video, Image as ImageIcon } from "lucide-react";
import { Job } from "@/types/job";
import ProcessingView from "@/components/features/jobs/JobViews/ProcessingView";
import FailedView from "@/components/features/jobs/JobViews/FailedView";
import DoneView from "@/components/features/jobs/JobViews/DoneView";

interface LastUploadPreviewProps {
  job: Job;
}

const LastUploadPreview: React.FC<LastUploadPreviewProps> = ({ job }) => {
  console.log("LastUploadPreview job:", job);
  console.log("Job results:", job.results);
  console.log("Job output_video:", job.output_video);
  console.log("Job input_video:", job.input_video);

  const getTitleText = () => {
    if (job.status === "pending" || job.status === "processing")
      return `Job ${job.id} - Processing`;
    if (job.status === "failed") return `Job ${job.id} - Failed`;
    if (job.status === "done") return `Job ${job.id} - Completed`;
    return `Job ${job.id} - ${job.status}`;
  };

  // Determine if this is an image job based on input file type
  const isImageJob =
    job.input_video &&
    (job.input_video.toLowerCase().includes(".jpg") ||
      job.input_video.toLowerCase().includes(".jpeg") ||
      job.input_video.toLowerCase().includes(".png") ||
      job.input_video.toLowerCase().includes(".gif") ||
      job.input_video.toLowerCase().includes(".webp") ||
      job.input_video.toLowerCase().includes(".bmp"));

  console.log("Is image job:", isImageJob, "for file:", job.input_video);

  const getHeaderText = () => {
    const mediaType = isImageJob ? "image" : "video";

    if (job.status === "pending" || job.status === "processing")
      return `Your ${mediaType} is being analyzed.`;
    if (job.status === "failed")
      return `There was an error processing your ${mediaType}.`;
    if (job.status === "done")
      return `Here's the result of your most recent analysis.`;
    return "Here is the status of your latest job.";
  };

  const getIconAndMediaType = () => {
    if (isImageJob) {
      return { icon: ImageIcon, mediaType: "Image" };
    } else {
      return { icon: Video, mediaType: "Video" };
    }
  };

  const renderJobContent = () => {
    switch (job.status) {
      case "pending":
      case "processing":
        return <ProcessingView />;
      case "failed":
        return <FailedView />;
      case "done":
        return <DoneView job={job} />;
      default:
        return <ProcessingView />;
    }
  };

  const { icon: IconComponent, mediaType } = getIconAndMediaType();

  return (
    <div className="fade-in-up">
      <div className="text-center mb-8">
        <h3 className="text-3xl font-bold text-white tracking-tight mb-2">
          Latest Upload Status
        </h3>
        <p className="text-white/70 text-lg">{getHeaderText()}</p>
      </div>
      <Card className="bg-white/5 backdrop-blur-3xl shadow-2xl border border-white/10 rounded-5xl scale-in card-hover transition-all duration-500">
        <CardHeader className="pb-4">
          <CardTitle className="text-white flex items-center text-2xl font-bold tracking-tight">
            <IconComponent className="w-7 h-7 mr-3 text-cyan-400" />
            {getTitleText()}
          </CardTitle>
        </CardHeader>
        <CardContent className="p-6 pt-0">
          <div className="animate-fluid-in">{renderJobContent()}</div>
        </CardContent>
      </Card>
    </div>
  );
};

export default LastUploadPreview;
