
import React from "react";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Activity, RefreshCw, Hash, Tag, Clock, Star, Wrench } from "lucide-react";
import JobCard from "@/components/features/jobs/JobCard";
import { Button } from "@/components/ui/button";
import { Job } from "@/types/job";

interface JobsSectionProps {
  jobs: Job[];
  loadingJobs: boolean;
  onManualRefresh: () => void;
  onView: (job: Job) => void;
}

const JobsSection: React.FC<JobsSectionProps> = ({
  jobs,
  loadingJobs,
  onManualRefresh,
  onView
}) => {
  const sortedJobs = React.useMemo(() =>
    [...jobs].sort((a, b) => b.id - a.id),
  [jobs]);

  return (
  <Card className="glass-intense backdrop-blur-3xl border border-white/15 shadow-2xl rounded-4xl overflow-hidden">
    <CardHeader className="bg-gradient-to-r from-white/5 to-white/2 border-b border-white/10 py-6">
      <div className="flex justify-between items-center">
        <div>
          <CardTitle className="text-2xl font-bold text-white flex items-center tracking-tight mb-1">
            <Activity className="w-6 h-6 mr-3 text-white/80" />
            Processing Jobs
          </CardTitle>
          <CardDescription className="text-white/60 text-base">
            Track the status of your submitted videos
          </CardDescription>
        </div>
        <Button
          onClick={onManualRefresh}
          disabled={loadingJobs}
          variant="outline"
          className="border-blue-500/30 bg-white/10 backdrop-blur-xl text-white hover:bg-blue-500/20 hover:text-white transition-all duration-300 font-semibold rounded-2xl px-6 py-2 shadow-lg"
        >
          {loadingJobs ? (
            <Activity className="w-4 h-4 mr-2 animate-spin" />
          ) : (
            <RefreshCw className="w-4 h-4 mr-2" />
          )}
          {loadingJobs ? 'Loading...' : 'Refresh'}
        </Button>
      </div>
    </CardHeader>
    <CardContent className="p-0">
      {sortedJobs.length === 0 && !loadingJobs ? (
        <div className="py-12 px-6 text-center">
          <p className="text-white/60 text-lg">No jobs submitted yet.</p>
          <p className="text-white/50 text-sm mt-2">Upload a video above to get started!</p>
        </div>
      ) : (
        <div className="flex flex-col">
          {/* Table Header */}
          <div className="grid grid-cols-12 items-center gap-6 px-6 py-4 bg-white/5 backdrop-blur-sm border-b border-white/10 text-white/70 text-sm font-semibold">
            <div className="col-span-1 flex items-center">
              <Hash className="w-4 h-4 mr-2" />
              Job ID
            </div>
            <div className="col-span-3 flex items-center">
              <Tag className="w-4 h-4 mr-2" />
              Job Type
            </div>
            <div className="col-span-2 flex items-center">
              <Clock className="w-4 h-4 mr-2" />
              Submitted
            </div>
            <div className="col-span-3 flex justify-center items-center">
              <Star className="w-4 h-4 mr-2" />
              Status
            </div>
            <div className="col-span-3 flex justify-center items-center">
              <Wrench className="w-4 h-4 mr-2" />
              Actions
            </div>
          </div>

          {/* Job Rows */}
          {sortedJobs.map((job) => (
            <JobCard key={job.id} job={job} onView={onView} />
          ))}
        </div>
      )}
    </CardContent>
  </Card>
  );
};

export default JobsSection;
