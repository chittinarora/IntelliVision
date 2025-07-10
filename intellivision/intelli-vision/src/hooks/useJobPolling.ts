
import { useState, useEffect, useCallback } from 'react';
import { Job } from '@/types/job';
import { authFetch } from '@/utils/authFetch';
import { toast } from '@/hooks/use-toast';

interface UseJobPollingOptions {
  jobId: number | null;
  onJobComplete?: (job: Job) => void;
  pollInterval?: number;
}

export const useJobPolling = ({ 
  jobId, 
  onJobComplete, 
  pollInterval = 3000 
}: UseJobPollingOptions) => {
  const [job, setJob] = useState<Job | null>(null);
  const [isPolling, setIsPolling] = useState(false);

  const pollJob = useCallback(async (id: number) => {
    try {
      const response = await authFetch(`/jobs/${id}/`, {
        method: 'GET',
        requireAuth: true,
      });

      if (response.ok) {
        const jobData: Job = await response.json();
        setJob(jobData);

        if (jobData.status === 'done') {
          setIsPolling(false);
          onJobComplete?.(jobData);
          
          // Show success notification
          toast({
            title: "Analysis Complete!",
            description: "Your video has been processed successfully.",
          });
        } else if (jobData.status === 'failed') {
          setIsPolling(false);
          toast({
            title: "Analysis Failed",
            description: "There was an error processing your video.",
            variant: "destructive",
          });
        }
      } else {
        console.error('Failed to fetch job status:', response.status);
      }
    } catch (error) {
      console.error('Error polling job:', error);
    }
  }, [onJobComplete]);

  useEffect(() => {
    if (!jobId || !isPolling) return;

    const intervalId = setInterval(() => {
      pollJob(jobId);
    }, pollInterval);

    // Poll immediately
    pollJob(jobId);

    return () => clearInterval(intervalId);
  }, [jobId, isPolling, pollInterval, pollJob]);

  const startPolling = useCallback((id: number) => {
    setIsPolling(true);
    setJob(null);
  }, []);

  const stopPolling = useCallback(() => {
    setIsPolling(false);
  }, []);

  return {
    job,
    isPolling,
    startPolling,
    stopPolling,
  };
};
