
import { useState, useEffect, useCallback } from 'react';
import { authFetch } from '@/utils/authFetch';
import { toast } from '@/hooks/use-toast';

interface EmergencyJobResult {
  job_id: number;
  status: 'pending' | 'processing' | 'done' | 'failed';
  results?: {
    unique_people: number;
    total_crossings: number;
    final_counts: {
      line1: { in: number; out: number };
      line2: { in: number; out: number };
    };
    output_path: string;
  };
}

interface UseEmergencyJobPollingOptions {
  jobId: number | null;
  onJobComplete?: (job: EmergencyJobResult) => void;
  pollInterval?: number;
}

export const useEmergencyJobPolling = ({ 
  jobId, 
  onJobComplete, 
  pollInterval = 3000 
}: UseEmergencyJobPollingOptions) => {
  const [job, setJob] = useState<EmergencyJobResult | null>(null);
  const [isPolling, setIsPolling] = useState(false);

  const pollJob = useCallback(async (id: number) => {
    try {
      const response = await authFetch(`/emergency-count/${id}/`, {
        method: 'GET',
        requireAuth: true,
      });

      if (response.ok) {
        const jobData: EmergencyJobResult = await response.json();
        setJob(jobData);

        if (jobData.status === 'done') {
          setIsPolling(false);
          onJobComplete?.(jobData);
          
          toast({
            title: "Analysis Complete!",
            description: "Your emergency counting analysis has been processed successfully.",
          });
        } else if (jobData.status === 'failed') {
          setIsPolling(false);
          toast({
            title: "Analysis Failed",
            description: "There was an error processing your emergency count video.",
            variant: "destructive",
          });
        }
      } else {
        console.error('Failed to fetch emergency job status:', response.status);
      }
    } catch (error) {
      console.error('Error polling emergency job:', error);
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
