
import { useState, useEffect, useRef, useCallback } from 'react';
import { toast } from "@/components/ui/use-toast";
import { Job } from "@/types/job";
import { authFetch } from "@/utils/authFetch";

export const useJobs = () => {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loadingJobs, setLoadingJobs] = useState(false);
  const isFetching = useRef(false);

  const fetchJobs = useCallback(async (isManualRefresh = false) => {
    if (isFetching.current && !isManualRefresh) {
      return;
    }
    isFetching.current = true;
    if (isManualRefresh) {
      setLoadingJobs(true);
    }
    
    console.log("Fetching jobs from: /jobs/");
    try {
      const response = await authFetch('/jobs/', {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        },
      });

      console.log("Fetch jobs response status:", response.status);

      if (response.ok) {
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
          const jobsData = await response.json();
          
          setJobs(jobsData);
          console.log("Jobs fetched successfully:", jobsData);
          if (isManualRefresh) {
            toast({
              title: "Jobs refreshed",
              description: "The job list has been updated.",
            });
          }
        } else {
          const textResponse = await response.text();
          console.error("Fetch jobs error: Unexpected content type", contentType, textResponse);
          if (isManualRefresh) {
            toast({
              title: "Server Error",
              description: `Received unexpected response format from server. Content-Type: ${contentType || 'N/A'}`,
              variant: "destructive",
            });
          }
        }
      } else {
        const errorText = await response.text();
        console.error("Failed to fetch jobs, status:", response.status, errorText);
        if (isManualRefresh) {
          toast({
            title: "Failed to fetch jobs",
            description: `Server returned status ${response.status}. ${errorText ? `Details: ${errorText.substring(0,100)}` : 'Please try again.'}`,
            variant: "destructive",
          });
        }
      }
    } catch (error) {
      console.error("Network error fetching jobs:", error);
      if (isManualRefresh) {
        let errorMessage = "Could not fetch job list from server.";
        if (error instanceof Error) {
          errorMessage += ` Message: ${error.message}`;
        }
        toast({
          title: "Network Error",
          description: errorMessage,
          variant: "destructive",
        });
      }
    } finally {
      if (isManualRefresh) {
        setLoadingJobs(false);
      }
      isFetching.current = false;
    }
  }, []);

  useEffect(() => {
    fetchJobs(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const shouldPoll = jobs.some(job => job.status === 'pending' || job.status === 'processing');

    if (shouldPoll) {
      const intervalId = setInterval(() => {
        console.log("Polling for job updates...");
        fetchJobs(false);
      }, 5000);

      return () => clearInterval(intervalId);
    }
  }, [jobs, fetchJobs]);

  const handleManualRefresh = () => {
    fetchJobs(true);
  };

  return {
    jobs,
    loadingJobs,
    fetchJobs,
    handleManualRefresh
  };
};
