import React, { useState } from "react";
import { useJobs } from "@/hooks/useJobs";
import JobsSection from "@/components/features/jobs/JobsSection";
import JobModal from "@/components/features/jobs/JobModal";
import AnimatedBackground from "@/components/features/dashboard/AnimatedBackground";
import WelcomeSection from "@/components/features/dashboard/WelcomeSection";
import TaskSelector from "@/components/features/dashboard/TaskSelector";
import StatsOverview from "@/components/features/dashboard/StatsOverview";
import { Job } from "@/types/job";
import { authFetch } from "@/utils/authFetch";
import { useEffect } from "react";
import Navbar from "@/components/layout/Navbar";

const Dashboard = () => {
  const { jobs, loadingJobs, handleManualRefresh } = useJobs();
  const [viewingJob, setViewingJob] = useState<Job | null>(null);
  const [username, setUsername] = useState<string | null>(null);

  const closeJobModal = () => setViewingJob(null);

  // Fetch logged-in user info from backend using authFetch
  useEffect(() => {
    const fetchUser = async () => {
      try {
        const res = await authFetch("/me/");
        if (res.ok) {
          const data = await res.json();
          setUsername(data.username);
        } else {
          console.warn("Failed to fetch user info:", res.status);
        }
      } catch (error) {
        console.error("Error fetching username:", error);
      }
    };

    fetchUser();
  }, []);

  return (
    <div className="min-h-screen bg-navy-gradient relative overflow-hidden page-enter page-enter-active">
      <AnimatedBackground />
      <Navbar mode="dashboard" />

      <div className="pt-32 max-w-8xl mx-auto px-4 sm:px-6 lg:px-20 py-8">
        <WelcomeSection username={username} />
        <TaskSelector />
        <StatsOverview jobs={jobs} />

        {/* Jobs Section */}
        <JobsSection
          jobs={jobs}
          loadingJobs={loadingJobs}
          onManualRefresh={handleManualRefresh}
          onView={setViewingJob}
        />
      </div>
      <JobModal job={viewingJob} onClose={closeJobModal} />
    </div>
  );
};

export default Dashboard;
