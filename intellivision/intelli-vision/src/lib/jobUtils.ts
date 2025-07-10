
export const formatJobDate = (dateString: string): string => {
  const jobDate = new Date(dateString);
  const today = new Date();
  
  const isToday = jobDate.getFullYear() === today.getFullYear() &&
                  jobDate.getMonth() === today.getMonth() &&
                  jobDate.getDate() === today.getDate();

  if (isToday) {
    return `Today at ${jobDate.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}`;
  } else {
    return jobDate.toLocaleDateString();
  }
};

export const formatDuration = (seconds: number): string => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);

  if (mins === 0) return `${secs} second${secs !== 1 ? 's' : ''}`;
  if (secs === 0) return `${mins} minute${mins !== 1 ? 's' : ''}`;
  return `${mins} minute${mins !== 1 ? 's' : ''} and ${secs} second${secs !== 1 ? 's' : ''}`;
};

export const getJobFilename = (inputVideo: string): string => {
  return inputVideo.split('/').pop() || 'N/A';
};
