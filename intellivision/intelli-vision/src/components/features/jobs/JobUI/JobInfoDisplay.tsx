
import React, { useMemo } from 'react';
import { File, Calendar, Clock, Users } from 'lucide-react';
import { formatJobDate, formatDuration, getJobFilename } from '@/lib/jobUtils';

interface JobInfoDisplayProps {
  inputVideo: string;
  createdAt: string;
  duration?: number | null;
  peopleCount?: number | null;
}

const JobInfoDisplay: React.FC<JobInfoDisplayProps> = ({
  inputVideo,
  createdAt,
  duration,
  peopleCount
}) => {
  const filename = useMemo(() => getJobFilename(inputVideo), [inputVideo]);
  const formattedDate = useMemo(() => formatJobDate(createdAt), [createdAt]);

  return (
    <div className="bg-white/5 backdrop-blur-sm rounded-3xl border border-white/10 p-5 flex items-center gap-x-5 overflow-hidden">
      <File className="w-8 h-8 text-cyan-400 flex-shrink-0" />
      <div className="overflow-hidden flex-1">
        <p className="font-semibold text-white truncate" title={filename}>
          {filename}
        </p>
        <div className="flex items-center gap-x-4 text-sm text-white/70 mt-1">
          <span className="flex items-center">
            <Calendar className="w-4 h-4 mr-1.5" />
            {formattedDate}
          </span>
          {duration && (
            <>
              <span className="text-white/30">&middot;</span>
              <span className="flex items-center">
                <Clock className="w-4 h-4 mr-1.5" />
                {formatDuration(duration)}
              </span>
            </>
          )}
          {peopleCount !== null && peopleCount !== undefined && (
            <>
              <span className="text-white/30">&middot;</span>
              <span className="flex items-center">
                <Users className="w-4 h-4 mr-1.5" />
                {peopleCount} people
              </span>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default JobInfoDisplay;
