import React from 'react';
import { Button } from '@/components/ui/button';
import { Download } from 'lucide-react';
import { toast } from '@/components/ui/use-toast';

interface CSVDownloadButtonProps {
  csvPath: string;
  className?: string;
}

const CSVDownloadButton: React.FC<CSVDownloadButtonProps> = ({ csvPath, className }) => {
  const handleDownload = () => {
    const ensureAbsoluteUrl = (path: string) => {
      if (path.startsWith('/media/')) {
        return `https://mx-smart-persons-expertise.trycloudflare.com${path}`;
      }
      return path;
    };

    const fullUrl = ensureAbsoluteUrl(csvPath);
    window.open(fullUrl, '_blank', 'noopener,noreferrer');

    toast({
      title: "CSV Report Download",
      description: "The CSV report is opening in a new tab. Right-click and select 'Save As' to download.",
    });
  };

  return (
    <Button
      onClick={handleDownload}
      variant="outline"
      size="sm"
      className={className}
    >
      <Download className="w-4 h-4 mr-2" />
      Download CSV Report
    </Button>
  );
};

export default CSVDownloadButton;
