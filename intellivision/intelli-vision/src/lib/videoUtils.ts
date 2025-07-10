
import { ensureHttpsUrl } from "@/lib/utils";
import { getWebFriendlyMediaUrl } from "@/lib/mediaFormatUtils";
import { toast } from "@/components/ui/use-toast";

export const downloadVideo = async (outputVideoUrl?: string | null) => {
  if (!outputVideoUrl) {
    toast({
      title: "Download Unavailable",
      description: "No video link found for this job.",
      variant: "destructive",
    });
    return;
  }
  
  const httpsUrl = ensureHttpsUrl(outputVideoUrl);
  if (!httpsUrl) {
    toast({
      title: "Invalid download URL",
      description: "No valid video link found for this job.",
      variant: "destructive",
    });
    return;
  }
  
  try {
    // Convert to web-friendly format before download
    const webFriendlyUrl = await getWebFriendlyMediaUrl(httpsUrl, true);
    
    window.open(webFriendlyUrl, '_blank', 'noopener,noreferrer');
    toast({
      title: "How to download",
      description: "Right-click the video in the new tab and choose 'Save As' to download.",
    });
  } catch (error) {
    console.error('Failed to prepare video for download:', error);
    // Fallback to original URL
    window.open(httpsUrl, '_blank', 'noopener,noreferrer');
    toast({
      title: "How to download",
      description: "Right-click the video in the new tab and choose 'Save As' to download.",
    });
  }
};
