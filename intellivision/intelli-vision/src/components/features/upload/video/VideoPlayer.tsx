
import React, { useEffect, useRef, useState } from 'react';
import { getWebFriendlyMediaUrl } from '@/lib/mediaFormatUtils';

interface VideoPlayerProps {
  url: string;
  videoKey: string;
  className?: string;
  autoPlay?: boolean;
}

const VideoPlayer: React.FC<VideoPlayerProps> = React.memo(({ url, videoKey, className = "w-full h-full object-contain", autoPlay = false }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const mountTimeRef = useRef<number>(Date.now());
  const [webFriendlyUrl, setWebFriendlyUrl] = useState<string>(url);
  const [isConverting, setIsConverting] = useState(false);
  
  console.log('VideoPlayer RENDER - Key:', videoKey, 'URL:', url, 'Mount Time:', mountTimeRef.current);
  
  useEffect(() => {
    console.log('VideoPlayer MOUNTED - Key:', videoKey, 'Mount Time:', mountTimeRef.current);
    
    // Store reference to current video element
    const currentVideo = videoRef.current;
    if (currentVideo) {
      console.log('Video element created with src:', currentVideo.src);
    }
    
    return () => {
      console.log('VideoPlayer UNMOUNTED - Key:', videoKey, 'Mount Time:', mountTimeRef.current, 'Lifetime:', Date.now() - mountTimeRef.current, 'ms');
    };
  }, [videoKey]);

  // Convert video to web-friendly format when URL changes
  useEffect(() => {
    console.log('VideoPlayer URL CHANGED:', url, 'Key:', videoKey);
    
    const convertVideo = async () => {
      if (!url) return;
      
      setIsConverting(true);
      try {
        const friendlyUrl = await getWebFriendlyMediaUrl(url, true);
        setWebFriendlyUrl(friendlyUrl);
      } catch (error) {
        console.error('Failed to convert video:', error);
        setWebFriendlyUrl(url); // Fallback to original
      } finally {
        setIsConverting(false);
      }
    };
    
    convertVideo();
  }, [url, videoKey]);

  // Monitor all prop changes
  useEffect(() => {
    console.log('VideoPlayer PROPS CHANGED - className:', className, 'autoPlay:', autoPlay);
  }, [className, autoPlay]);

  if (isConverting) {
    return (
      <div className={`${className} flex items-center justify-center bg-black/20 rounded-3xl border border-white/10`}>
        <div className="text-white/70 text-center">
          <div className="animate-spin w-8 h-8 border-2 border-white/30 border-t-white/70 rounded-full mx-auto mb-2"></div>
          <p>Converting video...</p>
        </div>
      </div>
    );
  }

  return (
    <video 
      ref={videoRef}
      controls 
      muted 
      loop={!autoPlay}
      autoPlay={autoPlay}
      className={className}
      onLoadStart={() => console.log('Video loadstart event - Key:', videoKey)}
      onLoadedData={() => console.log('Video loadeddata event - Key:', videoKey)}
      onPlay={() => console.log('Video play event - Key:', videoKey)}
      onPause={() => console.log('Video pause event - Key:', videoKey)}
      onEnded={() => console.log('Video ended event - Key:', videoKey)}
      onError={(e) => console.error('Video error - Key:', videoKey, e)}
    >
      <source src={webFriendlyUrl} type="video/mp4" />
      Your browser does not support the video tag.
    </video>
  );
}, (prevProps, nextProps) => {
  // Custom comparison function - only re-render if props actually changed
  const same = prevProps.url === nextProps.url &&
              prevProps.videoKey === nextProps.videoKey &&
              prevProps.className === nextProps.className &&
              prevProps.autoPlay === nextProps.autoPlay;
  
  console.log('VideoPlayer memo comparison:', {
    urlSame: prevProps.url === nextProps.url,
    keySame: prevProps.videoKey === nextProps.videoKey,
    classNameSame: prevProps.className === nextProps.className,
    autoPlaySame: prevProps.autoPlay === nextProps.autoPlay,
    shouldSkipRender: same
  });
  
  return same;
});

VideoPlayer.displayName = 'VideoPlayer';

export default VideoPlayer;
