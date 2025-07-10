
/**
 * Utility functions for converting media files to web-friendly formats
 */

export interface MediaConversionOptions {
  quality?: number; // 0.1 to 1.0
  maxWidth?: number;
  maxHeight?: number;
}

/**
 * Checks if a video URL has a web-friendly format
 */
export const isWebFriendlyVideo = (url: string): boolean => {
  if (!url) return false;
  
  const webFriendlyExtensions = ['.mp4', '.webm'];
  const urlLower = url.toLowerCase();
  
  return webFriendlyExtensions.some(ext => urlLower.includes(ext));
};

/**
 * Checks if an image URL has a web-friendly format
 */
export const isWebFriendlyImage = (url: string): boolean => {
  if (!url) return false;
  
  const webFriendlyExtensions = ['.jpg', '.jpeg', '.png', '.webp'];
  const urlLower = url.toLowerCase();
  
  return webFriendlyExtensions.some(ext => urlLower.includes(ext));
};

/**
 * Converts an image to WebP format using HTML5 Canvas
 */
export const convertImageToWebFormat = async (
  imageUrl: string, 
  options: MediaConversionOptions = {}
): Promise<string> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      if (!ctx) {
        reject(new Error('Could not get canvas context'));
        return;
      }
      
      // Set canvas dimensions (with optional max constraints)
      const maxWidth = options.maxWidth || 1920;
      const maxHeight = options.maxHeight || 1080;
      
      let { naturalWidth: imgWidth, naturalHeight: imgHeight } = img;
      
      // Scale down if necessary
      if (imgWidth > maxWidth || imgHeight > maxHeight) {
        const aspectRatio = imgWidth / imgHeight;
        if (imgWidth > imgHeight) {
          imgWidth = maxWidth;
          imgHeight = maxWidth / aspectRatio;
        } else {
          imgHeight = maxHeight;
          imgWidth = maxHeight * aspectRatio;
        }
      }
      
      canvas.width = imgWidth;
      canvas.height = imgHeight;
      
      // Draw image to canvas
      ctx.drawImage(img, 0, 0, imgWidth, imgHeight);
      
      // Convert to WebP blob
      canvas.toBlob((blob) => {
        if (blob) {
          const convertedUrl = URL.createObjectURL(blob);
          resolve(convertedUrl);
        } else {
          reject(new Error('Failed to convert image'));
        }
      }, 'image/webp', options.quality || 0.8);
    };
    
    img.onerror = () => {
      reject(new Error('Failed to load image for conversion'));
    };
    
    img.src = imageUrl;
  });
};

/**
 * Converts a video to MP4 format using HTML5 Canvas and MediaRecorder API
 */
export const convertVideoToWebFormat = async (
  videoUrl: string, 
  options: MediaConversionOptions = {}
): Promise<string> => {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    video.crossOrigin = 'anonymous';
    video.muted = true;
    
    video.onloadedmetadata = () => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      if (!ctx) {
        reject(new Error('Could not get canvas context'));
        return;
      }
      
      // Set canvas dimensions (with optional max constraints)
      const maxWidth = options.maxWidth || 1920;
      const maxHeight = options.maxHeight || 1080;
      
      let { videoWidth, videoHeight } = video;
      
      // Scale down if necessary
      if (videoWidth > maxWidth || videoHeight > maxHeight) {
        const aspectRatio = videoWidth / videoHeight;
        if (videoWidth > videoHeight) {
          videoWidth = maxWidth;
          videoHeight = maxWidth / aspectRatio;
        } else {
          videoHeight = maxHeight;
          videoWidth = maxHeight * aspectRatio;
        }
      }
      
      canvas.width = videoWidth;
      canvas.height = videoHeight;
      
      // Create MediaRecorder for MP4 output
      const stream = canvas.captureStream(30); // 30 FPS
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'video/mp4; codecs=h264',
        videoBitsPerSecond: 2500000 // 2.5 Mbps
      });
      
      const chunks: Blob[] = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'video/mp4' });
        const convertedUrl = URL.createObjectURL(blob);
        resolve(convertedUrl);
      };
      
      mediaRecorder.onerror = (event) => {
        reject(new Error('MediaRecorder error: ' + event));
      };
      
      // Start recording
      mediaRecorder.start();
      
      video.onloadeddata = () => {
        video.currentTime = 0;
        video.play();
      };
      
      video.ontimeupdate = () => {
        ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
      };
      
      video.onended = () => {
        mediaRecorder.stop();
        video.remove();
        canvas.remove();
      };
    };
    
    video.onerror = () => {
      reject(new Error('Failed to load video for conversion'));
    };
    
    video.src = videoUrl;
  });
};

/**
 * Gets a web-friendly media URL, converting if necessary
 */
export const getWebFriendlyMediaUrl = async (originalUrl: string, isVideo: boolean = false): Promise<string> => {
  if (!originalUrl) return originalUrl;
  
  // Determine if conversion is needed
  const isAlreadyWebFriendly = isVideo ? isWebFriendlyVideo(originalUrl) : isWebFriendlyImage(originalUrl);
  
  if (isAlreadyWebFriendly) {
    return originalUrl;
  }
  
  console.log(`Converting ${isVideo ? 'video' : 'image'} to web-friendly format:`, originalUrl);
  
  try {
    const convertedUrl = isVideo 
      ? await convertVideoToWebFormat(originalUrl, {
          quality: 0.8,
          maxWidth: 1920,
          maxHeight: 1080
        })
      : await convertImageToWebFormat(originalUrl, {
          quality: 0.8,
          maxWidth: 1920,
          maxHeight: 1080
        });
    
    console.log(`${isVideo ? 'Video' : 'Image'} conversion successful`);
    return convertedUrl;
  } catch (error) {
    console.error(`${isVideo ? 'Video' : 'Image'} conversion failed, using original URL:`, error);
    return originalUrl; // Fallback to original
  }
};

/**
 * Legacy function for backward compatibility
 */
export const getWebFriendlyVideoUrl = (originalUrl: string): Promise<string> => {
  return getWebFriendlyMediaUrl(originalUrl, true);
};
