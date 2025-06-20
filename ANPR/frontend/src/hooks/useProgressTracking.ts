
import { useState, useEffect, useRef } from 'react';
import { toast } from '@/hooks/use-toast';

export interface ProgressData {
  progress: number;
  status: string;
  message: string;
  estimated_time?: number;
  current_step?: string;
}

export interface UseProgressTrackingOptions {
  taskId: string | null;
  onComplete?: (data: any) => void;
  onError?: (error: Error) => void;
  pollInterval?: number;
  maxRetries?: number;
}

export function useProgressTracking({
  taskId,
  onComplete,
  onError,
  pollInterval = 1000,
  maxRetries = 3
}: UseProgressTrackingOptions) {
  const [progressData, setProgressData] = useState<ProgressData>({
    progress: 0,
    status: 'pending',
    message: 'Initializing...'
  });
  const [isTracking, setIsTracking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const retriesRef = useRef(0);
  const wsRef = useRef<WebSocket | null>(null);

  const pollProgress = async () => {
    if (!taskId) return;

    try {
      const response = await fetch(`http://localhost:8000/detect/progress/${taskId}`);
      
      if (!response.ok) {
        throw new Error(`Progress check failed: ${response.status}`);
      }

      const data: ProgressData = await response.json();
      setProgressData(data);
      retriesRef.current = 0; // Reset retries on success

      if (data.status === 'completed' || data.progress >= 100) {
        setIsTracking(false);
        onComplete?.(data);
      } else if (data.status === 'failed' || data.status === 'error') {
        setIsTracking(false);
        const error = new Error(data.message || 'Task failed');
        setError(error.message);
        onError?.(error);
      }
    } catch (err) {
      retriesRef.current += 1;
      
      if (retriesRef.current >= maxRetries) {
        setIsTracking(false);
        const error = err instanceof Error ? err : new Error('Progress tracking failed');
        setError(error.message);
        onError?.(error);
      }
      
      console.warn(`Progress polling failed (attempt ${retriesRef.current}/${maxRetries}):`, err);
    }
  };

  const connectWebSocket = () => {
    if (!taskId) return;

    try {
      wsRef.current = new WebSocket(`ws://localhost:8000/detect/progress/ws/${taskId}`);
      
      wsRef.current.onmessage = (event) => {
        const data: ProgressData = JSON.parse(event.data);
        setProgressData(data);
        
        if (data.status === 'completed' || data.progress >= 100) {
          setIsTracking(false);
          onComplete?.(data);
        } else if (data.status === 'failed' || data.status === 'error') {
          setIsTracking(false);
          const error = new Error(data.message || 'Task failed');
          setError(error.message);
          onError?.(error);
        }
      };

      wsRef.current.onerror = () => {
        console.warn('WebSocket failed, falling back to polling');
        startPolling();
      };

      wsRef.current.onclose = () => {
        if (isTracking) {
          console.warn('WebSocket closed, falling back to polling');
          startPolling();
        }
      };
    } catch (err) {
      console.warn('WebSocket connection failed, using polling:', err);
      startPolling();
    }
  };

  const startPolling = () => {
    if (intervalRef.current) return;
    
    intervalRef.current = setInterval(pollProgress, pollInterval);
  };

  const startTracking = () => {
    if (!taskId || isTracking) return;
    
    setIsTracking(true);
    setError(null);
    retriesRef.current = 0;
    
    // Try WebSocket first, fallback to polling
    connectWebSocket();
    
    // Start polling as backup after a short delay
    setTimeout(() => {
      if (isTracking && !wsRef.current?.readyState) {
        startPolling();
      }
    }, 2000);
  };

  const stopTracking = () => {
    setIsTracking(false);
    
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  };

  useEffect(() => {
    if (taskId) {
      startTracking();
    }

    return () => {
      stopTracking();
    };
  }, [taskId]);

  useEffect(() => {
    return () => {
      stopTracking();
    };
  }, []);

  return {
    progressData,
    isTracking,
    error,
    startTracking,
    stopTracking
  };
}
