
import { authFetch } from '@/utils/authFetch';
import { DJANGO_API_BASE } from '@/constants/api';

export interface LoginRequest {
  username: string;
  image: Blob;
}

export interface RegisterRequest {
  username: string;
  image: Blob;
}

export interface AuthResponse {
  task_id: string;
}

export interface TaskStatusResponse {
  state: string;
  result?: any;
  success?: boolean;
  message?: string;
  token?: string;
  name?: string;
  image?: string;
}

export const loginWithFace = async (data: LoginRequest): Promise<AuthResponse> => {
  const formData = new FormData();
  formData.append('username', data.username);
  formData.append('image', data.image, 'face_capture.jpg');

  const response = await fetch(`${DJANGO_API_BASE}/faceauth/login-face/`, {
    method: 'POST',
    headers: { 'ngrok-skip-browser-warning': 'true' },
    body: formData,
  });

  if (!response.ok) {
    let errorMessage = 'Login failed';
    try {
      const errorData = await response.json();
      errorMessage = errorData.message || errorMessage;
    } catch {
      errorMessage = `Login failed with status ${response.status}`;
    }
    throw new Error(errorMessage);
  }

  return response.json();
};

export const registerWithFace = async (data: RegisterRequest): Promise<AuthResponse> => {
  const formData = new FormData();
  formData.append('username', data.username);
  formData.append('image', data.image, 'face_capture.jpg');

  const response = await fetch(`${DJANGO_API_BASE}/faceauth/register-face/`, {
    method: 'POST',
    headers: { 'ngrok-skip-browser-warning': 'true' },
    body: formData,
  });

  if (!response.ok) {
    let errorMessage = 'Registration failed';
    try {
      const errorData = await response.json();
      errorMessage = errorData.message || errorMessage;
    } catch {
      errorMessage = `Registration failed with status ${response.status}`;
    }
    throw new Error(errorMessage);
  }

  return response.json();
};

export const pollTaskStatus = async (taskId: string, interval = 2000, timeout = 60000): Promise<any> => {
  const start = Date.now();

  return new Promise((resolve, reject) => {
    const check = async () => {
      if (Date.now() - start > timeout) {
        reject(new Error('Verification timed out'));
        return;
      }

      try {
        const res = await authFetch(`/faceauth/task-status/${taskId}/`, {
          requireAuth: false,
          method: 'GET'
        });

        if (!res.ok) {
          reject(new Error(`Task status check failed with status ${res.status}`));
          return;
        }

        let data: TaskStatusResponse;
        try {
          data = await res.json();
        } catch (parseError) {
          console.error('Failed to parse JSON response:', parseError);
          const rawText = await res.text().catch(() => 'Unable to read response');
          console.error('Raw response:', rawText);
          reject(new Error('Invalid response format from server'));
          return;
        }

        console.log('Task status response:', data);

        if (data.state === 'SUCCESS') {
          // For SUCCESS state, the response structure is different
          // The actual result data is directly in the response, not nested under 'result'
          const result = {
            success: data.success !== undefined ? data.success : true,
            message: data.message || 'Authentication completed',
            token: data.token || null,
            name: data.name || null,
            image: data.image || null
          };
          
          console.log('Processed result:', result);
          resolve(result);
        } else if (data.state === 'FAILURE') {
          const errorMessage = (data.result && typeof data.result === 'string') 
            ? data.result 
            : (data.result && data.result.message) 
            ? data.result.message 
            : 'Verification failed';
          reject(new Error(errorMessage));
        } else {
          // Still pending, continue polling
          setTimeout(check, interval);
        }
      } catch (err) {
        console.error('Error checking task status:', err);
        reject(new Error('Error checking task status'));
      }
    };

    check();
  });
};
