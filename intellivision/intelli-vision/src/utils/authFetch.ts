
import { DJANGO_API_BASE } from '@/constants/api';

export interface AuthFetchOptions extends RequestInit {
  requireAuth?: boolean;
}

export const authFetch = async (
  endpoint: string,
  options: AuthFetchOptions = {}
): Promise<Response> => {
  const { requireAuth = true, headers = {}, ...restOptions } = options;
  
  const token = localStorage.getItem('accessToken');
  
  if (requireAuth && !token) {
    // Redirect to login if no token found
    window.location.href = '/login';
    throw new Error('No authentication token found');
  }

  const authHeaders: Record<string, string> = {
    'ngrok-skip-browser-warning': 'true',
    ...(headers as Record<string, string>),
  };

  // Only add Content-Type for non-FormData requests
  if (!(restOptions.body instanceof FormData)) {
    authHeaders['Content-Type'] = 'application/json';
  }

  if (token && requireAuth) {
    authHeaders.Authorization = `Bearer ${token}`;
  }

  const url = endpoint.startsWith('http') ? endpoint : `${DJANGO_API_BASE}${endpoint}`;

  const response = await fetch(url, {
    ...restOptions,
    headers: authHeaders,
  });

  // Handle token expiration
  if (response.status === 401 && requireAuth) {
    console.log('Token expired or invalid, redirecting to login');
    localStorage.removeItem('accessToken');
    window.location.href = '/login';
    throw new Error('Authentication token expired');
  }

  return response;
};

export const logout = () => {
  localStorage.removeItem('accessToken');
  window.location.href = '/';
};
