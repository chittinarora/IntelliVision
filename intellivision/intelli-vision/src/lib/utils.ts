import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Ensures the input URL uses HTTPS protocol, replacing http:// with https://
 */
export function ensureHttpsUrl(url?: string | null): string | undefined {
  if (!url) return url ?? undefined;
  return url.startsWith("http://") ? url.replace(/^http:\/\//, "https://") : url;
}
