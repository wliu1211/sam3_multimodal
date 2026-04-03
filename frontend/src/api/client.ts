import axios from "axios";
import type { AxiosError, AxiosInstance } from "axios";

// ── snake_case → camelCase transformer ───────────────────────────────────────
// The FastAPI backend returns snake_case JSON keys. The frontend uses camelCase
// TypeScript types. This transformer converts all keys recursively so callers
// never need to handle snake_case manually.

function snakeToCamel(str: string): string {
  return str.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function transformKeys(obj: any): any {
  if (Array.isArray(obj)) {
    return obj.map(transformKeys);
  }
  if (obj !== null && typeof obj === "object") {
    return Object.fromEntries(
      Object.entries(obj).map(([key, value]) => [
        snakeToCamel(key),
        transformKeys(value),
      ])
    );
  }
  return obj;
}
// Base URL comes from Vite's env system.
// In development, Vite proxies /api → http://localhost:8000 (see vite.config.ts).
// In production, set VITE_API_BASE_URL to your deployed FastAPI origin.
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "/api";

// How long to wait before treating a request as timed out.
// Inference can take a few seconds, so this is intentionally generous.
const TIMEOUT_MS = 60_000;

// ── Axios instance ────────────────────────────────────────────────────────────

export const apiClient: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  timeout: TIMEOUT_MS,
  headers: {
    Accept: "application/json",
    // Content-Type is intentionally NOT set here.
    // For multipart/form-data requests (image upload), Axios sets the
    // correct boundary automatically when it sees a FormData body.
    // Setting it manually here would break that.
  },
});

// ── Request interceptor ───────────────────────────────────────────────────────

apiClient.interceptors.request.use(
  (config) => {
    // Log outgoing requests in development for easy debugging
    if (import.meta.env.DEV) {
      console.debug(`[API] ${config.method?.toUpperCase()} ${config.url}`);
    }
    return config;
  },
  (error: AxiosError) => {
    return Promise.reject(error);
  }
);

// ── Response interceptor ──────────────────────────────────────────────────────

apiClient.interceptors.response.use(
  (response) => {
    response.data = transformKeys(response.data);
    return response;
  },
  (error: AxiosError<ApiErrorBody>) => {
    // Normalise every error into a plain Error with a readable message,
    // so callers don't need to inspect AxiosError internals.
    const message = extractErrorMessage(error);
    const normalised = new Error(message);
    // Preserve HTTP status on the error object for callers that need it
    (normalised as ApiError).status = error.response?.status;
    return Promise.reject(normalised);
  }
);

// ── Error helpers ─────────────────────────────────────────────────────────────

interface ApiErrorBody {
  detail?: string;   // FastAPI's default error field
  message?: string;  // Custom field used by some endpoints
}

export interface ApiError extends Error {
  status?: number;
}

function extractErrorMessage(error: AxiosError<ApiErrorBody>): string {
  if (!error.response) {
    // No response at all — network down, server unreachable, or CORS
    if (error.code === "ECONNABORTED") {
      return "Request timed out. Inference may be taking too long — try a smaller image.";
    }
    return "Could not reach the server. Make sure the backend is running on port 8000.";
  }

  const { status, data } = error.response;

  // Use server-provided message if available
  if (data?.detail) return data.detail;
  if (data?.message) return data.message;

  // Fall back to HTTP status descriptions
  switch (status) {
    case 400: return "Bad request — check your image and prompt.";
    case 413: return "Image is too large. Please upload an image under 20MB.";
    case 422: return "Invalid request format. Please try again.";
    case 429: return "Too many requests. Please wait a moment before trying again.";
    case 500: return "Server error during inference. Check the backend logs.";
    case 503: return "Server is busy. Try again in a moment.";
    default:  return `Unexpected error (HTTP ${status}).`;
  }
}