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
// SAM3 inference on CPU can take 30-60s; set this well above that.
// The backend queue will reject after 100s with a clean 503 message,
// so this 120s ceiling mainly guards against truly hung requests.
const TIMEOUT_MS = 120_000;

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

interface FastApiValidationItem {
  loc?: unknown;
  msg?: string;
  type?: string;
}

interface ApiErrorBody {
  /** FastAPI: string for HTTPException, or validation error array */
  detail?: string | FastApiValidationItem[];
  message?: string; // Custom field used by some endpoints
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
  if (data?.detail !== undefined && data.detail !== null) {
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail)) {
      const parts = data.detail
        .map((item) => (item && typeof item.msg === "string" ? item.msg : null))
        .filter(Boolean);
      if (parts.length) return parts.join(" · ");
    }
  }
  if (data?.message) return data.message;

  // Fall back to HTTP status descriptions
  switch (status) {
    case 400: return "Bad request — check your image and prompt.";
    case 415: return "Unsupported image type. Use JPEG, PNG, or WebP.";
    case 413: return "Image is too large. Please upload an image under 20MB.";
    case 422: return "Invalid request format. Please try again.";
    case 429: return "Too many requests. Please wait a moment before trying again.";
    case 500: return "Server error during inference. Check the backend logs.";
    case 502: return "Backend is not reachable on port 8000. Run ./start-api.sh or ./run.sh (repo root), or from frontend: npm run dev:all — keep the API running while you use the app.";
    case 503: return "Server is busy or a model is unavailable. Check backend logs.";
    case 504: return "Gateway timeout — the backend took too long to respond.";
    default:  return `Unexpected error (HTTP ${status}).`;
  }
}