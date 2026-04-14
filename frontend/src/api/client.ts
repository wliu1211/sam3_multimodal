import axios from "axios";
import type { AxiosError, AxiosInstance } from "axios";
function snakeToCamel(str: string): string {
    return str.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
}
function transformKeys(obj: any): any {
    if (Array.isArray(obj)) {
        return obj.map(transformKeys);
    }
    if (obj !== null && typeof obj === "object") {
        return Object.fromEntries(Object.entries(obj).map(([key, value]) => [
            snakeToCamel(key),
            transformKeys(value),
        ]));
    }
    return obj;
}
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "/api";
const TIMEOUT_MS = 120000;
export const apiClient: AxiosInstance = axios.create({
    baseURL: BASE_URL,
    timeout: TIMEOUT_MS,
    headers: {
        Accept: "application/json",
    },
});
apiClient.interceptors.request.use((config) => {
    if (import.meta.env.DEV) {
        console.debug(`[API] ${config.method?.toUpperCase()} ${config.url}`);
    }
    return config;
}, (error: AxiosError) => {
    return Promise.reject(error);
});
apiClient.interceptors.response.use((response) => {
    response.data = transformKeys(response.data);
    return response;
}, (error: AxiosError<ApiErrorBody>) => {
    const message = extractErrorMessage(error);
    const normalised = new Error(message);
    (normalised as ApiError).status = error.response?.status;
    return Promise.reject(normalised);
});
interface FastApiValidationItem {
    loc?: unknown;
    msg?: string;
    type?: string;
}
interface ApiErrorBody {
    detail?: string | FastApiValidationItem[];
    message?: string;
}
export interface ApiError extends Error {
    status?: number;
}
function extractErrorMessage(error: AxiosError<ApiErrorBody>): string {
    if (!error.response) {
        if (error.code === "ECONNABORTED") {
            return "Request timed out. Inference may be taking too long — try a smaller image.";
        }
        return "Could not reach the server. Make sure the backend is running on port 8000.";
    }
    const { status, data } = error.response;
    if (status === 429) {
        return "Please wait a bit before another request.";
    }
    if (data?.detail !== undefined && data.detail !== null) {
        if (typeof data.detail === "string")
            return data.detail;
        if (Array.isArray(data.detail)) {
            const parts = data.detail
                .map((item) => (item && typeof item.msg === "string" ? item.msg : null))
                .filter(Boolean);
            if (parts.length)
                return parts.join(" · ");
        }
    }
    if (data?.message)
        return data.message;
    switch (status) {
        case 400: return "Bad request — check your image and prompt.";
        case 415: return "Unsupported image type. Use JPEG, PNG, or WebP.";
        case 413: return "Image is too large. Please upload an image under 20MB.";
        case 422: return "Invalid request format. Please try again.";
        case 429: return "Please wait a bit before another request.";
        case 500: return "Server error during inference. Check the backend logs.";
        case 502: return "Backend is not reachable on port 8000. Run ./start-api.sh or ./run.sh (repo root), or from frontend: npm run dev:all — keep the API running while you use the app.";
        case 503: return "Server is busy or a model is unavailable. Check backend logs.";
        case 504: return "Gateway timeout — the backend took too long to respond.";
        default: return `Unexpected error (HTTP ${status}).`;
    }
}
