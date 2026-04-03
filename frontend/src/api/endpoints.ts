import { apiClient } from "./client";
import type { QueryRequest, QueryResponse } from "../types/query";
import type { QueryHistoryResponse } from "../types/history";

// ── /query ────────────────────────────────────────────────────────────────────

/**
 * POST /query
 *
 * Sends the uploaded image and natural language prompt to the backend.
 * The server runs the full pipeline:
 *   LLM parse → SAM 3 detect → CLIP verify → render → return result
 *
 * Sent as multipart/form-data so the image binary is streamed directly.
 * Returns a QueryResponse containing mask groups and the annotated image URL.
 */
export async function submitQuery(request: QueryRequest): Promise<QueryResponse> {
  const form = new FormData();
  form.append("image",  request.image);
  form.append("prompt", request.prompt);

  const response = await apiClient.post<QueryResponse>("/query", form);
  return response.data;
}

// ── /query/:id ────────────────────────────────────────────────────────────────

/**
 * GET /query/:id
 *
 * Fetches a previously run query result by its UUID.
 * Useful for re-loading a result from history without re-running inference.
 */
export async function fetchQueryById(id: string): Promise<QueryResponse> {
  const response = await apiClient.get<QueryResponse>(`/query/${id}`);
  return response.data;
}

// ── /history ──────────────────────────────────────────────────────────────────

interface FetchHistoryParams {
  page?: number;
  pageSize?: number;
}

/**
 * GET /history
 *
 * Returns a paginated list of past query log entries from SQLite.
 * Each entry includes the prompt, match count, inference time,
 * and a URL to the saved annotated result image.
 */
export async function fetchHistory(
  params: FetchHistoryParams = {}
): Promise<QueryHistoryResponse> {
  const { page = 1, pageSize = 20 } = params;
  const response = await apiClient.get<QueryHistoryResponse>("/history", {
    params: { page, page_size: pageSize },
  });
  return response.data;
}

// ── /health ───────────────────────────────────────────────────────────────────

interface HealthResponse {
  status: "ok" | "degraded";
  modelsLoaded: {
    sam3: boolean;
    clip: boolean;
  };
  gpuAvailable: boolean;
  uptimeSeconds: number;
}

/**
 * GET /health
 *
 * Polls the backend to check if models are loaded and ready.
 * Used on app startup to show a loading screen if SAM 3 / CLIP
 * are still initialising (can take 10–15s on first boot).
 */
export async function fetchHealth(): Promise<HealthResponse> {
  const response = await apiClient.get<HealthResponse>("/health");
  return response.data;
}

// ── /results/:filename ────────────────────────────────────────────────────────

/**
 * Constructs the full URL for a result image served by FastAPI's
 * static file mount at /results.
 *
 * Not an API call — just a URL builder. The <img> or <canvas> tag
 * fetches it directly from the static file server.
 *
 * Example: getResultImageUrl("abc123.jpg")
 *   → "/api/results/abc123.jpg"  (proxied in dev via Vite)
 *   → "https://your-api.com/api/results/abc123.jpg"  (in production)
 */
export function getResultImageUrl(filename: string): string {
  const base = import.meta.env.VITE_API_BASE_URL ?? "/api";
  return `${base}/results/${filename}`;
}