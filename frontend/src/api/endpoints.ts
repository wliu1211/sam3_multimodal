import { apiClient } from "./client";
import type { QueryRequest, QueryResponse, DiscoveryResponse, MaskResponse } from "../types/query";
import type { QueryHistoryResponse } from "../types/history";
export async function submitQuery(request: QueryRequest): Promise<QueryResponse> {
    const form = new FormData();
    form.append("image", request.image);
    form.append("prompt", request.prompt);
    const response = await apiClient.post<QueryResponse>("/query", form);
    return response.data;
}
export async function analyzeImage(image: File): Promise<DiscoveryResponse> {
    const form = new FormData();
    form.append("image", image);
    const response = await apiClient.post<DiscoveryResponse>("/analyze", form);
    return response.data;
}
export async function maskGroup(queryId: string, nodeIds: string[], color: string): Promise<MaskResponse> {
    const form = new FormData();
    form.append("query_id", queryId);
    form.append("node_ids", nodeIds.join(","));
    form.append("color", color);
    const response = await apiClient.post<MaskResponse>("/mask", form);
    return response.data;
}
export async function fetchQueryById(id: string): Promise<QueryResponse> {
    const response = await apiClient.get<QueryResponse>(`/query/${id}`);
    return response.data;
}
interface FetchHistoryParams {
    page?: number;
    pageSize?: number;
}
export async function fetchHistory(params: FetchHistoryParams = {}): Promise<QueryHistoryResponse> {
    const { page = 1, pageSize = 20 } = params;
    const response = await apiClient.get<QueryHistoryResponse>("/history", {
        params: { page, page_size: pageSize },
    });
    return response.data;
}
interface HealthResponse {
    status: "ok" | "degraded";
    modelsLoaded: {
        sam3: boolean;
        clip: boolean;
    };
    gpuAvailable: boolean;
    uptimeSeconds: number;
}
export async function fetchHealth(): Promise<HealthResponse> {
    const response = await apiClient.get<HealthResponse>("/health");
    return response.data;
}
export function getResultImageUrl(filename: string): string {
    const base = import.meta.env.VITE_API_BASE_URL ?? "/api";
    return `${base}/results/${filename}`;
}
