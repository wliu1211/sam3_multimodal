export interface QueryLogEntry {
    id: string;
    prompt: string;
    baseClass: string;
    targetCount: number;
    matchCount: number;
    inferenceMs: number;
    resultImageUrl: string | null;
    createdAt: string;
}
export interface QueryHistoryResponse {
    entries: QueryLogEntry[];
    total: number;
    page: number;
    pageSize: number;
}
