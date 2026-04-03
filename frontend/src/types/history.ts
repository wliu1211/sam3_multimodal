// Mirrors a row in the SQLite query_log table
export interface QueryLogEntry {
    id: string;
    prompt: string;
    baseClass: string;
    matchCount: number;
    inferenceMs: number;
    resultImageUrl: string | null;  // null if rendering failed
    createdAt: string;              // ISO 8601 timestamp
  }
  
  export interface QueryHistoryResponse {
    entries: QueryLogEntry[];
    total: number;
    page: number;
    pageSize: number;
  }