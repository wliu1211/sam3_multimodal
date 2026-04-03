import type { MaskGroup } from "./mask";

// ── Outgoing request ──────────────────────────────────────────────────────────

// Sent as multipart/form-data — FormData is constructed in useQuerySubmit
export interface QueryRequest {
  image: File;
  prompt: string;
}

// What the LLM parser extracts from the raw prompt — mirrors backend parser.py
export interface ParsedQuery {
  baseClass: string;           // e.g. "car"
  attributes: {
    color: string | null;
    make: string | null;       // Brand / manufacturer
    type: string | null;       // sedan, SUV, etc.
    other: string[];
  };
  confidenceThreshold: number; // Default 0.65
}

// ── Incoming response ─────────────────────────────────────────────────────────

export interface QueryResponse {
  id: string;                  // UUID — used to fetch result image
  prompt: string;              // Echo of the original prompt
  parsedQuery: ParsedQuery;    // What the LLM extracted
  groups: MaskGroup[];         // One entry per matched attribute group
  annotatedImageUrl: string;   // URL to the rendered overlay image
  inferenceMs: number;         // Total server-side inference time
  createdAt: string;           // ISO timestamp
}

// ── UI state ──────────────────────────────────────────────────────────────────

export type QueryStatus = "idle" | "loading" | "success" | "error";

export interface QueryState {
  status: QueryStatus;
  result: QueryResponse | null;
  error: string | null;
}