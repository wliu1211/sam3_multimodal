import type { MaskGroup, SceneGraph } from "./mask";

// ── Outgoing request ──────────────────────────────────────────────────────────

// Sent as multipart/form-data — FormData is constructed in useQuerySubmit
export interface QueryRequest {
  image: File;
  prompt: string;
}

// ── Parsed query types — mirrors backend parser.py ────────────────────────────

// Attributes for one target concept extracted from the user's prompt
export interface ParsedTargetAttributes {
  color: string | null;
  make: string | null;         // Brand / manufacturer
  type: string | null;         // Sub-type: sedan, SUV, retriever…
  other: string[];
}

// A single target concept (object class + attributes + role)
export interface ParsedTarget {
  baseClass: string;                    // e.g. "monkey", "banana", "car"
  attributes: ParsedTargetAttributes;
  confidenceThreshold: number;          // Per-target CLIP threshold
  role: "primary" | "related";         // primary = user's focus; related = context
}

// Full parsed query — one entry per object type mentioned in the prompt
export interface ParsedQuery {
  targets: ParsedTarget[];
  relationalHints: string[];           // e.g. ["holding", "on_top_of"]
}

// ── Incoming response ─────────────────────────────────────────────────────────

export interface QueryResponse {
  id: string;                  // UUID — used to fetch result image
  prompt: string;              // Echo of the original prompt
  parsedQuery: ParsedQuery;    // What the LLM extracted (multi-target)
  groups: MaskGroup[];         // One entry per target (primary + related)
  annotatedImageUrl: string;   // URL to the rendered overlay image
  inferenceMs: number;         // Total server-side inference time
  createdAt: string;           // ISO timestamp
  sceneGraph: SceneGraph;      // Nodes + verified cross-class edges
}

// ── UI state ──────────────────────────────────────────────────────────────────

export type QueryStatus = "idle" | "loading" | "success" | "error";

export interface QueryState {
  status: QueryStatus;
  result: QueryResponse | null;
  error: string | null;
}