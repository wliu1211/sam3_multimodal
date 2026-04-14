import type { MaskGroup, SceneGraph } from "./mask";
export interface QueryRequest {
    image: File;
    prompt: string;
}
export interface ParsedTargetAttributes {
    color: string | null;
    make: string | null;
    type: string | null;
    other: string[];
}
export interface ParsedTarget {
    baseClass: string;
    attributes: ParsedTargetAttributes;
    confidenceThreshold: number;
    role: "primary" | "related";
}
export interface ParsedQuery {
    targets: ParsedTarget[];
    relationalHints: string[];
}
export interface QueryResponse {
    id: string;
    prompt: string;
    parsedQuery: ParsedQuery;
    groups: MaskGroup[];
    annotatedImageUrl: string;
    inferenceMs: number;
    createdAt: string;
    sceneGraph: SceneGraph;
}
export interface ActivityGroup {
    id: string;
    label: string;
    nodeIds: string[];
    predicate: string | null;
    confidence: number;
    subjectCount: number;
    color: string;
}
export interface DiscoveryResponse {
    id: string;
    mode: "discovery";
    activities: ActivityGroup[];
    noun_tags: string[];
    sceneGraph: SceneGraph;
    inferenceMs: number;
    createdAt: string;
}
export interface MaskResponse {
    annotatedImageUrl: string;
    nodeCount: number;
    color: string;
}
export type QueryStatus = "idle" | "loading" | "success" | "error";
export interface QueryState {
    status: QueryStatus;
    result: QueryResponse | null;
    error: string | null;
}
export interface DiscoveryState {
    status: QueryStatus;
    response: DiscoveryResponse | null;
    selectedGroupId: string | null;
    maskedImageUrl: string | null;
    maskLoading: boolean;
    error: string | null;
}
