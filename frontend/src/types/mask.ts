export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

// A single detected instance — one car, one person, one dog, etc.
export interface MaskInstance {
  id: string;
  boundingBox: BoundingBox;
  // RLE-encoded binary mask from SAM 3, same dimensions as the source image
  maskRle: number[];
  // CLIP cosine similarity score against the group's attribute description
  confidence: number;
}

// A group of instances that matched one target concept from the parsed query
export interface MaskGroup {
  label: string;               // Human-readable: "blue Honda car", "banana"
  color: string;               // Hex color for this group's overlay
  attributes: string[];        // Parsed attributes: ["blue", "Honda"]
  role: "primary" | "related"; // primary = user's main focus; related = context
  instances: MaskInstance[];
  matchCount: number;          // Convenience — instances.length
  avgConfidence: number;       // Mean CLIP score across all instances
}

// Full decoded mask as a flat Uint8Array (1 = matched pixel, 0 = background)
// Dimensions match the source image width × height
export interface DecodedMask {
  data: Uint8Array;
  width: number;
  height: number;
}

// ── NEW: Scene graph types ────────────────────────────────────────────────────

// A verified relationship between two detected nodes.
// Mirrors the Edge dataclass in backend/services/graph.py
export interface Edge {
  subjectId: string;
  objectId: string;
  predicate: string;              // e.g. "holding", "walking", "near"
  confidence: number;             // SGG prediction confidence in [0, 1]
  verified: boolean;              // true if VTT geometric check passed
  downgraded: boolean;            // true if predicate was lowered from original
  originalPredicate: string | null;
}

// A node in the scene graph — maps to a detected instance
export interface SceneGraphNode {
  id: string;
  label: string;
  boundingBox: BoundingBox;
  maskRle: number[];
  confidence: number;
  isAttribute: boolean;
  parentId: string | null;
  role: "primary" | "related";   // mirrors ParsedTarget.role
  centroid: [number, number];    // [x, y] pixel coordinates
}

// The complete scene graph returned by the backend for one query.
// Mirrors SceneGraph.to_dict() in backend/services/graph.py
export interface SceneGraph {
  nodes: SceneGraphNode[];
  edges: Edge[];
  summary: {
    totalNodes: number;
    baseNodes: number;
    attributeNodes: number;
    totalEdges: number;
    verifiedEdges: number;
  };
}