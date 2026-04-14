export interface BoundingBox {
    x1: number;
    y1: number;
    x2: number;
    y2: number;
}
export interface MaskInstance {
    id: string;
    boundingBox: BoundingBox;
    maskRle: number[];
    confidence: number;
}
export interface MaskGroup {
    label: string;
    color: string;
    attributes: string[];
    role: "primary" | "related";
    instances: MaskInstance[];
    matchCount: number;
    avgConfidence: number;
}
export interface DecodedMask {
    data: Uint8Array;
    width: number;
    height: number;
}
export interface Edge {
    subjectId: string;
    objectId: string;
    predicate: string;
    confidence: number;
    verified: boolean;
    downgraded: boolean;
    originalPredicate: string | null;
}
export interface SceneGraphNode {
    id: string;
    label: string;
    boundingBox: BoundingBox;
    maskRle: number[];
    confidence: number;
    isAttribute: boolean;
    parentId: string | null;
    role: "primary" | "related";
    centroid: [
        number,
        number
    ];
}
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
