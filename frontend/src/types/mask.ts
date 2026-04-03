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
  
  // A group of instances that matched the same parsed attribute set
  // e.g. all "blue Honda" cars form one group
  export interface MaskGroup {
    label: string;               // Human-readable: "blue Honda"
    color: string;               // Hex color for this group's overlay
    attributes: string[];        // Parsed attributes: ["blue", "Honda"]
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