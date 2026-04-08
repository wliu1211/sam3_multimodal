import { useEffect } from "react";
import type { RefObject } from "react";
// CHANGED: added Edge to the type import
import type { MaskGroup, Edge, SceneGraphNode } from "../types/mask";
import { renderAllGroups } from "../utils/canvasUtils";

// CHANGED: added edges and showEdges (both optional with safe defaults
// so any existing call sites without scene graph still work unchanged)
interface UseCanvasRendererOptions {
  canvasRef:          RefObject<HTMLCanvasElement | null>;
  imageUrl:           string | null;
  groups:             MaskGroup[];
  showMasks:          boolean;
  edges?:             Edge[];             // verified scene graph edges to draw
  showEdges?:         boolean;            // whether to draw relationship lines
  nodes?:             SceneGraphNode[];   // scene graph nodes for relation masks
  showRelationMasks?: boolean;            // whether to draw per-edge pair masks
}


export function useCanvasRenderer({
  canvasRef,
  imageUrl,
  groups,
  showMasks,
  edges             = [],
  showEdges         = false,
  nodes             = [],
  showRelationMasks = false,
}: UseCanvasRendererOptions): void {
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageUrl) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let cancelled = false;

    const img = new Image();
    img.crossOrigin = "anonymous";

    img.onload = () => {
      if (cancelled) return;
      canvas.width  = img.naturalWidth;
      canvas.height = img.naturalHeight;

      // CHANGED: pass edges and showEdges through to renderAllGroups
      renderAllGroups(ctx, canvas, img, groups, showMasks, edges, showEdges, nodes, showRelationMasks);
    };

    img.onerror = () => {
      if (cancelled) return;
      console.error("[useCanvasRenderer] Failed to load image:", imageUrl);
    };

    img.src = imageUrl;

    return () => { cancelled = true; };

  }, [canvasRef, imageUrl, groups, showMasks, edges, showEdges, nodes, showRelationMasks]);
}