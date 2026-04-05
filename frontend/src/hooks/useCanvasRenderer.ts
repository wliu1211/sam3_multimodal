import { useEffect } from "react";
import type { RefObject } from "react";
// CHANGED: added Edge to the type import
import type { MaskGroup, Edge } from "../types/mask";
import { renderAllGroups } from "../utils/canvasUtils";

// CHANGED: added edges and showEdges (both optional with safe defaults
// so any existing call sites without scene graph still work unchanged)
interface UseCanvasRendererOptions {
  canvasRef:  RefObject<HTMLCanvasElement | null>;
  imageUrl:   string | null;
  groups:     MaskGroup[];
  showMasks:  boolean;
  edges?:     Edge[];      // verified scene graph edges to draw
  showEdges?: boolean;     // whether to draw relationship lines
}

/**
 * Loads the source image, sizes the canvas to match it,
 * then calls renderAllGroups to paint masks, bounding boxes,
 * and (if provided) scene graph relationship edges.
 *
 * Re-runs whenever imageUrl, groups, showMasks, edges, or showEdges changes.
 * Does nothing if canvasRef or imageUrl is null.
 */
// CHANGED: destructure edges and showEdges with safe defaults
export function useCanvasRenderer({
  canvasRef,
  imageUrl,
  groups,
  showMasks,
  edges     = [],
  showEdges = false,
}: UseCanvasRendererOptions): void {
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageUrl) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let cancelled = false;

    const img = new Image();
    img.crossOrigin = "anonymous"; // needed for images served from FastAPI /results

    img.onload = () => {
      if (cancelled) return;

      canvas.width  = img.naturalWidth;
      canvas.height = img.naturalHeight;

      // CHANGED: pass edges and showEdges through to renderAllGroups
      renderAllGroups(ctx, canvas, img, groups, showMasks, edges, showEdges);
    };

    img.onerror = () => {
      if (cancelled) return;
      console.error("[useCanvasRenderer] Failed to load image:", imageUrl);
    };

    img.src = imageUrl;

    return () => { cancelled = true; };

  // CHANGED: added edges and showEdges to the dependency array
  }, [canvasRef, imageUrl, groups, showMasks, edges, showEdges]);
}