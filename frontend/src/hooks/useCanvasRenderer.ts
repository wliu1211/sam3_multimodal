import { useEffect } from "react";
import type { RefObject } from "react";
import type { MaskGroup } from "../types/mask";
import { renderAllGroups } from "../utils/canvasUtils";

interface UseCanvasRendererOptions {
  canvasRef: RefObject<HTMLCanvasElement | null>;
  imageUrl: string | null;
  groups: MaskGroup[];
  showMasks: boolean;
}

/**
 * Loads the source image, sizes the canvas to match it,
 * then calls renderAllGroups to paint masks and bounding boxes.
 *
 * Re-runs whenever imageUrl, groups, or showMasks changes.
 * Does nothing if canvasRef or imageUrl is null.
 */
export function useCanvasRenderer({
  canvasRef,
  imageUrl,
  groups,
  showMasks,
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

      // Size canvas to natural image dimensions
      canvas.width  = img.naturalWidth;
      canvas.height = img.naturalHeight;

      renderAllGroups(ctx, canvas, img, groups, showMasks);
    };

    img.onerror = () => {
      if (cancelled) return;
      console.error("[useCanvasRenderer] Failed to load image:", imageUrl);
    };

    img.src = imageUrl;

    // Cleanup: prevent stale renders if the effect re-runs before load completes
    return () => { cancelled = true; };
  }, [canvasRef, imageUrl, groups, showMasks]);
}