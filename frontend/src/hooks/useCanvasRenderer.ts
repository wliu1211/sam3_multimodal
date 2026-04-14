import { useEffect } from "react";
import type { RefObject } from "react";
import type { MaskGroup, Edge, SceneGraphNode } from "../types/mask";
import { renderAllGroups } from "../utils/canvasUtils";
interface UseCanvasRendererOptions {
    canvasRef: RefObject<HTMLCanvasElement | null>;
    imageUrl: string | null;
    groups: MaskGroup[];
    showMasks: boolean;
    edges?: Edge[];
    showEdges?: boolean;
    nodes?: SceneGraphNode[];
    showRelationMasks?: boolean;
}
export function useCanvasRenderer({ canvasRef, imageUrl, groups, showMasks, edges = [], showEdges = false, nodes = [], showRelationMasks = false, }: UseCanvasRendererOptions): void {
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || !imageUrl)
            return;
        const ctx = canvas.getContext("2d");
        if (!ctx)
            return;
        let cancelled = false;
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => {
            if (cancelled)
                return;
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            renderAllGroups(ctx, canvas, img, groups, showMasks, edges, showEdges, nodes, showRelationMasks);
        };
        img.onerror = () => {
            if (cancelled)
                return;
            console.error("[useCanvasRenderer] Failed to load image:", imageUrl);
        };
        img.src = imageUrl;
        return () => { cancelled = true; };
    }, [canvasRef, imageUrl, groups, showMasks, edges, showEdges, nodes, showRelationMasks]);
}
