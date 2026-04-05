import { useEffect, useRef, useState } from "react";
import { useCanvasRenderer } from "../hooks/useCanvasRenderer";
// CHANGED: added SceneGraph to the type import
import type { MaskGroup, SceneGraph } from "../types/mask";

// CHANGED: added optional sceneGraph prop
interface Props {
  imageUrl: string;
  groups: MaskGroup[];
  isAnnotated: boolean;
  sceneGraph?: SceneGraph;
}

// CHANGED: destructure sceneGraph from props
export default function ResultCanvas({ imageUrl, groups, isAnnotated, sceneGraph }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [imageSize, setImageSize] = useState<{ w: number; h: number } | null>(null);
  const [zoom, setZoom] = useState(1);
  const [showMasks, setShowMasks] = useState(true);
  // CHANGED: new showEdges state — mirrors showMasks pattern
  const [showEdges, setShowEdges] = useState(true);

  // CHANGED: pass edges and showEdges to useCanvasRenderer
  const verifiedEdges = sceneGraph?.edges.filter((e) => e.verified) ?? [];
  useCanvasRenderer({ canvasRef, imageUrl, groups, showMasks, edges: verifiedEdges, showEdges });

  // Compute natural size on load
  useEffect(() => {
    const img = new Image();
    img.onload = () => setImageSize({ w: img.naturalWidth, h: img.naturalHeight });
    img.src = imageUrl;
  }, [imageUrl]);

  const matchCount = groups.reduce((acc, g) => acc + g.matchCount, 0);
  const edgeCount  = verifiedEdges.length;

  return (
    <div className="result-canvas-wrapper">
      {/* Toolbar */}
      <div className="canvas-toolbar">
        {isAnnotated && (
          <div className="canvas-stats">
            <span className="match-badge">
              {matchCount} match{matchCount !== 1 ? "es" : ""} found
            </span>
            {groups.map((g) => (
              <span
                key={g.label}
                className="group-badge"
                style={{ borderColor: g.color, color: g.color }}
              >
                {g.matchCount}× {g.label}
              </span>
            ))}
            {/* CHANGED: edge count badge — only shown when edges exist */}
            {edgeCount > 0 && (
              <span className="group-badge" style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}>
                {edgeCount} relationship{edgeCount !== 1 ? "s" : ""}
              </span>
            )}
          </div>
        )}

        <div className="canvas-controls">
          {isAnnotated && (
            <>
              <button
                className={`mask-toggle ${showMasks ? "mask-toggle--on" : ""}`}
                onClick={() => setShowMasks((v) => !v)}
                title={showMasks ? "Hide masks" : "Show masks"}
              >
                <EyeIcon open={showMasks} />
                <span>{showMasks ? "Masks on" : "Masks off"}</span>
              </button>

              {/* CHANGED: edge toggle button — only rendered when edges exist */}
              {edgeCount > 0 && (
                <button
                  className={`mask-toggle ${showEdges ? "mask-toggle--on" : ""}`}
                  onClick={() => setShowEdges((v) => !v)}
                  title={showEdges ? "Hide relationships" : "Show relationships"}
                >
                  <EdgeIcon />
                  <span>{showEdges ? "Edges on" : "Edges off"}</span>
                </button>
              )}
            </>
          )}

          <div className="zoom-controls">
            <button
              className="zoom-btn"
              onClick={() => setZoom((z) => Math.max(0.25, z - 0.25))}
              disabled={zoom <= 0.25}
              aria-label="Zoom out"
            >−</button>
            <span className="zoom-label">{Math.round(zoom * 100)}%</span>
            <button
              className="zoom-btn"
              onClick={() => setZoom((z) => Math.min(4, z + 0.25))}
              disabled={zoom >= 4}
              aria-label="Zoom in"
            >+</button>
            <button
              className="zoom-btn zoom-btn--reset"
              onClick={() => setZoom(1)}
              aria-label="Reset zoom"
            >↺</button>
          </div>
        </div>
      </div>

      {/* Scrollable canvas area — unchanged */}
      <div className="canvas-scroll-area">
        <div
          className="canvas-scaler"
          style={{
            transform: `scale(${zoom})`,
            transformOrigin: "top left",
            width: imageSize ? imageSize.w : "100%",
            height: imageSize ? imageSize.h : "auto",
          }}
        >
          <canvas
            ref={canvasRef}
            className="result-canvas"
            aria-label={isAnnotated ? "Annotated image with mask overlays" : "Uploaded image"}
          />
        </div>
      </div>

      {/* Size info — unchanged */}
      {imageSize && (
        <div className="canvas-info">
          {imageSize.w} × {imageSize.h}px
        </div>
      )}
    </div>
  );
}

// Unchanged
function EyeIcon({ open }: { open: boolean }) {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
      <ellipse cx="7" cy="7" rx="5.5" ry="3.5" stroke="currentColor" strokeWidth="1.2" />
      <circle cx="7" cy="7" r="1.5" fill="currentColor" />
      {!open && (
        <path d="M2 12L12 2" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
      )}
    </svg>
  );
}

// NEW: icon for the edge toggle button
function EdgeIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
      <circle cx="3" cy="7" r="2" stroke="currentColor" strokeWidth="1.2" />
      <circle cx="11" cy="7" r="2" stroke="currentColor" strokeWidth="1.2" />
      <line x1="5" y1="7" x2="9" y2="7" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
    </svg>
  );
}