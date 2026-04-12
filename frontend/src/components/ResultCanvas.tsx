import { useEffect, useRef, useState } from "react";
import { useCanvasRenderer } from "../hooks/useCanvasRenderer";
// CHANGED: added SceneGraph to the type import
import type { MaskGroup, SceneGraph } from "../types/mask";

interface Props {
  imageUrl: string;
  groups: MaskGroup[];
  isAnnotated: boolean;
  sceneGraph?: SceneGraph;
  relationalHints?: string[];
}


// Maps raw relational hint strings (from parser) to the set of predicates that
// should be displayed. Mirrors renderer.py's _HINT_TO_ALLOWED on the backend.
// Returns null when any hint is "open" (near / no hints) → show everything.
const HINT_TO_ALLOWED: Record<string, string[] | null> = {
  holding:        ["holding", "carrying"],
  holds:          ["holding", "carrying"],
  carrying:       ["carrying", "holding"],
  carries:        ["carrying", "holding"],
  wearing:        ["wearing"],
  wears:          ["wearing"],
  sitting:        ["sitting_on", "on_top_of"],
  sitting_on:     ["sitting_on", "on_top_of"],
  standing:       ["standing_on", "on_top_of"],
  standing_on:    ["standing_on", "on_top_of"],
  on:             ["on_top_of", "sitting_on", "standing_on"],
  on_top_of:      ["on_top_of", "sitting_on", "standing_on"],
  walking:        ["walking"],
  playing:        ["playing_with", "holding"],
  playing_with:   ["playing_with", "holding"],
  with:           ["holding", "carrying"],
  next_to:        ["parked_next_to", "near"],
  parked_next_to: ["parked_next_to", "near"],
  near:           null,   // weakest — show everything
  by:             null,
  alongside:      null,
};
function getAllowedPredicates(hints: string[]): Set<string> | null {
  if (hints.length === 0) return null;
  const allowed = new Set<string>();
  for (const hint of hints) {
    const expansion = HINT_TO_ALLOWED[hint.toLowerCase()];
    if (expansion === null) return null; // open hint → show all
    if (expansion) expansion.forEach((p) => allowed.add(p));
  }
  return allowed.size > 0 ? allowed : null;
}


export default function ResultCanvas({ imageUrl, groups, isAnnotated, sceneGraph, relationalHints }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [imageSize, setImageSize] = useState<{ w: number; h: number } | null>(null);
  const [zoom, setZoom] = useState(1);
  const [showEdges, setShowEdges] = useState(true);
  const [showRelationMasks, setShowRelationMasks] = useState(false);


  const allowedPredicates = getAllowedPredicates(relationalHints ?? []);
  const verifiedEdges = sceneGraph?.edges.filter((e) => {
    if (!e.verified) return false;
    if (e.predicate === "near") return false;
    if (allowedPredicates !== null && !allowedPredicates.has(e.predicate)) return false;
    return true;
  }) ?? [];

  const sceneNodes = sceneGraph?.nodes ?? [];

  useCanvasRenderer({
    canvasRef,
    imageUrl,
    groups,
    showMasks: true,
    edges: verifiedEdges,
    showEdges,
    nodes: sceneNodes,
    showRelationMasks,
  });

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
        {isAnnotated && matchCount > 0 && (
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
                className={`mask-toggle ${showEdges ? "mask-toggle--on" : ""}`}
                onClick={() => edgeCount > 0 && setShowEdges((v) => !v)}
                title={
                  edgeCount === 0
                    ? "No relationships detected"
                    : showEdges
                    ? "Hide relationship edges"
                    : "Show relationship edges"
                }
                style={edgeCount === 0 ? { opacity: 0.4, cursor: "default" } : undefined}
              >
                <EdgeIcon active={showEdges && edgeCount > 0} />
                <span>
                  {edgeCount === 0 ? "No edges" : showEdges ? "Edges on" : "Edges off"}
                </span>
              </button>
              <button
                className={`mask-toggle ${showRelationMasks ? "mask-toggle--on" : ""}`}
                onClick={() => edgeCount > 0 && setShowRelationMasks((v) => !v)}
                title={
                  edgeCount === 0
                    ? "No relationships to mask"
                    : showRelationMasks
                    ? "Switch to normal class masks"
                    : "Show grouped mask per relationship pair"
                }
                style={edgeCount === 0 ? { opacity: 0.4, cursor: "default" } : undefined}
              >
                <RelationMaskIcon active={showRelationMasks && edgeCount > 0} />
                <span>{showRelationMasks ? "Mask on" : "Mask off"}</span>
              </button>
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

function EdgeIcon({ active }: { active: boolean }) {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
      <circle cx="3" cy="7" r="2" stroke="currentColor" strokeWidth="1.2" />
      <circle cx="11" cy="7" r="2" stroke="currentColor" strokeWidth="1.2" />
      <line
        x1="5" y1="7" x2="9" y2="7"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeLinecap="round"
        strokeDasharray={active ? "none" : "2 2"}
      />
    </svg>
  );
}
function RelationMaskIcon({ active }: { active: boolean }) {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
      {/* Two overlapping rounded rects representing a paired mask group */}
      <rect x="1" y="4" width="6" height="6" rx="1.5"
        stroke="currentColor" strokeWidth="1.2"
        fill={active ? "currentColor" : "none"} fillOpacity="0.25"
      />
      <rect x="7" y="4" width="6" height="6" rx="1.5"
        stroke="currentColor" strokeWidth="1.2"
        fill={active ? "currentColor" : "none"} fillOpacity="0.25"
      />
      {/* Connecting dot */}
      <circle cx="7" cy="7" r="1" fill="currentColor" />
    </svg>
  );
}
