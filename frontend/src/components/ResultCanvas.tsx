import { useEffect, useRef, useState } from "react";
import { useCanvasRenderer } from "../hooks/useCanvasRenderer";
import type { MaskGroup } from "../types/mask";

interface Props {
  imageUrl: string;
  groups: MaskGroup[];
  isAnnotated: boolean;
}

export default function ResultCanvas({ imageUrl, groups, isAnnotated }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [imageSize, setImageSize] = useState<{ w: number; h: number } | null>(null);
  const [zoom, setZoom] = useState(1);
  const [showMasks, setShowMasks] = useState(true);

  useCanvasRenderer({ canvasRef, imageUrl, groups, showMasks });

  // Compute natural size on load
  useEffect(() => {
    const img = new Image();
    img.onload = () => setImageSize({ w: img.naturalWidth, h: img.naturalHeight });
    img.src = imageUrl;
  }, [imageUrl]);

  const matchCount = groups.reduce((acc, g) => acc + g.matchCount, 0);

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
          </div>
        )}

        <div className="canvas-controls">
          {isAnnotated && (
            <button
              className={`mask-toggle ${showMasks ? "mask-toggle--on" : ""}`}
              onClick={() => setShowMasks((v) => !v)}
              title={showMasks ? "Hide masks" : "Show masks"}
            >
              <EyeIcon open={showMasks} />
              <span>{showMasks ? "Masks on" : "Masks off"}</span>
            </button>
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

      {/* Scrollable canvas area */}
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

      {/* Size info */}
      {imageSize && (
        <div className="canvas-info">
          {imageSize.w} × {imageSize.h}px
        </div>
      )}
    </div>
  );
}

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