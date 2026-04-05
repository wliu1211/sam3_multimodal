// CHANGED: added Edge to the import
import type { MaskGroup, DecodedMask, BoundingBox, Edge } from "../types/mask";
import { hexToRgba, darkenHex } from "./colorUtils";

const MASK_ALPHA       = 0.45;
const BORDER_ALPHA     = 0.9;
const BORDER_WIDTH     = 2;
const LABEL_FONT       = "600 12px 'JetBrains Mono', monospace";
const LABEL_PADDING    = 4;
const LABEL_OFFSET_Y   = -6;

// NEW: edge drawing constants
const EDGE_LINE_WIDTH  = 1.5;
const EDGE_FONT        = "500 11px 'JetBrains Mono', monospace";
const EDGE_LABEL_PAD   = 3;
const EDGE_DOT_RADIUS  = 3;

/**
 * Decodes a run-length encoded (RLE) mask into a flat Uint8Array.
 * RLE format: alternating [background_count, foreground_count, ...]
 * starting from the top-left pixel, row-major order.
 */
export function decodeRleMask(rle: number[], width: number, height: number): DecodedMask {
  const data = new Uint8Array(width * height);
  let pixel = 0;
  let value = 0; // starts with background (0)
  for (const count of rle) {
    data.fill(value, pixel, pixel + count);
    pixel += count;
    value = value === 0 ? 1 : 0;
  }
  return { data, width, height };
}

/**
 * Draws a semi-transparent color fill onto the canvas for every pixel
 * where the mask is 1. Operates on a pre-existing ImageData pixel buffer
 * for performance (batch write, single putImageData).
 */
export function paintMask(
  imageData: ImageData,
  mask: DecodedMask,
  color: string,
): void {
  const [r, g, b] = hexToRgba(color, MASK_ALPHA);
  const d = imageData.data;
  for (let i = 0; i < mask.data.length; i++) {
    if (mask.data[i] !== 1) continue;
    const px = i * 4;
    const a = MASK_ALPHA;
    d[px]     = Math.round(r * a + d[px]     * (1 - a));
    d[px + 1] = Math.round(g * a + d[px + 1] * (1 - a));
    d[px + 2] = Math.round(b * a + d[px + 2] * (1 - a));
  }
}

/**
 * Draws a bounding box and label badge for a single instance.
 */
export function drawBoundingBox(
  ctx: CanvasRenderingContext2D,
  box: BoundingBox,
  label: string,
  color: string,
  confidence: number,
): void {
  const borderColor = darkenHex(color, 10);
  const w = box.x2 - box.x1;
  const h = box.y2 - box.y1;

  ctx.strokeStyle = borderColor;
  ctx.globalAlpha = BORDER_ALPHA;
  ctx.lineWidth   = BORDER_WIDTH;
  ctx.strokeRect(box.x1, box.y1, w, h);
  ctx.globalAlpha = 1;

  const badgeText = `${label} ${Math.round(confidence * 100)}%`;
  ctx.font = LABEL_FONT;
  const textWidth = ctx.measureText(badgeText).width;
  const badgeW = textWidth + LABEL_PADDING * 2;
  const badgeH = 18;
  const badgeX = box.x1;
  const badgeY = box.y1 + LABEL_OFFSET_Y - badgeH;

  ctx.fillStyle = borderColor;
  ctx.globalAlpha = 0.85;
  roundRect(ctx, badgeX, badgeY, badgeW, badgeH, 3);
  ctx.fill();
  ctx.globalAlpha = 1;

  ctx.fillStyle = "#ffffff";
  ctx.fillText(badgeText, badgeX + LABEL_PADDING, badgeY + badgeH - LABEL_PADDING - 1);
}

// ── NEW: drawEdges ─────────────────────────────────────────────────────────────

/**
 * Draws relationship lines between matched instance centroids.
 *
 * Derives centroid coordinates from MaskGroup → MaskInstance → boundingBox
 * so no extra props need to be passed — the groups data already in scope
 * is sufficient. Builds an id → centroid lookup, then draws a line for
 * each verified edge with a predicate label badge at the midpoint.
 *
 * Line opacity scales with edge confidence: high-confidence edges are
 * more opaque and therefore more prominent.
 * Downgraded edges are prefixed with "~" to signal they were inferred
 * from a weaker version of the original predicted predicate.
 */
export function drawEdges(
  ctx:    CanvasRenderingContext2D,
  groups: MaskGroup[],
  edges:  Edge[],
): void {
  if (edges.length === 0) return;

  // Build id → centroid from all instances in all groups
  const centroids = new Map<string, { x: number; y: number }>();
  for (const group of groups) {
    for (const inst of group.instances) {
      const b = inst.boundingBox;
      centroids.set(inst.id, {
        x: (b.x1 + b.x2) / 2,
        y: (b.y1 + b.y2) / 2,
      });
    }
  }

  for (const edge of edges) {
    const sub = centroids.get(edge.subjectId);
    const obj = centroids.get(edge.objectId);
    if (!sub || !obj) continue;

    // Opacity scaled by confidence, floored at 0.25 for readability
    const alpha = Math.max(0.25, Math.min(1.0, edge.confidence));

    // ── Draw relationship line ──────────────────────────────────────────────
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = "rgba(220, 220, 220, 1)";
    ctx.lineWidth   = EDGE_LINE_WIDTH;
    ctx.setLineDash([5, 4]);    // dashed so it doesn't compete with mask borders
    ctx.beginPath();
    ctx.moveTo(sub.x, sub.y);
    ctx.lineTo(obj.x, obj.y);
    ctx.stroke();
    ctx.setLineDash([]);

    // ── Endpoint dots ───────────────────────────────────────────────────────
    ctx.fillStyle = "rgba(220, 220, 220, 1)";
    ctx.beginPath();
    ctx.arc(sub.x, sub.y, EDGE_DOT_RADIUS, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(obj.x, obj.y, EDGE_DOT_RADIUS, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // ── Predicate label badge at midpoint ───────────────────────────────────
    const midX = (sub.x + obj.x) / 2;
    const midY = (sub.y + obj.y) / 2;

    // Prefix downgraded predicates with ~ to show they were inferred weakly
    const predText  = `${edge.downgraded ? "~" : ""}${edge.predicate.replace(/_/g, " ")}`;
    const confText  = ` ${Math.round(edge.confidence * 100)}%`;
    const badgeText = predText + confText;

    ctx.font = EDGE_FONT;
    const textW  = ctx.measureText(badgeText).width;
    const badgeW = textW + EDGE_LABEL_PAD * 2;
    const badgeH = 16;
    const badgeX = midX - badgeW / 2;
    const badgeY = midY - badgeH / 2;

    // Semi-transparent dark background
    ctx.save();
    ctx.globalAlpha = 0.80;
    ctx.fillStyle = "#1a1a2a";
    roundRect(ctx, badgeX, badgeY, badgeW, badgeH, 3);
    ctx.fill();
    ctx.globalAlpha = 1;

    // Label text
    ctx.fillStyle = "#e0e0e0";
    ctx.fillText(
      badgeText,
      badgeX + EDGE_LABEL_PAD,
      badgeY + badgeH - EDGE_LABEL_PAD - 2,
    );
    ctx.restore();
  }
}

// ── renderAllGroups ────────────────────────────────────────────────────────────

/**
 * Full render pass: paints all mask groups onto a canvas in order.
 * 1. Draws the source image
 * 2. Applies mask fills via ImageData (one putImageData pass for all groups)
 * 3. Draws bounding boxes and labels on top
 * 4. NEW: draws scene graph relationship edges if showEdges is true
 */
// CHANGED: added edges and showEdges params
export function renderAllGroups(
  ctx:       CanvasRenderingContext2D,
  canvas:    HTMLCanvasElement,
  img:       HTMLImageElement,
  groups:    MaskGroup[],
  showMasks: boolean,
  edges:     Edge[]    = [],
  showEdges: boolean   = false,
): void {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  if (!showMasks || groups.length === 0) {
    // Still draw edges even when masks are hidden, if requested
    if (showEdges && edges.length > 0) {
      drawEdges(ctx, groups, edges);
    }
    return;
  }

  // Paint mask fills
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  for (const group of groups) {
    for (const instance of group.instances) {
      const decoded = decodeRleMask(instance.maskRle, canvas.width, canvas.height);
      paintMask(imageData, decoded, group.color);
    }
  }
  ctx.putImageData(imageData, 0, 0);

  // Draw bounding boxes and labels
  for (const group of groups) {
    for (const instance of group.instances) {
      drawBoundingBox(ctx, instance.boundingBox, group.label, group.color, instance.confidence);
    }
  }

  // NEW: draw relationship edges on top of everything
  if (showEdges && edges.length > 0) {
    drawEdges(ctx, groups, edges);
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number, y: number,
  w: number, h: number,
  r: number,
): void {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}