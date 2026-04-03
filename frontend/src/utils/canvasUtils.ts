import type { MaskGroup, DecodedMask, BoundingBox } from "../types/mask";
import { hexToRgba, darkenHex } from "./colorUtils";

const MASK_ALPHA       = 0.45;  // Opacity of the filled mask overlay
const BORDER_ALPHA     = 0.9;   // Opacity of the bounding box stroke
const BORDER_WIDTH     = 2;     // Bounding box line width in px
const LABEL_FONT       = "600 12px 'JetBrains Mono', monospace";
const LABEL_PADDING    = 4;     // px around label text
const LABEL_OFFSET_Y   = -6;    // px above the bounding box top edge

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
    // Alpha-blend: out = src * alpha + dst * (1 - alpha)
    const a = MASK_ALPHA;
    d[px]     = Math.round(r * a + d[px]     * (1 - a));
    d[px + 1] = Math.round(g * a + d[px + 1] * (1 - a));
    d[px + 2] = Math.round(b * a + d[px + 2] * (1 - a));
    // Leave alpha channel (d[px+3]) untouched
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

  // Box stroke
  ctx.strokeStyle = borderColor;
  ctx.globalAlpha = BORDER_ALPHA;
  ctx.lineWidth   = BORDER_WIDTH;
  ctx.strokeRect(box.x1, box.y1, w, h);
  ctx.globalAlpha = 1;

  // Label badge background
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

  // Label text
  ctx.fillStyle = "#ffffff";
  ctx.fillText(badgeText, badgeX + LABEL_PADDING, badgeY + badgeH - LABEL_PADDING - 1);
}

/**
 * Full render pass: paints all mask groups onto a canvas in order.
 * 1. Draws the source image
 * 2. Applies mask fills via ImageData (one putImageData per group)
 * 3. Draws bounding boxes and labels on top
 */
export function renderAllGroups(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  img: HTMLImageElement,
  groups: MaskGroup[],
  showMasks: boolean,
): void {
  // Clear and draw base image
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  if (!showMasks || groups.length === 0) return;

  // Paint mask fills
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  for (const group of groups) {
    for (const instance of group.instances) {
      const decoded = decodeRleMask(instance.maskRle, canvas.width, canvas.height);
      paintMask(imageData, decoded, group.color);
    }
  }
  ctx.putImageData(imageData, 0, 0);

  // Draw bounding boxes and labels on top of masks
  for (const group of groups) {
    for (const instance of group.instances) {
      drawBoundingBox(ctx, instance.boundingBox, group.label, group.color, instance.confidence);
    }
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