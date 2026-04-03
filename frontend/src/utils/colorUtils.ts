// Distinct, accessible colors for mask overlay groups.
// Chosen to be visually separable and legible at 40% opacity over photos.
const GROUP_PALETTE: string[] = [
    "#4ADE80", // green
    "#F87171", // red
    "#60A5FA", // blue
    "#FBBF24", // amber
    "#E879F9", // fuchsia
    "#34D399", // emerald
    "#FB923C", // orange
    "#A78BFA", // violet
    "#F472B6", // pink
    "#38BDF8", // sky
  ];
  
  /**
   * Returns a stable hex color for a given group index.
   * Wraps around if there are more groups than palette entries.
   */
  export function getGroupColor(index: number): string {
    return GROUP_PALETTE[index % GROUP_PALETTE.length];
  }
  
  /**
   * Converts a hex color string to an RGBA array.
   * Used by the canvas renderer to apply alpha-blended overlays.
   *
   * @param hex   - e.g. "#4ADE80"
   * @param alpha - 0.0 to 1.0
   * @returns     [r, g, b, a] each 0–255
   */
  export function hexToRgba(hex: string, alpha: number): [number, number, number, number] {
    const clean = hex.replace("#", "");
    const r = parseInt(clean.slice(0, 2), 16);
    const g = parseInt(clean.slice(2, 4), 16);
    const b = parseInt(clean.slice(4, 6), 16);
    return [r, g, b, Math.round(alpha * 255)];
  }
  
  /**
   * Generates a CSS rgba() string from a hex color + alpha.
   * Used for UI elements like legend swatches and badges.
   */
  export function hexToRgbaString(hex: string, alpha: number): string {
    const [r, g, b] = hexToRgba(hex, alpha);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }
  
  /**
   * Darkens a hex color by a given percentage (0–100).
   * Used to draw bounding box borders that contrast against the fill.
   */
  export function darkenHex(hex: string, percent: number): string {
    const clean = hex.replace("#", "");
    const factor = 1 - percent / 100;
    const r = Math.round(parseInt(clean.slice(0, 2), 16) * factor);
    const g = Math.round(parseInt(clean.slice(2, 4), 16) * factor);
    const b = Math.round(parseInt(clean.slice(4, 6), 16) * factor);
    return `#${[r, g, b].map((v) => v.toString(16).padStart(2, "0")).join("")}`;
  }