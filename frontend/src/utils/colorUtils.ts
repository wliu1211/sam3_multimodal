const GROUP_PALETTE: string[] = [
    "#4ADE80",
    "#F87171",
    "#60A5FA",
    "#FBBF24",
    "#E879F9",
    "#34D399",
    "#FB923C",
    "#A78BFA",
    "#F472B6",
    "#38BDF8",
];
export function getGroupColor(index: number): string {
    return GROUP_PALETTE[index % GROUP_PALETTE.length];
}
export function hexToRgba(hex: string, alpha: number): [
    number,
    number,
    number,
    number
] {
    const clean = hex.replace("#", "");
    const r = parseInt(clean.slice(0, 2), 16);
    const g = parseInt(clean.slice(2, 4), 16);
    const b = parseInt(clean.slice(4, 6), 16);
    return [r, g, b, Math.round(alpha * 255)];
}
export function hexToRgbaString(hex: string, alpha: number): string {
    const [r, g, b] = hexToRgba(hex, alpha);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
export function darkenHex(hex: string, percent: number): string {
    const clean = hex.replace("#", "");
    const factor = 1 - percent / 100;
    const r = Math.round(parseInt(clean.slice(0, 2), 16) * factor);
    const g = Math.round(parseInt(clean.slice(2, 4), 16) * factor);
    const b = Math.round(parseInt(clean.slice(4, 6), 16) * factor);
    return `#${[r, g, b].map((v) => v.toString(16).padStart(2, "0")).join("")}`;
}
