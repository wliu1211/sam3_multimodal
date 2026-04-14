import type { MaskGroup, DecodedMask, BoundingBox, Edge, SceneGraphNode } from "../types/mask";
import { hexToRgba, darkenHex } from "./colorUtils";
const RELATION_PALETTE = [
    "#60A5FA",
    "#F59E0B",
    "#10B981",
    "#EC4899",
    "#8B5CF6",
    "#F97316",
    "#06B6D4",
    "#84CC16",
];
const MASK_ALPHA = 0.45;
const BORDER_ALPHA = 0.9;
const BORDER_WIDTH = 2;
const LABEL_FONT = "600 12px 'JetBrains Mono', monospace";
const LABEL_PADDING = 4;
const LABEL_OFFSET_Y = -6;
const EDGE_LINE_WIDTH = 1.5;
const EDGE_FONT = "500 11px 'JetBrains Mono', monospace";
const EDGE_LABEL_PAD = 3;
const EDGE_DOT_RADIUS = 3;
export function decodeRleMask(rle: number[], width: number, height: number): DecodedMask {
    const data = new Uint8Array(width * height);
    let pixel = 0;
    let value = 0;
    for (const count of rle) {
        data.fill(value, pixel, pixel + count);
        pixel += count;
        value = value === 0 ? 1 : 0;
    }
    return { data, width, height };
}
export function paintMask(imageData: ImageData, mask: DecodedMask, color: string): void {
    const [r, g, b] = hexToRgba(color, MASK_ALPHA);
    const d = imageData.data;
    for (let i = 0; i < mask.data.length; i++) {
        if (mask.data[i] !== 1)
            continue;
        const px = i * 4;
        const a = MASK_ALPHA;
        d[px] = Math.round(r * a + d[px] * (1 - a));
        d[px + 1] = Math.round(g * a + d[px + 1] * (1 - a));
        d[px + 2] = Math.round(b * a + d[px + 2] * (1 - a));
    }
}
export function drawBoundingBox(ctx: CanvasRenderingContext2D, box: BoundingBox, label: string, color: string, confidence: number): void {
    const borderColor = darkenHex(color, 10);
    const w = box.x2 - box.x1;
    const h = box.y2 - box.y1;
    ctx.strokeStyle = borderColor;
    ctx.globalAlpha = BORDER_ALPHA;
    ctx.lineWidth = BORDER_WIDTH;
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
export function drawEdges(ctx: CanvasRenderingContext2D, groups: MaskGroup[], edges: Edge[], nodes: SceneGraphNode[] = []): void {
    if (edges.length === 0)
        return;
    const centroids = new Map<string, {
        x: number;
        y: number;
    }>();
    for (const group of groups) {
        for (const inst of group.instances) {
            const b = inst.boundingBox;
            centroids.set(inst.id, {
                x: (b.x1 + b.x2) / 2,
                y: (b.y1 + b.y2) / 2,
            });
        }
    }
    for (const node of nodes) {
        if (!centroids.has(node.id)) {
            centroids.set(node.id, {
                x: (node.boundingBox.x1 + node.boundingBox.x2) / 2,
                y: (node.boundingBox.y1 + node.boundingBox.y2) / 2,
            });
        }
    }
    for (const edge of edges) {
        const sub = centroids.get(edge.subjectId);
        const obj = centroids.get(edge.objectId);
        if (!sub || !obj)
            continue;
        const alpha = Math.max(0.25, Math.min(1.0, edge.confidence));
        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.strokeStyle = "rgba(220, 220, 220, 1)";
        ctx.lineWidth = EDGE_LINE_WIDTH;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(sub.x, sub.y);
        ctx.lineTo(obj.x, obj.y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "rgba(220, 220, 220, 1)";
        ctx.beginPath();
        ctx.arc(sub.x, sub.y, EDGE_DOT_RADIUS, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(obj.x, obj.y, EDGE_DOT_RADIUS, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
        const midX = (sub.x + obj.x) / 2;
        const midY = (sub.y + obj.y) / 2;
        const predText = `${edge.downgraded ? "~" : ""}${edge.predicate.replace(/_/g, " ")}`;
        const confText = ` ${Math.round(edge.confidence * 100)}%`;
        const badgeText = predText + confText;
        ctx.font = EDGE_FONT;
        const textW = ctx.measureText(badgeText).width;
        const badgeW = textW + EDGE_LABEL_PAD * 2;
        const badgeH = 16;
        const badgeX = midX - badgeW / 2;
        const badgeY = midY - badgeH / 2;
        ctx.save();
        ctx.globalAlpha = 0.80;
        ctx.fillStyle = "#1a1a2a";
        roundRect(ctx, badgeX, badgeY, badgeW, badgeH, 3);
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.fillStyle = "#e0e0e0";
        ctx.fillText(badgeText, badgeX + EDGE_LABEL_PAD, badgeY + badgeH - EDGE_LABEL_PAD - 2);
        ctx.restore();
    }
}
export function drawRelationMasks(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, nodes: SceneGraphNode[], edges: Edge[]): void {
    if (edges.length === 0 || nodes.length === 0)
        return;
    const nodeMap = new Map<string, SceneGraphNode>();
    for (const node of nodes)
        nodeMap.set(node.id, node);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    edges.forEach((edge, i) => {
        const color = RELATION_PALETTE[i % RELATION_PALETTE.length];
        const subject = nodeMap.get(edge.subjectId);
        const obj = nodeMap.get(edge.objectId);
        if (subject) {
            paintMask(imageData, decodeRleMask(subject.maskRle, canvas.width, canvas.height), color);
        }
        if (obj) {
            paintMask(imageData, decodeRleMask(obj.maskRle, canvas.width, canvas.height), color);
        }
    });
    ctx.putImageData(imageData, 0, 0);
    edges.forEach((edge, i) => {
        const color = RELATION_PALETTE[i % RELATION_PALETTE.length];
        const subject = nodeMap.get(edge.subjectId);
        const obj = nodeMap.get(edge.objectId);
        if (subject)
            drawBoundingBox(ctx, subject.boundingBox, subject.label, color, subject.confidence);
        if (obj)
            drawBoundingBox(ctx, obj.boundingBox, obj.label, color, obj.confidence);
    });
}
export function renderAllGroups(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, img: HTMLImageElement, groups: MaskGroup[], showMasks: boolean, edges: Edge[] = [], showEdges: boolean = false, nodes: SceneGraphNode[] = [], showRelationMasks: boolean = false): void {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    if (showRelationMasks && edges.length > 0 && nodes.length > 0) {
        drawRelationMasks(ctx, canvas, nodes, edges);
    }
    else if (showMasks && groups.length > 0) {
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        for (const group of groups) {
            for (const instance of group.instances) {
                paintMask(imageData, decodeRleMask(instance.maskRle, canvas.width, canvas.height), group.color);
            }
        }
        ctx.putImageData(imageData, 0, 0);
        for (const group of groups) {
            for (const instance of group.instances) {
                drawBoundingBox(ctx, instance.boundingBox, group.label, group.color, instance.confidence);
            }
        }
    }
    if (showEdges && edges.length > 0) {
        drawEdges(ctx, groups, edges, nodes);
    }
}
function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number): void {
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
