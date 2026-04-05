import io
import uuid
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional

# ── NEW: SceneGraph import ────────────────────────────────────────────────────
from services.graph import SceneGraph

# ── Constants ─────────────────────────────────────────────────────────────────

MASK_ALPHA        = 0.45
BORDER_THICKNESS  = 2
LABEL_FONT        = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE       = 0.45
LABEL_THICKNESS   = 1
LABEL_PAD         = 5
LABEL_OFFSET_Y    = 8

# Edge drawing constants
EDGE_THICKNESS    = 2       # Relationship line thickness
EDGE_LABEL_SCALE  = 0.38    # Smaller font for edge labels
EDGE_LABEL_PAD    = 3

GROUP_PALETTE = [
    (74,  222, 128),
    (248, 113, 113),
    (96,  165, 250),
    (251, 191,  36),
    (232, 121, 249),
    (52,  211, 153),
    (251, 146,  60),
    (167, 139, 250),
    (244, 114, 182),
    (56,  189, 248),
]

GROUP_PALETTE_HEX = [
    "#4ADE80", "#F87171", "#60A5FA", "#FBBF24", "#E879F9",
    "#34D399", "#FB923C", "#A78BFA", "#F472B6", "#38BDF8",
]

# Color used for relationship edge lines — white with alpha blending
EDGE_COLOR_BGR = (220, 220, 220)

# ── Main entry point ──────────────────────────────────────────────────────────

# CHANGED: added optional scene_graph parameter
def render_results(
    image_bytes:  bytes,
    groups:       list[dict],
    results_dir:  Path,
    scene_graph:  Optional[SceneGraph] = None,
) -> tuple[str, str]:
    """
    Draws colored mask overlays, bounding boxes, and (if provided)
    scene graph relationship edges onto the source image, then saves to disk.

    Returns (filename, full_path_str).
    """
    image_np = _bytes_to_bgr(image_bytes)
    h, w     = image_np.shape[:2]
    output   = image_np.copy()

    # Paint all mask fills in a single alpha-blend pass
    overlay = image_np.copy()
    for group in groups:
        color_bgr = _hex_to_bgr(group["color"])
        for instance in group["instances"]:
            mask = _decode_mask_rle(instance["mask_rle"], w, h)
            overlay[mask == 1] = color_bgr

    cv2.addWeighted(overlay, MASK_ALPHA, output, 1 - MASK_ALPHA, 0, output)

    # Draw bounding boxes and instance labels
    for group in groups:
        color_bgr = _hex_to_bgr(group["color"])
        for instance in group["instances"]:
            _draw_bounding_box(
                output,
                instance["bounding_box"],
                group["label"],
                instance["confidence"],
                color_bgr,
            )

    # ── NEW: draw scene graph edges if provided ───────────────────────────────
    if scene_graph is not None and scene_graph.verified_edges():
        _draw_edges(output, scene_graph)

    filename  = f"{uuid.uuid4().hex}.jpg"
    save_path = results_dir / filename
    cv2.imwrite(str(save_path), output, [cv2.IMWRITE_JPEG_QUALITY, 92])

    return filename, str(save_path)


# ── Drawing ───────────────────────────────────────────────────────────────────

def _draw_bounding_box(
    image:      np.ndarray,
    box:        dict,
    label:      str,
    confidence: float,
    color_bgr:  tuple[int, int, int],
) -> None:
    """Draws a bounding box stroke and label badge onto the image in-place."""
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

    cv2.rectangle(image, (x1, y1), (x2, y2), color_bgr, BORDER_THICKNESS, cv2.LINE_AA)

    badge_text = f"{label} {int(confidence * 100)}%"
    (text_w, text_h), baseline = cv2.getTextSize(
        badge_text, LABEL_FONT, LABEL_SCALE, LABEL_THICKNESS
    )

    badge_x1 = x1
    badge_y1 = max(0, y1 - text_h - LABEL_PAD * 2 - LABEL_OFFSET_Y)
    badge_x2 = x1 + text_w + LABEL_PAD * 2
    badge_y2 = max(text_h + LABEL_PAD, y1 - LABEL_OFFSET_Y)
    dark_bgr  = tuple(max(0, int(c * 0.7)) for c in color_bgr)

    cv2.rectangle(image, (badge_x1, badge_y1), (badge_x2, badge_y2), dark_bgr, -1)

    cv2.putText(
        image,
        badge_text,
        (badge_x1 + LABEL_PAD, badge_y2 - LABEL_PAD - baseline),
        LABEL_FONT,
        LABEL_SCALE,
        (255, 255, 255),
        LABEL_THICKNESS,
        cv2.LINE_AA,
    )


# ── NEW: edge drawing ─────────────────────────────────────────────────────────

def _draw_edges(image: np.ndarray, scene_graph: SceneGraph) -> None:
    """
    Draws a line between the centroids of each verified edge's subject and
    object, with a predicate label badge at the midpoint.

    Line opacity is scaled by edge confidence — higher confidence edges
    appear more opaque and therefore more prominent.

    Only draws verified edges to avoid cluttering the image with uncertain
    relationship proposals.
    """
    h, w = image.shape[:2]

    for edge in scene_graph.verified_edges():
        subject = scene_graph.get_node(edge.subject_id)
        obj     = scene_graph.get_node(edge.object_id)

        if subject is None or obj is None:
            continue

        # Centroid coordinates
        sx, sy = (int(c) for c in subject.centroid)
        ox, oy = (int(c) for c in obj.centroid)

        # Scale line opacity by confidence
        alpha      = max(0.3, min(1.0, edge.confidence))
        line_color = tuple(int(c * alpha) for c in EDGE_COLOR_BGR)

        # Draw the relationship line
        cv2.line(image, (sx, sy), (ox, oy), line_color, EDGE_THICKNESS, cv2.LINE_AA)

        # Draw small circles at each endpoint to anchor the line visually
        cv2.circle(image, (sx, sy), 4, line_color, -1, cv2.LINE_AA)
        cv2.circle(image, (ox, oy), 4, line_color, -1, cv2.LINE_AA)

        # Label badge at midpoint
        mid_x = (sx + ox) // 2
        mid_y = (sy + oy) // 2

        # Use arrow symbol if downgraded to show the original was stronger
        predicate_text = edge.predicate.replace("_", " ")
        if edge.downgraded and edge.original_predicate:
            predicate_text = f"~{predicate_text}"

        conf_text  = f"{int(edge.confidence * 100)}%"
        badge_text = f"{predicate_text} {conf_text}"

        (text_w, text_h), baseline = cv2.getTextSize(
            badge_text, LABEL_FONT, EDGE_LABEL_SCALE, LABEL_THICKNESS
        )

        bx1 = max(0,     mid_x - text_w // 2 - EDGE_LABEL_PAD)
        by1 = max(0,     mid_y - text_h - EDGE_LABEL_PAD * 2)
        bx2 = min(w - 1, mid_x + text_w // 2 + EDGE_LABEL_PAD)
        by2 = min(h - 1, mid_y + EDGE_LABEL_PAD)

        # Dark semi-transparent badge background
        badge_overlay = image.copy()
        cv2.rectangle(badge_overlay, (bx1, by1), (bx2, by2), (30, 30, 30), -1)
        cv2.addWeighted(badge_overlay, 0.75, image, 0.25, 0, image)

        cv2.putText(
            image,
            badge_text,
            (bx1 + EDGE_LABEL_PAD, by2 - EDGE_LABEL_PAD - baseline + text_h),
            LABEL_FONT,
            EDGE_LABEL_SCALE,
            (240, 240, 240),
            LABEL_THICKNESS,
            cv2.LINE_AA,
        )


# ── Encode / decode ───────────────────────────────────────────────────────────

def _decode_mask_rle(rle: list[int], width: int, height: int) -> np.ndarray:
    """
    Decodes a run-length encoded mask into a (H, W) uint8 binary array.
    Mirrors vision.py's _encode_mask_rle() — must stay in sync.
    """
    flat    = np.zeros(width * height, dtype=np.uint8)
    pixel   = 0
    value   = 0

    for count in rle:
        flat[pixel:pixel + count] = value
        pixel += count
        value  = 1 - value

    return flat.reshape((height, width))


def _bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    """Decodes raw image bytes to an OpenCV BGR numpy array."""
    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image. Ensure it is a valid JPEG, PNG, or WebP.")
    return image


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """Converts '#4ADE80' → (128, 222, 74) — RGB flipped to BGR for OpenCV."""
    clean = hex_color.lstrip("#")
    r = int(clean[0:2], 16)
    g = int(clean[2:4], 16)
    b = int(clean[4:6], 16)
    return (b, g, r)


# ── Group payload builder ─────────────────────────────────────────────────────

def build_groups_payload(
    parsed_query: dict,
    instances:    list[dict],
) -> list[dict]:
    """
    Assembles one group per target in parsed_query["targets"].

    Each matched instance carries a "target_label" set by vision.py that
    identifies which target it was best matched to. Groups are ordered so
    primary targets come first (preserving palette assignment).
    """
    targets = parsed_query["targets"]

    # Index matched instances by their target_label for O(1) lookup
    label_to_insts: dict[str, list[dict]] = {}
    for inst in instances:
        if inst.get("matched"):
            lbl = inst.get("target_label", inst.get("base_class", "object"))
            label_to_insts.setdefault(lbl, []).append(inst)

    groups: list[dict] = []
    for i, tgt in enumerate(targets):
        base_class = tgt["base_class"]
        attrs      = tgt["attributes"]

        attr_parts = [
            v for v in [attrs.get("color"), attrs.get("make"), attrs.get("type")]
            if v
        ] + attrs.get("other", [])

        label   = " ".join(attr_parts + [base_class]) if attr_parts else base_class
        matched = label_to_insts.get(label, [])

        avg_confidence = (
            round(sum(inst["confidence"] for inst in matched) / len(matched), 4)
            if matched else 0.0
        )

        groups.append({
            "label":         label,
            "color":         GROUP_PALETTE_HEX[i % len(GROUP_PALETTE_HEX)],
            "attributes":    attr_parts,
            "role":          tgt["role"],
            "instances":     matched,
            "matchCount":    len(matched),
            "avgConfidence": avg_confidence,
        })

    return groups