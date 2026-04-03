import io
import uuid
import cv2
import numpy as np
from PIL import Image
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

MASK_ALPHA        = 0.45   # Opacity of the colored mask fill
BORDER_THICKNESS  = 2      # Bounding box line thickness in pixels
LABEL_FONT        = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE       = 0.45
LABEL_THICKNESS   = 1
LABEL_PAD         = 5      # Pixels of padding inside the label badge
LABEL_OFFSET_Y    = 8      # Pixels above the bounding box top edge

# Distinct RGB colors for each group — same palette as frontend colorUtils.ts
# so mask colors in the rendered image match the legend exactly.
GROUP_PALETTE = [
    (74,  222, 128),  # green
    (248, 113, 113),  # red
    (96,  165, 250),  # blue
    (251, 191,  36),  # amber
    (232, 121, 249),  # fuchsia
    (52,  211, 153),  # emerald
    (251, 146,  60),  # orange
    (167, 139, 250),  # violet
    (244, 114, 182),  # pink
    (56,  189, 248),  # sky
]

# Hex versions of the same palette — used in the API response payload
# so the frontend legend colors match the rendered image colors.
GROUP_PALETTE_HEX = [
    "#4ADE80", "#F87171", "#60A5FA", "#FBBF24", "#E879F9",
    "#34D399", "#FB923C", "#A78BFA", "#F472B6", "#38BDF8",
]

# ── Main entry point ──────────────────────────────────────────────────────────

def render_results(
    image_bytes: bytes,
    groups:      list[dict],
    results_dir: Path,
) -> tuple[str, str]:
    """
    Draws colored mask overlays and bounding boxes for all matched instances,
    then saves the annotated image to storage/results/.

    groups format (one entry per attribute group):
    [
        {
            "label":     "blue Honda",
            "color":     "#4ADE80",
            "instances": [
                {
                    "id":           "0001",
                    "bounding_box": {x1, y1, x2, y2},
                    "mask_rle":     [int, ...],
                    "confidence":   0.82,
                }
            ]
        }
    ]

    Returns (filename, full_path_str).
    filename is stored in the DB and used to build the result image URL.
    """
    image_np = _bytes_to_bgr(image_bytes)
    h, w     = image_np.shape[:2]
    output   = image_np.copy()

    # Paint all mask fills onto an overlay, then alpha-blend once
    overlay = image_np.copy()
    for group in groups:
        color_bgr = _hex_to_bgr(group["color"])
        for instance in group["instances"]:
            mask = _decode_mask_rle(instance["mask_rle"], w, h)
            overlay[mask == 1] = color_bgr

    cv2.addWeighted(overlay, MASK_ALPHA, output, 1 - MASK_ALPHA, 0, output)

    # Draw bounding boxes and labels on top of the blended masks
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

    # Save annotated image to storage/results/
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

    # Box stroke
    cv2.rectangle(image, (x1, y1), (x2, y2), color_bgr, BORDER_THICKNESS, cv2.LINE_AA)

    # Label text
    badge_text = f"{label} {int(confidence * 100)}%"
    (text_w, text_h), baseline = cv2.getTextSize(
        badge_text, LABEL_FONT, LABEL_SCALE, LABEL_THICKNESS
    )

    # Badge background
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
    return (b, g, r)  # OpenCV is BGR


# ── Group payload builder ─────────────────────────────────────────────────────

def build_groups_payload(
    parsed_query: dict,
    instances:    list[dict],
) -> list[dict]:
    """
    Assembles the groups list returned in the API response and
    passed into render_results().

    Builds a human-readable label from parsed attributes and filters
    instances down to matched ones only. Assigns colors from GROUP_PALETTE_HEX
    so the API response hex color always matches what was painted on the image.

    Extension point for multi-group queries: split instances by matched
    attribute set here and create one group dict per set.
    """
    attributes = parsed_query["attributes"]
    base_class = parsed_query["base_class"]

    attr_parts = [
        v for v in [
            attributes.get("color"),
            attributes.get("make"),
            attributes.get("type"),
        ] if v
    ] + attributes.get("other", [])

    label   = " ".join(attr_parts + [base_class]) if attr_parts else base_class
    matched = [inst for inst in instances if inst.get("matched")]

    avg_confidence = (
        round(sum(i["confidence"] for i in matched) / len(matched), 4)
        if matched else 0.0
    )

    return [
        {
            "label":         label,
            "color":         GROUP_PALETTE_HEX[0],
            "attributes":    attr_parts,
            "instances":     matched,
            "matchCount":    len(matched),
            "avgConfidence": avg_confidence,
        }
    ]