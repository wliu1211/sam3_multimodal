"""
backend/services/vtt.py

Visual Turing Test (VTT) — verifies predicted edges from the SGG
using three independent checks, then either confirms or downgrades
each relationship.

Three check types:
  1. check_containment  — attribute mask must be ≥ CONTAINMENT_THRESHOLD
                          inside its parent node's mask
  2. check_geometry     — predicate-specific spatial rules using bounding
                          boxes (no pixel access needed, very fast)
  3. claude_spot_check  — Claude Vision binary yes/no for ambiguous edges
                          only (confidence in ambiguous zone)

Predicate downgrade chain:
  holding      → carrying → playing_with → near
  wearing      → near
  sitting_on   → on_top_of → near
  standing_on  → near
  on_top_of    → near
  walking      → near
  carrying     → near
  playing_with → near
  parked_next_to → near
  near         → (no downgrade — terminal predicate)
"""

import io
import os
import base64
import numpy as np
from anthropic import AsyncAnthropic
from services.graph import Node, Edge, SceneGraph

# ── Client ────────────────────────────────────────────────────────────────────

_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Config ────────────────────────────────────────────────────────────────────

# Attribute mask must overlap parent mask by at least this fraction
# for the containment check to pass.
CONTAINMENT_THRESHOLD = 0.90

# SGG confidence range that triggers Claude Vision spot-check.
# Below AMBIGUOUS_LOW → reject edge entirely.
# Above AMBIGUOUS_HIGH → accept without spot-check.
AMBIGUOUS_LOW  = 0.35
AMBIGUOUS_HIGH = 0.70

# Predicate downgrade chains — key → next predicate if check fails.
DOWNGRADE_CHAIN: dict[str, str | None] = {
    "holding":        "carrying",
    "carrying":       "playing_with",
    "playing_with":   "near",
    "wearing":        "near",
    "sitting_on":     "on_top_of",
    "on_top_of":      "near",
    "standing_on":    "near",
    "walking":        "near",
    "parked_next_to": "near",
    "near":           None,   # terminal — never downgrade further
}

# Geometry check parameters per predicate
_GEO_RULES: dict[str, dict] = {
    "on_top_of":      {"min_above": 0.20, "min_horiz_overlap": 0.05},
    "sitting_on":     {"min_above": 0.10, "min_horiz_overlap": 0.02},
    "standing_on":    {"max_base_gap_fraction": 0.08},
    "holding":        {"max_norm_dist": 0.22, "min_size_ratio": 2.5},
    "wearing":        {"min_iou": 0.08,  "min_size_ratio": 2.0},
    "carrying":       {"max_norm_dist": 0.18, "min_size_ratio": 2.0},
    "playing_with":   {"max_norm_dist": 0.28},
    "walking":        {"max_norm_dist": 0.32},
    "parked_next_to": {"max_norm_dist": 0.28, "max_vert_offset": 0.15},
    "near":           {},     # near always passes geometry check
}


# ── Main entry point ──────────────────────────────────────────────────────────

async def verify_edges(
    graph:      SceneGraph,
    image_pil:  "PIL.Image.Image",
    img_w:      int,
    img_h:      int,
) -> SceneGraph:
    """
    Runs VTT verification on every edge in the scene graph.

    For each edge:
      1. Geometry check — predicate-specific spatial rules
      2. If geometry passes and confidence is ambiguous → Claude spot-check
      3. If check fails → downgrade predicate or remove edge

    Modifies edges in-place and returns the updated graph.
    Also runs containment check on attribute nodes.
    """
    img_diag = (img_w ** 2 + img_h ** 2) ** 0.5

    # ── Step 1: Containment check for attribute nodes ─────────────────────────
    _verify_attribute_containment(graph, img_w, img_h)

    # ── Step 2: Verify each edge ──────────────────────────────────────────────
    verified_edges = []
    for edge in graph.edges:
        subject = graph.get_node(edge.subject_id)
        obj     = graph.get_node(edge.object_id)

        if subject is None or obj is None:
            continue

        result_edge = await _verify_single_edge(
            edge, subject, obj, image_pil, img_w, img_h, img_diag
        )

        if result_edge is not None:
            verified_edges.append(result_edge)

    graph.edges = verified_edges
    return graph


# ── Attribute containment check ───────────────────────────────────────────────

def _verify_attribute_containment(
    graph: SceneGraph,
    img_w: int,
    img_h: int,
) -> None:
    """
    For each attribute node (e.g. "red shirt", "banana"), verify that
    its mask is at least CONTAINMENT_THRESHOLD contained within its
    parent node's mask.

    Attribute nodes that fail this check have their parent_id cleared
    so they are not drawn as part of their claimed parent.
    """
    for attr_node in graph.attribute_nodes():
        if attr_node.parent_id is None:
            continue

        parent = graph.get_node(attr_node.parent_id)
        if parent is None:
            attr_node.parent_id = None
            continue

        overlap = _mask_containment_fraction(
            child_rle  = attr_node.mask_rle,
            parent_rle = parent.mask_rle,
            img_w      = img_w,
            img_h      = img_h,
        )

        if overlap < CONTAINMENT_THRESHOLD:
            # Attribute doesn't belong to this parent — disown it
            attr_node.parent_id = None


# ── Single edge verification ──────────────────────────────────────────────────

async def _verify_single_edge(
    edge:      Edge,
    subject:   Node,
    obj:       Node,
    image_pil: "PIL.Image.Image",
    img_w:     int,
    img_h:     int,
    img_diag:  float,
) -> Edge | None:
    """
    Runs geometry check → optional Claude spot-check → downgrade or accept.

    Returns the (possibly downgraded) Edge, or None if the edge should
    be removed entirely (confidence too low after all checks).
    """
    predicate  = edge.predicate
    confidence = edge.confidence

    # Reject edges below the ambiguous zone immediately
    if confidence < AMBIGUOUS_LOW:
        return None

    # ── Geometry check ────────────────────────────────────────────────────────
    geo_pass = _check_geometry(subject, obj, predicate, img_w, img_h, img_diag)

    if not geo_pass:
        # Try downgrading
        downgraded = _downgrade(edge, subject, obj, img_w, img_h, img_diag)
        return downgraded

    # ── Claude spot-check for ambiguous confidence ────────────────────────────
    if AMBIGUOUS_LOW <= confidence <= AMBIGUOUS_HIGH:
        confirmed = await _claude_spot_check(
            subject, obj, predicate, image_pil, img_w, img_h
        )
        if not confirmed:
            downgraded = _downgrade(edge, subject, obj, img_w, img_h, img_diag)
            return downgraded

    # ── Edge passes all checks ────────────────────────────────────────────────
    edge.verified = True
    return edge


# ── Geometry check ────────────────────────────────────────────────────────────

def _check_geometry(
    subject:   Node,
    obj:       Node,
    predicate: str,
    img_w:     int,
    img_h:     int,
    img_diag:  float,
) -> bool:
    """
    Runs predicate-specific geometric rules on bounding boxes.
    Returns True if the predicate is geometrically plausible.
    """
    rules = _GEO_RULES.get(predicate, {})
    if not rules:
        return True  # "near" and unknown predicates always pass

    iou        = subject.overlap_ratio(obj)
    dist       = subject.distance_to(obj)
    norm_dist  = dist / img_diag
    area_sub   = subject.box_area
    area_obj   = obj.box_area
    size_ratio = area_sub / area_obj if area_obj > 0 else 1.0
    above      = subject.vertical_position_relative_to(obj)

    # ── on_top_of / sitting_on ────────────────────────────────────────────────
    if "min_above" in rules:
        if above < rules["min_above"]:
            return False
    if "min_horiz_overlap" in rules:
        # Check horizontal overlap separately from IoU
        sx1, sx2 = subject.bounding_box["x1"], subject.bounding_box["x2"]
        ox1, ox2 = obj.bounding_box["x1"],     obj.bounding_box["x2"]
        horiz_overlap = max(0, min(sx2, ox2) - max(sx1, ox1))
        horiz_span    = max(sx2, ox2) - min(sx1, ox1)
        horiz_ratio   = horiz_overlap / horiz_span if horiz_span > 0 else 0
        if horiz_ratio < rules["min_horiz_overlap"]:
            return False

    # ── standing_on ───────────────────────────────────────────────────────────
    if "max_base_gap_fraction" in rules:
        a_base = subject.bounding_box["y2"]
        b_top  = obj.bounding_box["y1"]
        gap_fraction = abs(a_base - b_top) / (img_h + 1)
        if gap_fraction > rules["max_base_gap_fraction"]:
            return False

    # ── Distance rules ────────────────────────────────────────────────────────
    if "max_norm_dist" in rules:
        if norm_dist > rules["max_norm_dist"]:
            return False

    # ── Size ratio rules ──────────────────────────────────────────────────────
    if "min_size_ratio" in rules:
        if size_ratio < rules["min_size_ratio"]:
            return False

    # ── IoU rules ─────────────────────────────────────────────────────────────
    if "min_iou" in rules:
        if iou < rules["min_iou"]:
            return False

    # ── Vertical offset rules (parked_next_to) ────────────────────────────────
    if "max_vert_offset" in rules:
        if abs(above) > rules["max_vert_offset"]:
            return False

    return True


# ── Predicate downgrading ─────────────────────────────────────────────────────

def _downgrade(
    edge:     Edge,
    subject:  Node,
    obj:      Node,
    img_w:    int,
    img_h:    int,
    img_diag: float,
) -> Edge | None:
    """
    Attempts to downgrade the edge predicate to the next in the chain.
    Recursively checks geometry for the new predicate.
    Returns None if the predicate chain is exhausted.
    """
    current     = edge.predicate
    next_pred   = DOWNGRADE_CHAIN.get(current)

    if next_pred is None:
        return None  # Terminal — no further downgrade possible

    # Check geometry for the downgraded predicate
    geo_pass = _check_geometry(
        subject, obj, next_pred, img_w, img_h, img_diag
    )

    if not geo_pass:
        # Try downgrading again
        edge.predicate = next_pred
        return _downgrade(edge, subject, obj, img_w, img_h, img_diag)

    # Downgrade succeeds
    original = edge.original_predicate or edge.predicate
    edge.original_predicate = original
    edge.predicate          = next_pred
    edge.downgraded         = True
    edge.verified           = True
    # Reduce confidence to reflect the downgrade
    edge.confidence         = round(edge.confidence * 0.75, 4)
    return edge


# ── Claude Vision spot-check ──────────────────────────────────────────────────

async def _claude_spot_check(
    subject:   Node,
    obj:       Node,
    predicate: str,
    image_pil: "PIL.Image.Image",
    img_w:     int,
    img_h:     int,
) -> bool:
    """
    Asks Claude a binary yes/no question about the predicted relationship.
    Called only for edges in the ambiguous confidence zone.

    Crops a region around both bounding boxes with padding, encodes
    as base64 JPEG, and sends to Claude with a direct question.

    Returns True if Claude confirms, False otherwise.
    Falls back to True (trust geometry) on any API error.
    """
    pad   = 20
    x1    = max(0,    min(subject.bounding_box["x1"], obj.bounding_box["x1"]) - pad)
    y1    = max(0,    min(subject.bounding_box["y1"], obj.bounding_box["y1"]) - pad)
    x2    = min(img_w, max(subject.bounding_box["x2"], obj.bounding_box["x2"]) + pad)
    y2    = min(img_h, max(subject.bounding_box["y2"], obj.bounding_box["y2"]) + pad)
    crop  = image_pil.crop((x1, y1, x2, y2))

    question = (
        f"In this image, is the {subject.label} {predicate.replace('_', ' ')} "
        f"the {obj.label}? Answer only 'yes' or 'no'."
    )

    buffer = io.BytesIO()
    crop.save(buffer, format="JPEG", quality=80)
    img_b64 = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

    try:
        response = await _client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=5,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type":       "base64",
                            "media_type": "image/jpeg",
                            "data":       img_b64,
                        },
                    },
                    {"type": "text", "text": question},
                ],
            }],
        )
        answer = response.content[0].text.strip().lower()
        return answer.startswith("yes")

    except Exception as e:
        print(f"[vtt] Claude spot-check failed ({subject.label} {predicate} "
              f"{obj.label}): {e} — defaulting to geometry result")
        return True  # Trust geometry on API failure


# ── Mask containment helper ───────────────────────────────────────────────────

def _mask_containment_fraction(
    child_rle:  list[int],
    parent_rle: list[int],
    img_w:      int,
    img_h:      int,
) -> float:
    """
    Returns the fraction of child mask pixels that are inside the parent mask.

    Decodes both RLE masks to flat arrays and computes:
        intersection / child_foreground_count

    Returns 0.0 if the child mask is empty.
    """
    child  = _decode_rle(child_rle,  img_w * img_h)
    parent = _decode_rle(parent_rle, img_w * img_h)

    child_count = np.sum(child)
    if child_count == 0:
        return 0.0

    intersection = np.sum(child & parent)
    return float(intersection / child_count)


def _decode_rle(rle: list[int], total_pixels: int) -> np.ndarray:
    """Decodes alternating RLE [bg_count, fg_count, ...] into a flat bool array."""
    flat    = np.zeros(total_pixels, dtype=bool)
    pixel   = 0
    value   = False  # starts with background

    for count in rle:
        if value:
            flat[pixel:pixel + count] = True
        pixel += count
        value  = not value

    return flat