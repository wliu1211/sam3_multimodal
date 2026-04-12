import io
import uuid
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.graph import SceneGraph

# ── Constants ─────────────────────────────────────────────────────────────────

MASK_ALPHA        = 0.45
BORDER_THICKNESS  = 2
LABEL_FONT        = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE       = 0.45
LABEL_THICKNESS   = 1
LABEL_PAD         = 5
LABEL_OFFSET_Y    = 8

# For compositional queries (e.g. "man with paddle and umbrella"), a related
# object must be within this fraction of the image diagonal to be considered
# a companion of a given primary instance.
COMPOSITIONAL_MAX_DIST_FRACTION = 0.45

# Tighter distance threshold used as fallback when a strong relational hint
# (e.g. "holding") was given but the scene graph produced no qualifying
# verified edge.  Avoids accepting distant "near" pairs as holding companions.
COMPOSITIONAL_FALLBACK_DIST_FRACTION = 0.25

# Predicate vocabulary for expanding relational hints.
# Maps each normalised predicate → acceptable predicate set (including
# semantically similar / downgraded variants). None = accept any predicate.
_HINT_TO_ALLOWED: dict[str, set[str] | None] = {
    "holding":        {"holding", "carrying"},
    "carrying":       {"carrying", "holding"},
    "wearing":        {"wearing"},
    "riding":         {"riding", "on_top_of"},
    "sitting_on":     {"sitting_on", "on_top_of"},
    "standing_on":    {"standing_on", "on_top_of"},
    "on_top_of":      {"on_top_of", "sitting_on", "standing_on", "riding"},
    "walking":        {"walking", "near"},
    "playing_with":   {"playing_with", "holding", "carrying"},
    "parked_next_to": {"parked_next_to", "near"},
    "near":           None,   # weakest hint — any predicate qualifies
}

# Predicates strong enough to warrant edge-based matching (i.e. pure
# centroid-distance is not a reliable proxy for "holding" or "wearing").
_STRONG_PREDICATES: set[str] = {
    "holding", "carrying", "wearing",
    "riding", "sitting_on", "standing_on", "on_top_of", "playing_with",
}

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

# ── Main entry point ──────────────────────────────────────────────────────────

def render_results(
    image_bytes: bytes,
    groups:      list[dict],
    results_dir: Path,
) -> tuple[str, str]:
    """
    Draws colored mask overlays and bounding boxes onto the source image,
    then saves to disk.

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

def _centroid_dist(box_a: dict, box_b: dict) -> float:
    """Euclidean pixel distance between the centroids of two bounding boxes."""
    cx_a = (box_a["x1"] + box_a["x2"]) / 2
    cy_a = (box_a["y1"] + box_a["y2"]) / 2
    cx_b = (box_b["x1"] + box_b["x2"]) / 2
    cy_b = (box_b["y1"] + box_b["y2"]) / 2
    return ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5


def filter_compositional_instances(
    instances:    list[dict],
    parsed_query: dict,
    img_w:        int,
    img_h:        int,
    scene_graph:  "SceneGraph | None" = None,
) -> list[dict]:
    """
    Compositional set filtering for multi-class queries.

    For a query like "man holding racket", a detected man is kept only when
    there is a verified scene-graph edge (holding/carrying) from that man to
    a racket instance.  When no qualifying edge exists the filter gracefully
    falls back to centroid-distance, but with a tighter threshold than the
    pure-distance path so that distant coincidences are not accepted.

    Single-class queries (e.g. "persons") have no related targets and pass
    through unchanged.  Related instances are not exclusive: a companion can
    satisfy multiple primaries simultaneously.

    Priority of matching for each (primary, related-class) pair:
      1. Verified scene-graph edge whose predicate is in the allowed set
         derived from the query's relational hints.
      2. Nearest-centroid fallback within COMPOSITIONAL_FALLBACK_DIST_FRACTION
         of the image diagonal (when hints name a strong predicate and no
         qualifying edge was found).
      3. Nearest-centroid fallback within COMPOSITIONAL_MAX_DIST_FRACTION
         (when no strong hints are present — "near", no hints, etc.).
    """
    from collections import defaultdict
    from services.sgg import normalize_hints

    targets         = parsed_query["targets"]
    primary_classes = {t["base_class"] for t in targets if t["role"] == "primary"}
    related_classes = {t["base_class"] for t in targets if t["role"] == "related"}

    if not related_classes:
        return instances

    # ── Normalise relational hints → set of required predicates ──────────────
    raw_hints  = parsed_query.get("relational_hints", [])
    norm_hints = normalize_hints(raw_hints) if raw_hints else []

    # Build union of allowed predicates from all hints.
    # If any hint maps to None (e.g. "near"), every predicate is acceptable.
    allowed_predicates: set[str] | None = set()
    for pred in norm_hints:
        expansion = _HINT_TO_ALLOWED.get(pred)
        if expansion is None:
            allowed_predicates = None   # "near" → accept anything
            break
        allowed_predicates.update(expansion)  # type: ignore[union-attr]
    if not norm_hints:
        allowed_predicates = None              # no hints → any predicate ok

    # Edge-based matching is only meaningful for strong spatial predicates
    # (holding, wearing, etc.).  "near" and absent hints fall back to distance.
    has_strong_hints = bool(norm_hints and set(norm_hints) & _STRONG_PREDICATES)

    # ── Build edge index from verified scene-graph edges ─────────────────────
    # index: instance_id → [(other_id, predicate, confidence), ...]
    edge_index: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    if scene_graph is not None:
        for edge in scene_graph.verified_edges():
            edge_index[edge.subject_id].append(
                (edge.object_id, edge.predicate, edge.confidence)
            )
            edge_index[edge.object_id].append(
                (edge.subject_id, edge.predicate, edge.confidence)
            )

    img_diag      = (img_w ** 2 + img_h ** 2) ** 0.5
    max_dist      = COMPOSITIONAL_MAX_DIST_FRACTION * img_diag
    fallback_dist = COMPOSITIONAL_FALLBACK_DIST_FRACTION * img_diag

    matched             = [i for i in instances if i.get("matched")]
    primaries           = [i for i in matched if i["base_class"] in primary_classes]
    related_by_class: dict[str, list[dict]] = defaultdict(list)
    for inst in matched:
        if inst["base_class"] in related_classes:
            related_by_class[inst["base_class"]].append(inst)

    id_to_instance: dict[str, dict] = {inst["id"]: inst for inst in matched}
    valid_ids: set[str] = set()

    for primary in primaries:
        companions: dict[str, dict] = {}

        for rel_class in related_classes:
            pool = related_by_class[rel_class]
            if not pool:
                break

            companion: dict | None = None

            # ── 1. Edge-based matching ────────────────────────────────────────
            # Only attempt when the query carries a strong predicate hint
            # (e.g. "holding") and the scene graph has verified edges.
            if has_strong_hints and edge_index:
                pool_ids  = {inst["id"] for inst in pool}
                best: dict | None = None
                best_conf = -1.0

                for other_id, predicate, conf in edge_index.get(primary["id"], []):
                    if other_id not in pool_ids:
                        continue
                    if (
                        allowed_predicates is not None
                        and predicate not in allowed_predicates
                    ):
                        continue
                    if conf > best_conf:
                        best_conf = conf
                        best      = id_to_instance.get(other_id)

                companion = best

            # ── 2. Distance fallback ──────────────────────────────────────────
            # For strong predicate hints (holding, wearing, carrying, etc.):
            # skip distance-based fallback when SGG edges are present.
            # The VTT already enforced one-to-one assignment for these
            # predicates, so persons that did NOT receive a verified edge
            # should not qualify via proximity alone — centroid distance
            # is not a reliable proxy for physical contact.
            #
            # If edge_index is completely empty (SGG/VTT pipeline failure),
            # we fall back to the tight fallback_dist as a last resort so
            # the pipeline degrades gracefully rather than returning nothing.
            if companion is None:
                use_distance = (not has_strong_hints) or (not edge_index)
                if use_distance:
                    threshold = fallback_dist if has_strong_hints else max_dist
                    nearest   = min(
                        pool,
                        key=lambda r: _centroid_dist(
                            primary["bounding_box"], r["bounding_box"]
                        ),
                    )
                    if _centroid_dist(primary["bounding_box"], nearest["bounding_box"]) <= threshold:
                        companion = nearest

            if companion is not None:
                companions[rel_class] = companion

        if all(rc in companions for rc in related_classes):
            valid_ids.add(primary["id"])
            for companion in companions.values():
                valid_ids.add(companion["id"])

    return [inst for inst in instances if inst["id"] in valid_ids]


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

    # Index matched instances by their target_label for O(1) lookup.
    # target_label format: "<color> <make> <type> <base_class>" — must match
    # the _make_target_label() function in vision.py exactly.
    label_to_insts: dict[str, list[dict]] = {}
    for inst in instances:
        if inst.get("matched"):
            lbl = inst.get("target_label", inst.get("base_class", "object"))
            label_to_insts.setdefault(lbl, []).append(inst)

    groups: list[dict] = []
    for i, tgt in enumerate(targets):
        base_class = tgt["base_class"]
        attrs      = tgt["attributes"]

 
        structured_parts = [
            v for v in [attrs.get("color"), attrs.get("make"), attrs.get("type"), base_class]
            if v
        ]
        label = " ".join(structured_parts)

        attr_parts = [
            v for v in [attrs.get("color"), attrs.get("make"), attrs.get("type")]
            if v
        ] + list(attrs.get("other") or [])

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