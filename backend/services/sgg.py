"""
backend/services/sgg.py

Scene Graph Generation (SGG) — builds the relational graph between
detected nodes using geometry-based spatial scoring.

No learned model weights are needed. All predicate predictions are
derived from spatial features: relative position, overlap, size ratio,
and vertical alignment. This is robust and fast enough for a demo.

Predicate vocabulary:
  holding      — subject grasps/carries object (close, size ratio ~3:1)
  wearing      — object is partially inside subject (high containment)
  walking      — subject and animal-class object are close and similarly sized
  sitting_on   — subject centroid is above object, overlap exists
  standing_on  — subject base touches object top, low overlap
  on_top_of    — subject is significantly above object, high horizontal overlap
  parked_next_to — two large similar objects side by side
  near         — proximity without clear structural relationship
  carrying     — subject contains object partially, subject much larger
  playing_with — close proximity, small object relative to subject
"""

import numpy as np
from itertools import combinations
from services.graph import Node, Edge, SceneGraph


# ── Config ────────────────────────────────────────────────────────────────────

# Maximum centroid-to-centroid distance (as fraction of image diagonal)
# for a pair to be considered as a candidate relationship.
MAX_DISTANCE_FRACTION = 0.50

# Minimum confidence of a predicate score to produce an Edge.
MIN_PREDICATE_CONFIDENCE = 0.30

# Size ratio threshold — if one object is this many times larger than
# the other, "wearing" / "holding" / "carrying" become possible.
SIZE_RATIO_LARGE = 3.0
SIZE_RATIO_SMALL = 0.3

# Object classes considered "animate" for motion-related predicates.
ANIMATE_CLASSES = {
    "person", "man", "woman", "child", "boy", "girl",
    "monkey", "dog", "cat", "bird", "horse", "elephant",
    "bear", "zebra", "giraffe", "cow", "sheep", "animal",
}

# Object classes considered "large structures" for parked_next_to.
LARGE_STRUCTURE_CLASSES = {
    "car", "truck", "bus", "van", "vehicle", "house",
    "building", "tree", "bench", "table",
}

# Object classes that a person can ride — used to distinguish "riding"
# (person on vehicle/animal) from "sitting_on" (person on a static object).
RIDEABLE_CLASSES = {
    # wheeled vehicles
    "bike", "bicycle", "motorcycle", "motorbike", "scooter",
    "car", "truck", "bus", "van", "vehicle", "moped",
    # ridden animals
    "horse", "donkey", "camel", "elephant",
    # board-style vehicles
    "skateboard", "snowboard", "surfboard", "sled", "boat",
}


# ── Hint normaliser ───────────────────────────────────────────────────────────

# Maps raw parser output (prepositions, verbs, phrases) → SGG predicate names.
# Multiple raw strings can map to the same predicate; the SGG boost logic
# uses set membership so duplicates are harmless.
_HINT_MAP: dict[str, str] = {
    # on / above
    "on":            "on_top_of",
    "on top of":     "on_top_of",
    "on top":        "on_top_of",
    "above":         "on_top_of",
    "over":          "on_top_of",
    "sitting on":    "sitting_on",
    "sitting":       "sitting_on",
    "perched on":    "sitting_on",
    "standing on":   "standing_on",
    "standing":      "standing_on",
    # holding / carrying
    "holding":       "holding",
    "holds":         "holding",
    "grasping":      "holding",
    "carrying":      "carrying",
    "carries":       "carrying",
    "with":          "holding",        # heuristic: "monkey with banana"
    # wearing
    "wearing":       "wearing",
    "wears":         "wearing",
    "dressed in":    "wearing",
    # motion / play
    "walking":       "walking",
    "walking with":  "walking",
    "playing with":  "playing_with",
    "playing":       "playing_with",
    # riding
    "riding":        "riding",
    "rides":         "riding",
    "riding on":     "riding",
    # near / beside
    "near":          "near",
    "next to":       "parked_next_to",
    "beside":        "parked_next_to",
    "parked next to":"parked_next_to",
    "parked beside": "parked_next_to",
    "by":            "near",
    "alongside":     "near",
    "in":            "near",
    "inside":        "near",
    "at":            "near",
}

_KNOWN_PREDICATES = {
    "holding", "wearing", "walking", "sitting_on", "standing_on",
    "on_top_of", "carrying", "playing_with", "parked_next_to", "near",
    "riding",
}


def normalize_hints(raw_hints: list[str]) -> list[str]:
    """
    Converts raw parser hints (plain prepositions / verbs) into the
    predicate vocabulary understood by build_scene_graph().

    Examples:
      ["on", "with", "holding"]  → ["on_top_of", "holding"]
      ["sitting on", "near"]     → ["sitting_on", "near"]
      ["holding"]                → ["holding"]   (already valid, pass-through)
    """
    normalised: set[str] = set()
    for hint in raw_hints:
        h = hint.lower().strip()
        if h in _KNOWN_PREDICATES:
            normalised.add(h)
        elif h in _HINT_MAP:
            normalised.add(_HINT_MAP[h])
        else:
            # Try prefix / substring match for multi-word phrases
            for phrase, pred in _HINT_MAP.items():
                if phrase in h or h in phrase:
                    normalised.add(pred)
                    break
    return list(normalised)


# ── Main entry point ──────────────────────────────────────────────────────────

def build_scene_graph(
    nodes:     list[Node],
    img_w:     int,
    img_h:     int,
    rel_hints: list[str] | None = None,
) -> SceneGraph:
    """
    Builds a SceneGraph from a list of detected Nodes.

    Steps:
      1. Add all nodes to the graph
      2. Generate candidate pairs (prune by distance and size)
      3. Score each candidate pair → predicate + confidence
      4. Keep edges above MIN_PREDICATE_CONFIDENCE threshold

    rel_hints is an optional list of predicate strings parsed from the
    user's prompt (e.g. ["holding", "near"]). When provided, scores for
    those predicates get a +0.10 boost to reflect user intent.

    Returns a SceneGraph with nodes and unverified edges.
    VTT verification happens separately in vtt.py.
    """
    graph    = SceneGraph(nodes=nodes)
    img_diag = (img_w ** 2 + img_h ** 2) ** 0.5
    hints    = set(rel_hints or [])

    # Only propose edges between base-class nodes
    base_nodes = [n for n in nodes if not n.is_attribute]

    # For multi-class queries (primary + related roles present), only generate
    # edges between a primary node and a related node. This prevents spurious
    # cross-primary or cross-related pairs (e.g. person↔person edges when the
    # user asked for "person holding racket").
    primary_nodes = [n for n in base_nodes if n.role == "primary"]
    related_nodes = [n for n in base_nodes if n.role == "related"]
    if primary_nodes and related_nodes:
        candidate_pairs = [(p, r) for p in primary_nodes for r in related_nodes]
    else:
        candidate_pairs = list(combinations(base_nodes, 2))
    for node_a, node_b in candidate_pairs:
        # ── Distance pruning ──────────────────────────────────────────────────
        dist = node_a.distance_to(node_b)
        if dist > MAX_DISTANCE_FRACTION * img_diag:
            continue

        # ── Size ratio pruning ────────────────────────────────────────────────
        area_a = node_a.box_area
        area_b = node_b.box_area
        if area_a == 0 or area_b == 0:
            continue

        # ── Score all predicates for this pair ────────────────────────────────
        scores = _score_predicates(node_a, node_b, img_w, img_h, hints)

        # Keep the best-scoring predicate if above threshold
        if not scores:
            continue

        best_predicate, best_score = max(scores.items(), key=lambda x: x[1])
        if best_score < MIN_PREDICATE_CONFIDENCE:
            continue

        # Determine subject / object ordering based on semantic role
        subject, obj = _assign_subject_object(node_a, node_b, best_predicate)

        graph.edges.append(Edge(
            subject_id = subject.id,
            object_id  = obj.id,
            predicate  = best_predicate,
            confidence = round(best_score, 4),
            verified   = False,
        ))

    return graph


# ── Predicate scoring ─────────────────────────────────────────────────────────

def _score_predicates(
    a:     Node,
    b:     Node,
    img_w: int,
    img_h: int,
    hints: set[str],
) -> dict[str, float]:
    """
    Computes a confidence score for every applicable predicate
    between nodes a and b.

    All scores are in [0, 1]. Returns only predicates with score > 0.
    """
    scores: dict[str, float] = {}

    img_diag   = (img_w ** 2 + img_h ** 2) ** 0.5
    dist       = a.distance_to(b)
    norm_dist  = dist / img_diag         # 0 = touching, 1 = opposite corners
    iou        = a.overlap_ratio(b)
    area_a     = a.box_area
    area_b     = b.box_area
    size_ratio = area_a / area_b if area_b > 0 else 1.0

    a_animate  = a.label.lower() in ANIMATE_CLASSES
    b_animate  = b.label.lower() in ANIMATE_CLASSES
    a_large    = a.label.lower() in LARGE_STRUCTURE_CLASSES
    b_large    = b.label.lower() in LARGE_STRUCTURE_CLASSES
    b_rideable = b.label.lower() in RIDEABLE_CLASSES

    # Vertical alignment: positive = a is above b
    a_above_b = a.vertical_position_relative_to(b)

    # ── on_top_of ──────────────────────────────────────────────────────────
    # a is significantly above b, horizontal overlap exists
    if a_above_b > 0.3 and iou > 0.05 and norm_dist < 0.30:
        score = _clamp(a_above_b * 0.5 + iou * 0.5)
        scores["on_top_of"] = score

    # ── riding ─────────────────────────────────────────────────────────────
    # animate subject directly on/in a vehicle or animal — requires physical
    # contact: subject base (y2) must be within the rideable object's vertical
    # extent or no more than 5% of image height above it.
    if a_animate and b_rideable and a_above_b > 0.10:
        a_base   = a.bounding_box["y2"]
        b_top    = b.bounding_box["y1"]
        b_bot    = b.bounding_box["y2"]
        touching = b_top <= a_base <= b_bot          # base inside vehicle extent
        gap_frac = max(0.0, (b_top - a_base) / (img_h + 1)) if not touching else 0.0
        if touching or gap_frac < 0.05:
            score = _clamp(0.55 + a_above_b * 0.25 + iou * 0.20)
            scores["riding"] = score

    # ── sitting_on ─────────────────────────────────────────────────────────
    # animate subject on a non-vehicle/animal object — enforces direct contact:
    # subject base (y2) must be within the object or ≤8% image height above it.
    if a_animate and not b_rideable and a_above_b > 0.15 and iou > 0.02 and norm_dist < 0.25:
        a_base   = a.bounding_box["y2"]
        b_top    = b.bounding_box["y1"]
        b_bot    = b.bounding_box["y2"]
        touching = b_top <= a_base <= b_bot
        gap_frac = max(0.0, (b_top - a_base) / (img_h + 1)) if not touching else 0.0
        if touching or gap_frac < 0.08:
            score = _clamp(0.4 + a_above_b * 0.3 + iou * 0.3)
            scores["sitting_on"] = score

    # ── standing_on ────────────────────────────────────────────────────────
    # animate subject, base of a near top of b, very low iou
    a_base = a.bounding_box["y2"]
    b_top  = b.bounding_box["y1"]
    base_proximity = 1 - abs(a_base - b_top) / (img_h + 1)
    if a_animate and base_proximity > 0.92 and iou < 0.1:
        score = _clamp(base_proximity * 0.7 + (1 - norm_dist) * 0.3)
        scores["standing_on"] = score

    # ── holding ────────────────────────────────────────────────────────────
    # animate subject, much larger than object, within arm's-length proximity.
    # 0.18 norm_dist: tight enough to exclude background people while still
    # covering extended-arm poses (rackets, paddles held at full reach).
    if a_animate and size_ratio > SIZE_RATIO_LARGE and norm_dist < 0.18:
        score = _clamp(0.4 + (1 - norm_dist) * 0.4 + min(size_ratio / 10, 0.2))
        scores["holding"] = score

    # ── wearing ────────────────────────────────────────────────────────────
    # animate subject, object partially inside subject mask, object smaller
    if a_animate and size_ratio > 2.0 and iou > 0.10:
        score = _clamp(iou * 0.6 + (1 - norm_dist) * 0.4)
        scores["wearing"] = score

    # ── carrying ───────────────────────────────────────────────────────────
    # animate subject, object near subject, subject much larger
    if a_animate and size_ratio > SIZE_RATIO_LARGE and norm_dist < 0.15:
        score = _clamp(0.35 + (1 - norm_dist) * 0.4 + iou * 0.25)
        scores["carrying"] = score

    # ── playing_with ───────────────────────────────────────────────────────
    # animate subject, small object nearby (ball, toy, etc.)
    if a_animate and SIZE_RATIO_SMALL < size_ratio < SIZE_RATIO_LARGE and norm_dist < 0.25:
        score = _clamp(0.30 + (1 - norm_dist) * 0.5 + iou * 0.20)
        scores["playing_with"] = score

    # ── walking (subject walks a/with object) ──────────────────────────────
    # both animate, close, similar size
    if a_animate and b_animate and norm_dist < 0.30:
        size_similarity = 1 - abs(1 - size_ratio) / max(size_ratio, 1)
        score = _clamp(0.30 + (1 - norm_dist) * 0.4 + size_similarity * 0.30)
        scores["walking"] = score

    # ── parked_next_to ─────────────────────────────────────────────────────
    # both large structures, side by side, minimal vertical offset
    if a_large and b_large and norm_dist < 0.25 and abs(a_above_b) < 0.15:
        score = _clamp(0.35 + (1 - norm_dist) * 0.65)
        scores["parked_next_to"] = score

    # ── near (universal fallback) ──────────────────────────────────────────
    # always computed — serves as the floor predicate when nothing else fits
    near_score = _clamp((1 - norm_dist) * 0.5)
    if near_score > MIN_PREDICATE_CONFIDENCE:
        scores["near"] = near_score

    # ── Apply hint boost ───────────────────────────────────────────────────
    for pred in hints:
        if pred in scores:
            scores[pred] = _clamp(scores[pred] + 0.10)

    return scores


# ── Subject / object assignment ───────────────────────────────────────────────

def _assign_subject_object(
    a:         Node,
    b:         Node,
    predicate: str,
) -> tuple[Node, Node]:
    """
    Given two nodes and a predicate, returns (subject, object) in the
    correct semantic order.

    Rules:
    - For positional predicates (on_top_of, sitting_on, standing_on):
      higher node = subject, lower node = object
    - For size predicates (holding, wearing, carrying):
      larger node = subject, smaller node = object
    - For motion predicates (walking, playing_with):
      animate node = subject if only one is animate
    - For symmetric predicates (near, parked_next_to):
      larger node = subject by convention
    """
    positional = {"on_top_of", "sitting_on", "standing_on", "riding"}
    size_based = {"holding", "wearing", "carrying"}

    if predicate in positional:
        # Higher centroid (lower y) = subject
        if a.centroid[1] <= b.centroid[1]:
            return a, b
        return b, a

    if predicate in size_based:
        # Larger area = subject
        if a.box_area >= b.box_area:
            return a, b
        return b, a

    if predicate == "walking":
        # If both animate, larger = subject (the one doing the walking)
        if a.box_area >= b.box_area:
            return a, b
        return b, a

    # Default: larger = subject
    if a.box_area >= b.box_area:
        return a, b
    return b, a


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))