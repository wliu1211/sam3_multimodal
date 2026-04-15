"""
backend/services/vision.py

SAM 3 detection + CLIP attribute verification + SGG + VTT.

SAM 3 API reference:
  from sam3.model_builder import build_sam3_image_model
  from sam3.model.sam3_image_processor import Sam3Processor

  processor.set_image(image)             → sets image features
  processor.set_text_prompt(text, state) → returns state with masks/boxes/scores
"""

import io
import os
import base64
import torch
import numpy as np
from PIL import Image
from anthropic import AsyncAnthropic

from middleware.queue import GpuQueue
from services.graph import Node, SceneGraph
from services.sgg   import build_scene_graph, normalize_hints
from services.vtt   import verify_edges


_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

CLIP_AMBIGUOUS_LOW    = 0.40
CLIP_AMBIGUOUS_HIGH   = 0.70
MIN_BOX_AREA_FRACTION = 0.001
CROP_PADDING_PX       = 12


DEDUP_IOU_THRESHOLD   = 0.70
SAM_INPUT_MAX_SIDE = 1024
MAX_DETECTIONS_PER_CLASS = 20


def _maybe_downscale(
    image: Image.Image,
    max_side: int,
) -> tuple[Image.Image, float, float]:
    """
    Proportionally downscales *image* so its longest side is at most *max_side*.
    Returns (downscaled_image, scale_x, scale_y) where scale factors convert
    downscaled pixel coordinates back to the original image space.
    If the image is already small enough, returns (image, 1.0, 1.0).
    """
    orig_w, orig_h = image.size
    if orig_w <= max_side and orig_h <= max_side:
        return image, 1.0, 1.0

    if orig_w >= orig_h:
        new_w = max_side
        new_h = max(1, round(orig_h * max_side / orig_w))
    else:
        new_h = max_side
        new_w = max(1, round(orig_w * max_side / orig_h))

    downscaled = image.resize((new_w, new_h), Image.BILINEAR)
    return downscaled, orig_w / new_w, orig_h / new_h


def _mask_to_uint8_hw(mask) -> np.ndarray:
    """SAM 3 returns masks as torch.BoolTensor (H, W) or numpy; RLE encoder needs uint8 ndarray."""
    if isinstance(mask, torch.Tensor):
        return mask.detach().cpu().numpy().astype(np.uint8, copy=False)
    return np.asarray(mask, dtype=np.uint8)


def _to_float_scalar(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def _box_to_int_xyxy(box) -> tuple[int, int, int, int]:
    if isinstance(box, torch.Tensor):
        b = box.detach().cpu().flatten().tolist()
        return int(b[0]), int(b[1]), int(b[2]), int(b[3])
    return tuple(int(v) for v in box)


                                                                                

def _box_iou(a: dict, b: dict) -> float:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter  = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _has_attributes(target: dict) -> bool:
    """Returns True if the target carries any attribute constraints that require CLIP verification."""
    attrs = target.get("attributes", {}) or {}
    return bool(
        attrs.get("color")
        or attrs.get("make")
        or attrs.get("type")
        or attrs.get("other")
    )


def _dedup_instances(instances: list[dict]) -> list[dict]:
    """
    Within each base_class group, removes lower-confidence duplicates
    when two bounding boxes overlap above DEDUP_IOU_THRESHOLD.
    """
    from collections import defaultdict
    by_class: dict[str, list[dict]] = defaultdict(list)
    for inst in instances:
        by_class[inst["base_class"]].append(inst)

    deduped: list[dict] = []
    for class_insts in by_class.values():
        sorted_insts = sorted(class_insts, key=lambda x: x["confidence"], reverse=True)
        kept: list[dict] = []
        for inst in sorted_insts:
            if not any(_box_iou(inst["bounding_box"], k["bounding_box"]) > DEDUP_IOU_THRESHOLD for k in kept):
                kept.append(inst)
        deduped.extend(kept)
    return deduped


                                                                                

async def run_vision_pipeline(
    image_bytes:  bytes,
    parsed_query: dict,
    models:       dict,
) -> tuple[list[dict], SceneGraph]:
    """
    Full multi-target vision pipeline. Returns (instances, scene_graph).

    parsed_query shape (from parser.py):
    {
        "targets": [
            {"base_class": str, "attributes": dict, "confidence_threshold": float, "role": str},
            ...
        ],
        "relational_hints": list[str],
    }

    instances — list of instance dicts:
    {
        "id":           str,
        "bounding_box": {x1, y1, x2, y2},
        "mask_rle":     list[int],
        "confidence":   float,
        "matched":      bool,
        "base_class":   str,
        "role":         str,          # "primary" | "related"
        "target_label": str,          # human-readable attribute description
    }

    scene_graph — SceneGraph with nodes + verified cross-class edges.
    """
    from collections import defaultdict

    image_pil    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_w, img_h = image_pil.size

    sam_pil, _sx, _sy = _maybe_downscale(image_pil, SAM_INPUT_MAX_SIDE)

    targets          = parsed_query["targets"]
    relational_hints = parsed_query.get("relational_hints", [])

    class_to_targets: dict[str, list[dict]] = defaultdict(list)
    for tgt in targets:
        class_to_targets[tgt["base_class"]].append(tgt)

    all_instances: list[dict] = []
    global_id = 0

    async with GpuQueue():
        for base_class, class_targets in class_to_targets.items():
            detections = _detect_with_sam3(sam_pil, base_class, models)
            if not detections:
                continue
            detections = sorted(detections, key=lambda d: d["sam_score"], reverse=True)
            detections = detections[:MAX_DETECTIONS_PER_CLASS]
            any_attrs = any(_has_attributes(t) for t in class_targets)

            for det in detections:
                raw_box = det["bounding_box"]
                box = {
                    "x1": round(raw_box["x1"] * _sx),
                    "y1": round(raw_box["y1"] * _sy),
                    "x2": round(raw_box["x2"] * _sx),
                    "y2": round(raw_box["y2"] * _sy),
                }

                mask = det["mask"]
                if _sx != 1.0 or _sy != 1.0:
                    mask = np.array(
                        Image.fromarray(mask).resize((img_w, img_h), Image.NEAREST)
                    )

                box_area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
                if box_area / (img_w * img_h) < MIN_BOX_AREA_FRACTION:
                    continue

                if not any_attrs:
                    best_target = class_targets[0]
                    best_score  = det["sam_score"]
                    all_instances.append({
                        "id":           str(global_id).zfill(4),
                        "bounding_box": box,
                        "mask_rle":     _encode_mask_rle(mask),
                        "confidence":   round(best_score, 4),
                        "matched":      True,
                        "base_class":   base_class,
                        "role":         best_target["role"],
                        "target_label": base_class,
                    })
                    global_id += 1
                    continue

                crop = _crop_instance(image_pil, box, img_w, img_h)

                best_target: dict | None = None
                best_score               = -1.0

                for tgt in class_targets:
                    if not _has_attributes(tgt):
                        if det["sam_score"] > best_score:
                            best_score  = det["sam_score"]
                            best_target = tgt
                        continue

                    attr_desc   = _build_attribute_description(base_class, tgt["attributes"])
                    clip_score  = _score_with_clip(crop, attr_desc, models)
                    final_score = clip_score

                    if (
                        tgt["attributes"].get("make")
                        and CLIP_AMBIGUOUS_LOW <= clip_score <= CLIP_AMBIGUOUS_HIGH
                    ):
                        vqa_match   = await _verify_with_claude_vision(crop, tgt["attributes"], base_class)
                        final_score = (clip_score + (1.0 if vqa_match else 0.0)) / 2

                    if final_score > best_score:
                        best_score  = final_score
                        best_target = tgt

                if best_target is None:
                    continue

                all_instances.append({
                    "id":           str(global_id).zfill(4),
                    "bounding_box": box,
                    "mask_rle":     _encode_mask_rle(mask),
                    "confidence":   round(best_score, 4),
                    "matched":      best_score >= best_target["confidence_threshold"],
                    "base_class":   base_class,
                    "role":         best_target["role"],
                    "target_label": _make_target_label(base_class, best_target["attributes"]),
                })
                global_id += 1

    all_instances = _dedup_instances(all_instances)

    distinct_classes = {inst["base_class"] for inst in all_instances}
    needs_graph = len(distinct_classes) > 1 or bool(relational_hints)

    if not needs_graph:
        return all_instances, SceneGraph()

    nodes = [
        Node(
            id           = inst["id"],
            label        = inst["base_class"],
            bounding_box = inst["bounding_box"],
            mask_rle     = inst["mask_rle"],
            confidence   = inst["confidence"],
            is_attribute = False,
            role         = inst["role"],
        )
        for inst in all_instances
        if inst.get("matched")
    ]

    if not nodes:
        return all_instances, SceneGraph()

    normalized_hints = normalize_hints(relational_hints)

    graph = build_scene_graph(
        nodes     = nodes,
        img_w     = img_w,
        img_h     = img_h,
        rel_hints = normalized_hints or None,
    )

    graph = await verify_edges(
        graph     = graph,
        image_pil = image_pil,
        img_w     = img_w,
        img_h     = img_h,
    )

    return all_instances, graph


def _detect_with_sam3(
    image_pil:  Image.Image,
    base_class: str,
    models:     dict,
) -> list[dict]:
    """
    Runs SAM 3 text-prompted segmentation on the image.

    state["masks"]  → (N, H, W) bool torch.Tensor or numpy
    state["boxes"]  → (N, 4)    [x1, y1, x2, y2] tensor or array
    state["scores"] → (N,)      confidence scores
    """
    processor = models.get("sam3")
    if processor is None:
        raise RuntimeError("SAM 3 model is not loaded. Check startup logs.")

    with torch.inference_mode():
        state = processor.set_image(image_pil)
        state = processor.set_text_prompt(base_class, state)

    masks  = state.get("masks")
    boxes  = state.get("boxes")
    scores = state.get("scores")

    if masks is None or boxes is None or scores is None:
        return []

    n = len(masks)
    if n == 0:
        return []

    detections = []
    for i in range(n):
        x1, y1, x2, y2 = _box_to_int_xyxy(boxes[i])
        detections.append({
            "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "mask":         _mask_to_uint8_hw(masks[i]),
            "sam_score":    _to_float_scalar(scores[i]),
        })

    return detections


def _score_with_clip(
    crop:             Image.Image,
    attr_description: str,
    models:           dict,
) -> float:
    """
    Scores the crop against attr_description using CLIP.
    Returns softmax probability of the positive class in [0, 1].
    """
    clip_model  = models.get("clip")
    preprocess  = models.get("clip_preprocess")
    tokenizer   = models.get("clip_tokenizer")
    device: str = models.get("device", "cpu")

    if clip_model is None:
        raise RuntimeError("CLIP model is not loaded. Check startup logs.")

    base_word = attr_description.split()[-1] if attr_description.split() else "object"
    positive  = attr_description
    negative  = f"not a {base_word}"

    texts      = tokenizer([positive, negative]).to(device)
    img_tensor = preprocess(crop).unsqueeze(0).to(device)

    with torch.inference_mode():
        img_feat = clip_model.encode_image(img_tensor)
        txt_feat = clip_model.encode_text(texts)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        logits   = (img_feat @ txt_feat.T) * 100
        probs    = logits.softmax(dim=-1)

    return float(probs[0][0].item())


async def _verify_with_claude_vision(
    crop:       Image.Image,
    attributes: dict,
    base_class: str,
) -> bool:
    """
    Asks Claude a yes/no question about the cropped instance.
    Only called when CLIP score is ambiguous on make/brand.
    """
    make  = attributes.get("make")
    color = attributes.get("color")

    if make and color:
        question = f"Is this a {color} {make} {base_class}? Answer only 'yes' or 'no'."
    elif make:
        question = f"Is this a {make} {base_class}? Answer only 'yes' or 'no'."
    else:
        question = f"Does this {base_class} have the color {color}? Answer only 'yes' or 'no'."

    buffer = io.BytesIO()
    crop.save(buffer, format="JPEG", quality=85)
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
        return response.content[0].text.strip().lower().startswith("yes")

    except Exception as e:
        print(f"[vision] Claude Vision fallback failed: {e}")
        return False


def _make_target_label(base_class: str, attributes: dict) -> str:
    """
    Short, stable label used as the grouping key shared between vision.py and
    build_groups_payload() in renderer.py.

    Format: "<color> <make> <type> <base_class>" — only structured fields,
    no "a" article, no free-text "other" items (those are CLIP-only).
    Must stay in sync with build_groups_payload().
    """
    parts = [
        v for v in [
            attributes.get("color"),
            attributes.get("make"),
            attributes.get("type"),
            base_class,
        ] if v
    ]
    return " ".join(parts)


def _build_attribute_description(base_class: str, attributes: dict) -> str:
    """
    Natural-language description passed to CLIP for similarity scoring.
    Includes free-text 'other' items (e.g. 'wearing blue shirt') so CLIP
    can reason about appearance. NOT used as a grouping key.
    """
    parts = ["a"]
    if attributes.get("color"): parts.append(attributes["color"])
    if attributes.get("make"):  parts.append(attributes["make"])
    if attributes.get("type"):  parts.append(attributes["type"])
    parts.append(base_class)
    for item in (attributes.get("other") or []):
        parts.append(item)
    return " ".join(parts)


def _crop_instance(
    image: Image.Image,
    box:   dict,
    img_w: int,
    img_h: int,
) -> Image.Image:
    x1 = max(0,     box["x1"] - CROP_PADDING_PX)
    y1 = max(0,     box["y1"] - CROP_PADDING_PX)
    x2 = min(img_w, box["x2"] + CROP_PADDING_PX)
    y2 = min(img_h, box["y2"] + CROP_PADDING_PX)
    return image.crop((x1, y1, x2, y2))


def _encode_mask_rle(mask: np.ndarray) -> list[int]:
    """
    RLE encodes a binary (H, W) uint8 mask.
    Format: alternating [background_count, foreground_count, ...]
    Must stay in sync with decodeRleMask() in frontend/utils/canvasUtils.ts.
    """
    flat    = mask.flatten()
    rle     = []
    current = 0
    count   = 0
    for pixel in flat:
        if pixel == current:
            count += 1
        else:
            rle.append(count)
            count   = 1
            current = pixel
    rle.append(count)
    return rle