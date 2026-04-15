"""
backend/services/discovery.py

Discovery pipeline — runs when the user asks an open-ended question like
"what is happening in this image?" instead of a targeted prompt.

Steps:
  1. RAM++ image tagging   → raw concept list from the image
  2. Noun filtering        → drop verbs/colors/scene words, normalise synonyms
  3. Top-K cap             → keep at most MAX_CONCEPTS to bound SAM call count
  4. Multi-class SAM 3     → one detection call per concept
  5. IoU deduplication     → merge overlapping masks from different calls
  6. Return merged node list for CLIP → SGG → VTT (same as targeted path)
"""

import io
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional

from services.graph import Node
from services.vision import _score_with_clip, _crop_instance, _encode_mask_rle
from middleware.queue import GpuQueue                                                                      
                                                             
MAX_CONCEPTS = 8
IOU_DEDUP_THRESHOLD = 0.60
CLIP_DROP_THRESHOLD = 0.35
                  
CLIP_SAM_BLEND = 0.55
SKIP_TAGS: set[str] = {
                     
    "walk", "walking", "run", "running", "sit", "sitting", "stand", "standing",
    "lay", "laying", "lie", "lying", "play", "playing", "swim", "swimming",
    "jump", "jumping", "ride", "riding", "hold", "holding", "carry", "carrying",
    "cook", "cooking", "eat", "eating", "drink", "drinking", "talk", "talking",
    "look", "looking", "wear", "wearing", "smile", "smiling", "wave", "waving",
    "climb", "climbing", "throw", "throwing", "catch", "catching", "kick",
    "hit", "swing", "swinging", "surf", "surfing", "dive", "diving",
                                                       
    "red", "blue", "green", "yellow", "white", "black", "brown", "gray",
    "grey", "orange", "purple", "pink", "beige", "tan", "teal", "gold",
    "silver", "dark", "light", "bright",
                        
    "indoor", "outdoor", "inside", "outside", "urban", "rural", "nature",
    "sunny", "cloudy", "foggy", "rainy", "warm", "cold", "hot", "cool",
    "bright", "dark", "daytime", "nighttime", "morning", "evening",
    "busy", "quiet", "crowded", "empty", "open", "close",
                              
    "sport", "sports", "activity", "recreation", "competition", "game",
    "fun", "happy", "sad", "excited", "relaxed", "comfortable",
    "home", "life", "style", "fashion", "design", "art",
    "large", "small", "big", "tiny", "tall", "short", "wide", "narrow",
    "old", "new", "young", "fresh", "clean", "dirty",
                          
    "living room", "dining room", "bedroom", "bathroom", "kitchen",
    "city street", "country road", "open field",
}

                                                                       
                                                                     
SYNONYM_MAP: dict[str, str] = {
    "sofa":          "couch",
    "automobile":    "car",
    "vehicle":       "car",
    "canine":        "dog",
    "feline":        "cat",
    "bicycle":       "bike",
    "motorbike":     "motorcycle",
    "aeroplane":     "airplane",
    "aeroplane":     "airplane",
    "settee":        "couch",
    "spectacles":    "glasses",
    "eyeglasses":    "glasses",
    "handbag":       "bag",
    "pocketbook":    "bag",
    "purse":         "bag",
    "rubbish bin":   "trash can",
    "dustbin":       "trash can",
    "rubbish":       "trash can",
    "trainers":      "sneakers",
    "jumper":        "sweater",
    "football":      "soccer ball",                                                 
    "american football": "football",
    "pickleball":    "pickleball",                                   
}


VERB_TO_OBJECT_HINTS: dict[str, list[str]] = {
    "surfing":    ["surfboard"],
    "skateboarding": ["skateboard"],
    "skiing":     ["ski", "skis"],
    "snowboarding": ["snowboard"],
    "cycling":    ["bike"],
    "rowing":     ["boat", "oar"],
    "golfing":    ["golf club", "golf ball"],
}




async def run_discovery_pipeline(
    image_bytes: bytes,
    models:      dict,
) -> tuple[list[dict], list[str], list[str]]:
    """
    Runs the full discovery pipeline on an image with no prior prompt.

    Returns:
        instances   — flat list of instance dicts (same shape as targeted pipeline)
        noun_tags   — filtered noun concepts that were detected
        context_tags — verbs/attrs kept for label generation in activity.py
    """
    import torch

    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                                                                                
    raw_tags = _run_ram(image_pil, models)

                                                                               
    noun_tags, context_tags = _filter_tags(raw_tags)

                                                                               
    noun_tags = noun_tags[:MAX_CONCEPTS]

    if not noun_tags:
        return [], [], context_tags

    img_w, img_h = image_pil.size

                                                                               
    async with GpuQueue():
                                             
        all_raw_instances: list[dict] = []
        for concept in noun_tags:
            concept_instances = _detect_concept(image_pil, concept, models)
            all_raw_instances.extend(concept_instances)

                                        
        deduped = _dedup_by_iou(all_raw_instances, img_w, img_h)

        deduped = _clip_verify_instances(deduped, image_pil, img_w, img_h, models)

    return deduped, noun_tags, context_tags


def _run_ram(image_pil: Image.Image, models: dict) -> list[str]:
    """
    Runs RAM++ on the image and returns the raw tag list sorted by confidence.
    Falls back to an empty list if RAM++ is not loaded.
    """
    ram = models.get("ram")
    if ram is None:
        return []

    try:
        import torch
        from ram import inference_ram, get_transform

        transform = get_transform(image_size=384)
        image_tensor = transform(image_pil).unsqueeze(0)

        device = next(ram.parameters()).device
        image_tensor = image_tensor.to(device)

        tags_str, _ = inference_ram(image_tensor, ram)

        return [t.strip().lower() for t in tags_str.split("|") if t.strip()]
    except Exception as e:
        print(f"[discovery] RAM++ inference failed: {e}")
        return []

def _filter_tags(raw_tags: list[str]) -> tuple[list[str], list[str]]:
    """
    Splits RAM++ tags into:
      noun_tags    — physical objects to feed into SAM 3
      context_tags — verbs and attributes to keep for label generation

    Also applies synonym normalisation and checks verb hints for implied objects.
    """
    noun_tags:    list[str] = []
    context_tags: list[str] = []
    seen:         set[str]  = set()

    implied: list[str] = []
    for tag in raw_tags:
        if tag in VERB_TO_OBJECT_HINTS:
            context_tags.append(tag)
            implied.extend(VERB_TO_OBJECT_HINTS[tag])

    all_tags = raw_tags + implied

    for tag in all_tags:
        tag = SYNONYM_MAP.get(tag, tag)

        if tag in seen:
            continue
        seen.add(tag)

        if tag in SKIP_TAGS:
            if tag not in context_tags:
                context_tags.append(tag)
            continue

        if len(tag.split()) > 3:
            continue

        if len(tag) <= 1 or tag.isdigit():
            continue

        noun_tags.append(tag)

    return noun_tags, context_tags

def _detect_concept(
    image_pil:  Image.Image,
    concept:    str,
    models:     dict,
) -> list[dict]:
    """
    Runs one SAM 3 text-prompted detection call for a single concept.
    Returns a list of raw instance dicts tagged with their concept label.
    """
    import torch

    processor = models.get("sam3")
    if processor is None:
        return []

    try:
        image_np = np.array(image_pil)
        with torch.inference_mode():
            state = processor.set_image(image_pil)
            state = processor.set_text_prompt(concept, state)

        masks  = state.get("masks")
        boxes  = state.get("boxes")
        scores = state.get("scores")

        if masks is None or len(masks) == 0:
            return []

        results = []
        for mask, box, score in zip(masks, boxes, scores):
            x1, y1, x2, y2 = _box_to_ints(box)
            results.append({
                "label":        concept,
                "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "mask":         _to_uint8(mask),
                "sam_score":    float(score) if hasattr(score, "item") else float(score),
            })
        return results

    except Exception as e:
        print(f"[discovery] SAM 3 failed for concept '{concept}': {e}")
        return []

def _dedup_by_iou(instances: list[dict], img_w: int, img_h: int) -> list[dict]:
    """
    Removes duplicate detections — instances from different SAM calls that
    overlap by more than IOU_DEDUP_THRESHOLD are considered the same physical
    object. The higher-confidence detection is kept.

    Uses mask IoU (pixel-level) when masks are present, falling back to box
    IoU. Mask IoU is significantly more accurate for non-rectangular or
    partially occluded objects where bounding boxes heavily overlap but the
    actual segmented regions do not.

    Returns a flat list with stable IDs and mask_rle assigned, ready for the
    CLIP pass and then SGG/VTT in routes.py.
    """
    if not instances:
        return []

    sorted_insts = sorted(instances, key=lambda x: x["sam_score"], reverse=True)
    kept: list[dict] = []

    for inst in sorted_insts:
        is_dup = False
        for existing in kept:
            iou = _mask_iou(inst["mask"], existing["mask"]) if (
                inst.get("mask") is not None and existing.get("mask") is not None
            ) else _box_iou(inst["bounding_box"], existing["bounding_box"])
            if iou > IOU_DEDUP_THRESHOLD:
                is_dup = True
                break
        if not is_dup:
            kept.append(inst)

    result = []
    for i, inst in enumerate(kept):
        result.append({
            "id":           str(i).zfill(4),
            "label":        inst["label"],
            "bounding_box": inst["bounding_box"],
            "mask":         inst["mask"],
            "sam_score":    inst["sam_score"],
            "confidence":   inst["sam_score"],
            "matched":      True,
            "mask_rle":     _encode_mask_rle(inst["mask"]),
        })
    return result

def _clip_verify_instances(
    instances: list[dict],
    image_pil: Image.Image,
    img_w:     int,
    img_h:     int,
    models:    dict,
) -> list[dict]:
    """
    Scores each deduped instance crop against its label using CLIP.

    For every instance:
      - Builds description "a {label}" vs "not a {label}"
      - Gets CLIP softmax probability for the positive class
      - Blends with SAM score: final = CLIP_SAM_BLEND * clip + (1-CLIP_SAM_BLEND) * sam
      - Drops instances where clip_score < CLIP_DROP_THRESHOLD

    If CLIP is not loaded, returns instances unchanged (graceful degradation).
    """
    if models.get("clip") is None:
        return instances

    verified = []
    for inst in instances:
        try:
            crop       = _crop_instance(image_pil, inst["bounding_box"], img_w, img_h)
            clip_score = _score_with_clip(crop, f"a {inst['label']}", models)
        except Exception as e:
            print(f"[discovery] CLIP failed for '{inst['label']}': {e}")
            verified.append(inst)
            continue

        if clip_score < CLIP_DROP_THRESHOLD:
            continue

        blended = CLIP_SAM_BLEND * clip_score + (1.0 - CLIP_SAM_BLEND) * inst["sam_score"]
        verified.append({**inst, "confidence": round(blended, 4)})

    return verified


                                                                                

def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Pixel-level intersection-over-union for two uint8 (H, W) binary masks."""
    try:
        inter = float(np.logical_and(a, b).sum())
        union = float(np.logical_or(a, b).sum())
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def _box_iou(a: dict, b: dict) -> float:
    """Intersection-over-union between two {x1,y1,x2,y2} dicts (fallback)."""
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union  = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _to_uint8(mask) -> np.ndarray:
    import torch
    if isinstance(mask, torch.Tensor):
        return mask.detach().cpu().numpy().astype(np.uint8, copy=False)
    return np.asarray(mask, dtype=np.uint8)


def _box_to_ints(box) -> tuple[int, int, int, int]:
    import torch
    if isinstance(box, torch.Tensor):
        b = box.detach().cpu().flatten().tolist()
        return int(b[0]), int(b[1]), int(b[2]), int(b[3])
    return tuple(int(v) for v in box)