"""
backend/services/vision.py

SAM 3 detection + CLIP attribute verification.

SAM 3 API reference:
  from sam3.model_builder import build_sam3_image_model
  from sam3.model.sam3_image_processor import Sam3Processor

  processor.set_image(image)          → sets image features
  processor.set_text_prompt(text, state) → returns state with masks/boxes/scores
"""

import io
import os
import base64
import torch
import numpy as np
from PIL import Image
from anthropic import AsyncAnthropic

# ── Client ────────────────────────────────────────────────────────────────────

_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Constants ─────────────────────────────────────────────────────────────────

CLIP_AMBIGUOUS_LOW    = 0.40
CLIP_AMBIGUOUS_HIGH   = 0.70
MIN_BOX_AREA_FRACTION = 0.001
CROP_PADDING_PX       = 12

# ── Main entry point ──────────────────────────────────────────────────────────

async def run_vision_pipeline(
    image_bytes:  bytes,
    parsed_query: dict,
    models:       dict,
) -> list[dict]:
    """
    Full vision pipeline. Returns a list of instance dicts.

    Each instance:
    {
        "id":           str,
        "bounding_box": {x1, y1, x2, y2},
        "mask_rle":     list[int],
        "confidence":   float,
        "matched":      bool,
    }
    """
    image_pil    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_w, img_h = image_pil.size

    base_class   = parsed_query["base_class"]
    attributes   = parsed_query["attributes"]
    threshold    = parsed_query["confidence_threshold"]
    attr_desc    = _build_attribute_description(base_class, attributes)

    # Step 1 — SAM 3 text-prompted detection
    detections = _detect_with_sam3(image_pil, base_class, models)
    if not detections:
        return []

    # Steps 2–3 — CLIP scoring + optional Claude Vision fallback
    results = []
    for i, det in enumerate(detections):
        box  = det["bounding_box"]
        mask = det["mask"]

        box_area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
        if box_area / (img_w * img_h) < MIN_BOX_AREA_FRACTION:
            continue

        crop       = _crop_instance(image_pil, box, img_w, img_h)
        clip_score = _score_with_clip(crop, attr_desc, models)

        final_score = clip_score
        if (
            attributes.get("make")
            and CLIP_AMBIGUOUS_LOW <= clip_score <= CLIP_AMBIGUOUS_HIGH
        ):
            vqa_match   = await _verify_with_claude_vision(crop, attributes, base_class)
            final_score = (clip_score + (1.0 if vqa_match else 0.0)) / 2

        results.append({
            "id":           str(i).zfill(4),
            "bounding_box": box,
            "mask_rle":     _encode_mask_rle(mask),
            "confidence":   round(final_score, 4),
            "matched":      final_score >= threshold,
        })

    return results


# ── SAM 3 detection ───────────────────────────────────────────────────────────

def _detect_with_sam3(
    image_pil:  Image.Image,
    base_class: str,
    models:     dict,
) -> list[dict]:
    """
    Runs SAM 3 text-prompted segmentation on the image.

    SAM 3 API:
      state = processor.set_image(image_pil)
      state = processor.set_text_prompt(text, state)

      state["masks"]  → (N, H, W) bool array
      state["boxes"]  → (N, 4)    array in [x1, y1, x2, y2] format
      state["scores"] → (N,)      confidence scores

    Returns list of {bounding_box, mask} dicts.
    """
    processor = models.get("sam3")
    if processor is None:
        raise RuntimeError("SAM 3 model is not loaded. Check startup logs.")

    # Set image — encodes visual features (expensive, done once per image)

    # Text prompt — finds all instances of base_class
    with torch.inference_mode():
        state = processor.set_image(image_pil)
        state = processor.set_text_prompt(base_class, state)

    masks  = state.get("masks")   # (N, H, W) bool numpy array
    boxes  = state.get("boxes")   # (N, 4) array
    scores = state.get("scores")  # (N,) array

    if masks is None or boxes is None or scores is None or len(masks) == 0:
        return []

    detections = []
    for mask, box, score in zip(masks, boxes, scores):
        x1, y1, x2, y2 = (int(v) for v in box)
        detections.append({
            "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "mask":         mask.astype(np.uint8),
            "sam_score":    float(score),
        })

    return detections


# ── CLIP scoring ──────────────────────────────────────────────────────────────

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


# ── Claude Vision fallback ────────────────────────────────────────────────────

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_attribute_description(base_class: str, attributes: dict) -> str:
    parts = ["a"]
    if attributes.get("color"): parts.append(attributes["color"])
    if attributes.get("make"):  parts.append(attributes["make"])
    if attributes.get("type"):  parts.append(attributes["type"])
    parts.append(base_class)
    if attributes.get("other"): parts.append(" ".join(attributes["other"]))
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