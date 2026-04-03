import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

import aiosqlite

from db.database import get_db
from db.queries import insert_query_log, fetch_query_by_id, fetch_query_history
from services.parser import parse_prompt
from services.vision import run_vision_pipeline
from services.renderer import render_results, build_groups_payload
from middleware.queue import GpuQueue
from middleware.limiter import rate_limit
from middleware.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR  = PROJECT_ROOT / "storage" / "results"

# ── Allowed image types ───────────────────────────────────────────────────────

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES       = 20 * 1024 * 1024  # 20 MB

# ─────────────────────────────────────────────────────────────────────────────
# POST /api/query
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/query", summary="Run a semantic visual query", dependencies=[Depends(rate_limit)])
async def query(
    request: Request,
    image:   UploadFile = File(...,  description="Source image (JPEG, PNG, or WebP)"),
    prompt:  str        = Form(...,  description="Natural language search description"),
    db:      aiosqlite.Connection = Depends(get_db),
):
    """
    Full pipeline endpoint:
      1. Validate image upload
      2. Parse prompt → structured JSON via Claude
      3. Detect + segment instances via SAM 3
      4. Verify attributes via CLIP (+ Claude Vision fallback)
      5. Render mask overlays via OpenCV
      6. Log result to SQLite
      7. Return response

    Returns a JSON payload with matched groups, mask data,
    and the URL of the annotated result image.
    """
    query_id   = uuid.uuid4().hex
    start_time = time.time()

    logger.info(f"[{query_id}] Received query: '{prompt}'")

    # ── 1. Validate upload ────────────────────────────────────────────────────

    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image type '{image.content_type}'. "
                   f"Use JPEG, PNG, or WebP.",
        )

    image_bytes = await image.read()

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 20MB size limit. Please upload a smaller image.",
        )

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")

    # ── 2. Parse prompt ───────────────────────────────────────────────────────

    try:
        parsed_query = await parse_prompt(prompt)
        logger.info(
            f"[{query_id}] Parsed: class='{parsed_query['base_class']}' "
            f"attrs={parsed_query['attributes']} "
            f"threshold={parsed_query['confidence_threshold']}"
        )
    except (ValueError, RuntimeError) as e:
        logger.error(f"[{query_id}] Parser failed: {e}")
        raise HTTPException(status_code=422, detail=f"Prompt parsing failed: {e}")

    # ── 3 & 4. Vision pipeline ────────────────────────────────────────────────

    models = request.app.state.models

    if models.get("sam3") is None:
        raise HTTPException(
            status_code=503,
            detail="SAM 3 model is not loaded. Check server startup logs.",
        )

    try:
        async with GpuQueue():
            instances = await run_vision_pipeline(image_bytes, parsed_query, models)
        logger.info(
            f"[{query_id}] Vision complete: "
            f"{len(instances)} instances detected, "
            f"{sum(1 for i in instances if i['matched'])} matched"
        )
    except Exception as e:
        logger.error(f"[{query_id}] Vision pipeline failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Vision pipeline error: {e}")

    # ── 5. Render ─────────────────────────────────────────────────────────────

    groups = build_groups_payload(parsed_query, instances)

    try:
        filename, _ = render_results(image_bytes, groups, RESULTS_DIR)
        logger.info(f"[{query_id}] Rendered result image: {filename}")
    except Exception as e:
        logger.error(f"[{query_id}] Renderer failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Rendering error: {e}")

    # ── 6. Log to DB ──────────────────────────────────────────────────────────

    inference_ms  = int((time.time() - start_time) * 1000)
    total_matches = sum(g["matchCount"] for g in groups)

    try:
        await insert_query_log(
            db           = db,
            id           = query_id,
            prompt       = prompt,
            base_class   = parsed_query["base_class"],
            parsed_attrs = parsed_query["attributes"],
            match_count  = total_matches,
            inference_ms = inference_ms,
            result_image = filename,
        )
    except Exception as e:
        # Non-fatal — log the error but don't fail the response over a DB write
        logger.warning(f"[{query_id}] DB log failed (non-fatal): {e}")

    # ── 7. Response ───────────────────────────────────────────────────────────

    logger.info(f"[{query_id}] Done in {inference_ms}ms — {total_matches} matches")

    return JSONResponse({
        "id":                query_id,
        "prompt":            prompt,
        "parsed_query":      parsed_query,
        "groups":            groups,
        "annotated_image_url": f"/results/{filename}",
        "inference_ms":      inference_ms,
        "created_at":        _utc_now(),
    })


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/query/:id
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/query/{query_id}", summary="Fetch a past query result by ID")
async def get_query(
    query_id: str,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Returns the stored log entry for a previously run query.
    Used by the frontend history sidebar to reload a past result
    without re-running inference.
    """
    row = await fetch_query_by_id(db, query_id)

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Query '{query_id}' not found.",
        )

    # Attach the result image URL if one was saved
    if row.get("result_image"):
        row["annotated_image_url"] = f"/results/{row['result_image']}"

    return JSONResponse(row)


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/history
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/history", summary="Paginated query history")
async def get_history(
    page:      int = 1,
    page_size: int = 20,
    db: aiosqlite.Connection = Depends(get_db),
):
    """
    Returns a paginated list of past query log entries, newest first.
    Each entry includes the prompt, match count, inference time,
    and the result image URL.

    Query params:
      page      — 1-indexed page number (default 1)
      page_size — entries per page, max 100 (default 20)
    """
    # Clamp page_size so no one requests 10,000 rows
    page_size = max(1, min(page_size, 100))
    page      = max(1, page)

    data = await fetch_query_history(db, page=page, page_size=page_size)

    # Attach result image URLs to each entry
    for entry in data["entries"]:
        if entry.get("result_image"):
            entry["annotated_image_url"] = f"/results/{entry['result_image']}"

    return JSONResponse(data)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()