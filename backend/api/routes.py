import io
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

import aiosqlite

from db.database import get_db
from db.queries import insert_query_log, fetch_query_by_id, fetch_query_history
from services.parser import parse_prompt
from services.vision import run_vision_pipeline
from services.renderer import render_results, build_groups_payload, filter_compositional_instances
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
      5. Build scene graph (SGG) + verify edges (VTT)
      6. Render mask overlays + relationship edges via OpenCV
      7. Log result to SQLite
      8. Return response with groups + scene graph
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
        primary = next((t for t in parsed_query["targets"] if t["role"] == "primary"), parsed_query["targets"][0])
        logger.info(
            f"[{query_id}] Parsed: {len(parsed_query['targets'])} target(s), "
            f"primary='{primary['base_class']}' "
            f"hints={parsed_query['relational_hints']}"
        )
    except (ValueError, RuntimeError) as e:
        logger.error(f"[{query_id}] Parser failed: {e}")
        raise HTTPException(status_code=422, detail=f"Prompt parsing failed: {e}")

    # ── 3, 4, 5. Vision pipeline (detection + CLIP + SGG + VTT) ──────────────

    models = request.app.state.models

    if models.get("sam3") is None:
        raise HTTPException(
            status_code=503,
            detail="SAM 3 model is not loaded. Check server startup logs.",
        )

    try:
        # GpuQueue is acquired internally by run_vision_pipeline for the
        # SAM3+CLIP section only. VTT / SGG run outside the GPU lock so
        # Claude API calls never block other inference requests.
        instances, scene_graph = await run_vision_pipeline(
            image_bytes, parsed_query, models
        )

        # Compositional filtering: for multi-class queries (e.g. "man holding
        # racket"), keep only primary instances that satisfy every required
        # related class via a verified scene-graph edge (when strong relational
        # hints are present) or centroid-distance fallback.
        # Single-class queries (e.g. "persons") pass through unchanged.
        _img_pil     = Image.open(io.BytesIO(image_bytes))
        _img_w, _img_h = _img_pil.size
        instances = filter_compositional_instances(
            instances, parsed_query, _img_w, _img_h, scene_graph
        )

        matched_count = sum(1 for i in instances if i["matched"])
        edge_count    = len(scene_graph.verified_edges())
        logger.info(
            f"[{query_id}] Vision complete: "
            f"{len(instances)} detected ({matched_count} matched across "
            f"{len(parsed_query['targets'])} target(s)), "
            f"{edge_count} verified edges"
        )
    except Exception as e:
        logger.error(f"[{query_id}] Vision pipeline failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Vision pipeline error: {e}")

    # ── 6. Render ─────────────────────────────────────────────────────────────

    groups = build_groups_payload(parsed_query, instances)

    try:
        filename, _ = render_results(image_bytes, groups, RESULTS_DIR)
        logger.info(f"[{query_id}] Rendered result image: {filename}")
    except Exception as e:
        logger.error(f"[{query_id}] Renderer failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Rendering error: {e}")

    # ── 7. Log to DB ──────────────────────────────────────────────────────────

    inference_ms  = int((time.time() - start_time) * 1000)
    total_matches = sum(g["matchCount"] for g in groups)

    primary_target = next(
        (t for t in parsed_query["targets"] if t["role"] == "primary"),
        parsed_query["targets"][0],
    )

    try:
        await insert_query_log(
            db            = db,
            id            = query_id,
            prompt        = prompt,
            primary_class = primary_target["base_class"],
            target_count  = len(parsed_query["targets"]),
            parsed_query  = parsed_query,
            match_count   = total_matches,
            inference_ms  = inference_ms,
            result_image  = filename,
        )
    except Exception as e:
        logger.warning(f"[{query_id}] DB log failed (non-fatal): {e}")

    # ── 8. Response ───────────────────────────────────────────────────────────

    logger.info(f"[{query_id}] Done in {inference_ms}ms — {total_matches} matches")

    return JSONResponse({
        "id":                  query_id,
        "prompt":              prompt,
        "parsed_query":        parsed_query,
        "groups":              groups,
        "annotated_image_url": f"/results/{filename}",
        "inference_ms":        inference_ms,
        "created_at":          _utc_now(),
        # CHANGED: scene graph added to response
        "scene_graph":         scene_graph.to_dict(),
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
    """
    row = await fetch_query_by_id(db, query_id)

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Query '{query_id}' not found.",
        )

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
    """
    page_size = max(1, min(page_size, 100))
    page      = max(1, page)

    data = await fetch_query_history(db, page=page, page_size=page_size)

    for entry in data["entries"]:
        if entry.get("result_image"):
            entry["annotated_image_url"] = f"/results/{entry['result_image']}"

    return JSONResponse(data)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()