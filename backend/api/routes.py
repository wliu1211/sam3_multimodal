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

from services.discovery import run_discovery_pipeline
from services.activity  import build_activity_groups, groups_to_dict
from services.graph     import Node
from services.sgg       import build_scene_graph
from services.vtt       import verify_edges

logger = get_logger(__name__)

router = APIRouter()

                                                                                

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR  = PROJECT_ROOT / "storage" / "results"

                                                                                

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES       = 20 * 1024 * 1024         


CACHE_TTL_SECONDS = 600                                                
CACHE_MAX_ENTRIES = 20                                                   

                                                                               
                 
                                                                               

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

                                                                               

    models = request.app.state.models

    if models.get("sam3") is None:
        raise HTTPException(
            status_code=503,
            detail="SAM 3 model is not loaded. Check server startup logs.",
        )

    try:
                                                                        
                                                                       
                                                                
        instances, scene_graph = await run_vision_pipeline(
            image_bytes, parsed_query, models
        )

                                                                             
                                                                           
                                                                               
                                                           
                                                                       
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

                                                                                

    groups = build_groups_payload(parsed_query, instances)

    try:
        filename, _ = render_results(image_bytes, groups, RESULTS_DIR)
        logger.info(f"[{query_id}] Rendered result image: {filename}")
    except Exception as e:
        logger.error(f"[{query_id}] Renderer failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Rendering error: {e}")

                                                                                

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

                                                                                

    logger.info(f"[{query_id}] Done in {inference_ms}ms — {total_matches} matches")

    return JSONResponse({
        "id":                  query_id,
        "prompt":              prompt,
        "parsed_query":        parsed_query,
        "groups":              groups,
        "annotated_image_url": f"/results/{filename}",
        "inference_ms":        inference_ms,
        "created_at":          _utc_now(),
                                                
        "scene_graph":         scene_graph.to_dict(),
    })


                                                                               
                    
                                                                               

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


                                                                                

def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


                                                                                   
                                                     
                                                                               

@router.post("/analyze", summary="Discover all activities in an image", dependencies=[Depends(rate_limit)])
async def analyze(
    request: Request,
    image:   UploadFile = File(...),
):
    """
    Discovery pipeline — runs when the user asks an open-ended question.

    Phase 1:   RAM++ → concept list → multi-class SAM 3 → CLIP → SGG → VTT
    Phase 1.5: Activity grouping → NLP label generation

    Returns an activity list. No image rendering happens here.
    User clicks a list item → POST /mask renders that group on-demand.
    """
    query_id   = uuid.uuid4().hex
    start_time = time.time()

    logger.info(f"[{query_id}] Discovery analysis started")

                     
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported image type '{image.content_type}'.")
    image_bytes = await image.read()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 20MB limit.")
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")

    models = request.app.state.models
    if models.get("sam3") is None:
        raise HTTPException(status_code=503, detail="SAM 3 model is not loaded.")

                                                                               
    try:
        raw_instances, noun_tags, context_tags = await run_discovery_pipeline(
            image_bytes, models
        )
    except Exception as e:
        logger.error(f"[{query_id}] Discovery pipeline failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Discovery pipeline error: {e}")

    if not raw_instances:
        return JSONResponse({
            "id":           query_id,
            "mode":         "discovery",
            "activities":   [],
            "noun_tags":    noun_tags,
            "inference_ms": int((time.time() - start_time) * 1000),
            "created_at":   _utc_now(),
        })

                                                                                
    image_pil    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_w, img_h = image_pil.size

    nodes = [
        Node(
            id           = inst["id"],
            label        = inst["label"],
            bounding_box = inst["bounding_box"],
            mask_rle     = inst["mask_rle"],
            confidence   = inst["confidence"],
        )
        for inst in raw_instances
    ]

                                                                                
    try:
        scene_graph = build_scene_graph(nodes=nodes, img_w=img_w, img_h=img_h)
        scene_graph = await verify_edges(
            graph=scene_graph, image_pil=image_pil, img_w=img_w, img_h=img_h
        )
    except Exception as e:
        logger.error(f"[{query_id}] SGG/VTT failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Scene graph error: {e}")

                                                                                
    try:
        activity_groups = await build_activity_groups(
            scene_graph  = scene_graph,
            context_tags = context_tags,
            noun_tags    = noun_tags,
        )
    except Exception as e:
        logger.error(f"[{query_id}] Activity grouping failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Activity grouping error: {e}")

                                                                                
    _cache_write(request, query_id, {
        "image_bytes":   image_bytes,
        "raw_instances": raw_instances,
        "scene_graph":   scene_graph,
        "noun_tags":     noun_tags,
        "context_tags":  context_tags,
    })

    inference_ms = int((time.time() - start_time) * 1000)
    logger.info(
        f"[{query_id}] Discovery done in {inference_ms}ms — "
        f"{len(raw_instances)} instances, {len(activity_groups)} activities"
    )

    return JSONResponse({
        "id":           query_id,
        "mode":         "discovery",
        "activities":   groups_to_dict(activity_groups),
        "noun_tags":    noun_tags,
        "scene_graph":  scene_graph.to_dict(),
        "inference_ms": inference_ms,
        "created_at":   _utc_now(),
    })


                                                                               
                                                                    
                                                                               

@router.post("/mask", summary="Render masks for a specific activity group")
async def mask(
    request:  Request,
    query_id: str  = Form(..., description="ID returned by /analyze"),
    node_ids: str  = Form(..., description="Comma-separated node IDs to mask"),
    color:    str  = Form("#4ADE80", description="Hex color for the mask overlay"),
):
    """
    On-demand masking — renders masks only for the nodes in the clicked
    activity group. Retrieves pre-computed data from the discovery cache.
    No ML inference runs here, only OpenCV rendering.
    Returns the URL of the annotated result image.
    """
    cached = _cache_read(request, query_id)
    if cached is None:
        raise HTTPException(
            status_code=410,
            detail="Session expired. Please re-analyze the image.",
        )

    requested_ids = {nid.strip() for nid in node_ids.split(",") if nid.strip()}
    if not requested_ids:
        raise HTTPException(status_code=400, detail="node_ids must not be empty.")

    selected = [inst for inst in cached["raw_instances"] if inst["id"] in requested_ids]
    if not selected:
        raise HTTPException(status_code=404, detail="None of the requested node_ids were found.")

    group = {
        "label":         "selected",
        "color":          color,
        "instances":      selected,
        "matchCount":     len(selected),
        "avgConfidence":  sum(i["confidence"] for i in selected) / len(selected),
    }

    try:
        filename, _ = render_results(
            image_bytes = cached["image_bytes"],
            groups      = [group],
            results_dir = RESULTS_DIR,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rendering error: {e}")

    return JSONResponse({
        "annotated_image_url": f"/results/{filename}",
        "node_count":          len(selected),
        "color":               color,
    })


                                                                                

def _cache_write(request: Request, query_id: str, data: dict) -> None:
    """Writes to the discovery cache with TTL and max-entry enforcement."""
    import time as _time
    cache = request.app.state.discovery_cache

    now     = _time.time()
    expired = [k for k, v in cache.items() if now > v.get("expires_at", 0)]
    for k in expired:
        del cache[k]

    if len(cache) >= CACHE_MAX_ENTRIES:
        oldest = min(cache, key=lambda k: cache[k].get("expires_at", 0))
        del cache[oldest]

    cache[query_id] = {**data, "expires_at": now + CACHE_TTL_SECONDS}


def _cache_read(request: Request, query_id: str) -> dict | None:
    """Reads from the cache, returning None if missing or expired."""
    import time as _time
    cache = request.app.state.discovery_cache
    entry = cache.get(query_id)
    if entry is None:
        return None
    if _time.time() > entry.get("expires_at", 0):
        del cache[query_id]
        return None
    return entry