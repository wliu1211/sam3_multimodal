import os
import time
import torch
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Must run before any custom imports so that modules like parser.py and
# vision.py can read ANTHROPIC_API_KEY from the environment at import time
# when they instantiate their Anthropic clients at module level.
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from db.database import init_db
from api.routes import router
from middleware.logger import log_requests, get_logger

logger = get_logger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
RESULTS_DIR   = PROJECT_ROOT / "storage" / "results"
UPLOADS_DIR   = PROJECT_ROOT / "storage" / "uploads"
MODELS_DIR    = PROJECT_ROOT / "models"
LOGS_DIR      = PROJECT_ROOT / "logs"

# Ensure all runtime directories exist on startup
for directory in [RESULTS_DIR, UPLOADS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ── Model store ───────────────────────────────────────────────────────────────

# Global dict that holds loaded model instances.
# Routes access models via request.app.state.models["sam3"] etc.
# Loading here (once at startup) rather than per-request saves ~15s per query.
models: dict = {}

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs startup logic before the server accepts requests,
    and teardown logic after it shuts down.

    Startup order:
      1. Initialise SQLite database + schema
      2. Load SAM 3 model weights into memory
      3. Load CLIP model weights into memory
    """

    # 1. Database
    print("[startup] Initialising database...")
    await init_db()

    # 2. SAM 3
    logger.info("[startup] Loading SAM 3...")
    t0 = time.time()
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        from pathlib import Path as _Path

        sam3_checkpoint = MODELS_DIR / "sam3" / "sam3.pt"

        if not sam3_checkpoint.exists():
            raise FileNotFoundError(
                f"SAM 3 weights not found at {sam3_checkpoint}. "
                "Run: python scripts/download_models.py"
            )

        # sam3 is a namespace package (__file__ is None), so derive the assets
        # path from model_builder.__file__ which has a real file path.
        bpe_path = _Path(build_sam3_image_model.__module__.replace(".", "/")).parent
        bpe_path = _Path(build_sam3_image_model.__code__.co_filename).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        def get_device() -> str:
            if torch.cuda.is_available():
                return "cuda"
            # MPS (Apple Silicon) is skipped — SAM3's vision_backbone has
            # submodules that initialize weights on CPU and are never moved,
            # causing an MPSFloatType vs torch.FloatTensor mismatch at inference.
            return "cpu"

        device = get_device()

        sam3_model = build_sam3_image_model(
            bpe_path        = str(bpe_path),
            device          = device,
            eval_mode       = True,
            checkpoint_path = str(sam3_checkpoint),
            load_from_HF    = False,
        )
        # Checkpoint weights are bfloat16; CPU cannot auto-cast between dtypes
        # the way CUDA does with torch.autocast, so we normalise everything to
        # float32 here to avoid "BFloat16 and Float" matmul mismatches.
        if device == "cpu":
            sam3_model = sam3_model.float()
        # Store the Sam3Processor — this is what vision.py uses
        models["sam3"] = Sam3Processor(sam3_model, device=device)
        logger.info(f"[startup] SAM 3 loaded in {time.time() - t0:.1f}s on {device}")

    except Exception as e:
        logger.warning(f"[startup] SAM 3 failed to load — {e}")
        models["sam3"] = None

    # 3. CLIP
    print("[startup] Loading CLIP...")
    t1 = time.time()
    try:
        import open_clip

        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained="openai",
            cache_dir=str(MODELS_DIR / "clip"),
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model = clip_model.to(device).eval()

        models["clip"]          = clip_model
        models["clip_preprocess"] = clip_preprocess
        models["clip_tokenizer"]  = open_clip.get_tokenizer("ViT-L-14")
        models["device"]          = device

        print(f"[startup] CLIP loaded in {time.time() - t1:.1f}s on {device}")

    except Exception as e:
        print(f"[startup] WARNING: CLIP failed to load — {e}")
        models["clip"] = None

    # Make models accessible via app.state
    app.state.models = models

    gpu_status = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"[startup] Server ready · {gpu_status} · SAM3={'✓' if models.get('sam3') else '✗'} · CLIP={'✓' if models.get('clip') else '✗'}")

    yield  # Server is now running and accepting requests

    # Teardown
    print("[shutdown] Cleaning up...")
    models.clear()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Semantic Visual Query",
    description="Natural language object detection and segmentation using SAM 3 + CLIP.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────

# In development, Vite runs on port 5173.
# In production, replace with your actual deployed frontend URL.
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Log every HTTP request + response status and duration
app.middleware("http")(log_requests)

# ── Static files ──────────────────────────────────────────────────────────────

# Serves annotated result images directly to the React frontend.
# e.g. GET /results/abc123.jpg → storage/results/abc123.jpg
app.mount(
    "/results",
    StaticFiles(directory=str(RESULTS_DIR)),
    name="results",
)

# ── Routes ────────────────────────────────────────────────────────────────────

# All API routes are defined in api/routes.py and registered under /api prefix
app.include_router(router, prefix="/api")


# ── Health check ─────────────────────────────────────────────────────────────

@app.get("/api/health", tags=["meta"])
async def health():
    """
    Quick liveness check. Returns model load status and hardware info.
    The frontend polls this on startup to know when models are ready.
    """
    return {
        "status":       "ok",
        "models_loaded": {
            "sam3": models.get("sam3") is not None,
            "clip": models.get("clip") is not None,
        },
        "gpu_available": torch.cuda.is_available(),
        "device":        models.get("device", "cpu"),
    }