"""
scripts/download_models.py

Downloads SAM 3 and CLIP model weights into the models/ directory.

Run this once before starting the server:
    python scripts/download_models.py

Requirements:
  - huggingface_hub installed  (pip install huggingface_hub)
  - Logged in to HuggingFace  (huggingface-cli login)
  - Access approved at        https://huggingface.co/facebook/sam3
"""

import sys
import os
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR   = PROJECT_ROOT / "models"

# ── Helpers ───────────────────────────────────────────────────────────────────

def banner(text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

def success(text: str) -> None:
    print(f"  ✓  {text}")

def info(text: str) -> None:
    print(f"  →  {text}")

def error(text: str) -> None:
    print(f"  ✗  {text}")

def _unwrap_hf_http_error(exc: BaseException):
    """Hub often wraps 401/403 in LocalEntryNotFoundError; recover the HTTP error."""
    from huggingface_hub.errors import HfHubHTTPError

    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, HfHubHTTPError):
            return cur
        cur = cur.__cause__ or cur.__context__
    return None

def _print_sam3_access_help(http_err) -> None:
    error("Hugging Face blocked this download (authentication / gated model).")
    error("  1. Open https://huggingface.co/facebook/sam3 and accept the license.")
    error("  2. Log in:  huggingface-cli login")
    error("     Or set env:  export HF_TOKEN=<your_token>")
    error("  3. Fine-grained tokens: at https://huggingface.co/settings/tokens enable")
    error('     "Access public gated repositories" (or use a classic "Read" token).')
    if http_err is not None:
        detail = str(http_err).strip().split("\n")[0]
        if detail:
            error(f"  Hub said: {detail[:200]}")

# ── SAM 3 ─────────────────────────────────────────────────────────────────────

def download_sam3() -> bool:
    """
    Downloads SAM 3 checkpoint from Hugging Face into models/sam3/.

    Requires:
      1. huggingface-cli login  (run in terminal before this script)
      2. Access approved at https://huggingface.co/facebook/sam3
    """
    banner("Downloading SAM 3 weights")

    sam3_dir = MODELS_DIR / "sam3"
    sam3_dir.mkdir(parents=True, exist_ok=True)

    # Must match facebook/sam3 on the Hub (see sam3.model_builder.download_ckpt_from_hf)
    checkpoint_path = sam3_dir / "sam3.pt"

    if checkpoint_path.exists():
        success(f"SAM 3 weights already present at {checkpoint_path}")
        return True

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

        info("Downloading sam3.pt from facebook/sam3 ...")
        info("This is a large checkpoint — may take several minutes depending on your connection.")

        downloaded = hf_hub_download(
            repo_id   = "facebook/sam3",
            filename  = "sam3.pt",
            local_dir = str(sam3_dir),
            token     = True,
        )

        success(f"SAM 3 saved to: {downloaded}")
        return True

    # LocalEntryNotFoundError subclasses HfHubHTTPError with response=None; real 401/403 is on __cause__.
    except LocalEntryNotFoundError as e:
        inner = e.__cause__ if isinstance(e.__cause__, HfHubHTTPError) else None
        code = getattr(getattr(inner, "response", None), "status_code", None)
        if code in (401, 403) or (inner and "403" in str(inner)):
            _print_sam3_access_help(inner)
        else:
            error(f"Download failed: {e}")
            if inner:
                error(f"  (caused by: {inner})")
        return False

    except HfHubHTTPError as e:
        code = getattr(e.response, "status_code", None)
        if code in (401, 403):
            _print_sam3_access_help(e)
        else:
            error(f"Download failed: {e}")
        return False

    except Exception as e:
        http_err = _unwrap_hf_http_error(e)
        if http_err and getattr(http_err.response, "status_code", None) in (401, 403):
            _print_sam3_access_help(http_err)
            return False
        error(f"Unexpected error: {e}")
        return False


# ── CLIP ──────────────────────────────────────────────────────────────────────

def download_clip() -> bool:
    """
    Downloads CLIP ViT-L/14 weights via open_clip.
    open_clip auto-downloads on first use and caches them.
    We trigger that download here so it happens at setup time,
    not mid-inference during a demo.
    """
    banner("Downloading CLIP ViT-L/14 weights")

    clip_dir = MODELS_DIR / "clip"
    clip_dir.mkdir(parents=True, exist_ok=True)

    try:
        import open_clip

        info("Loading CLIP ViT-L/14 (openai) — ~900 MB on first download...")
        info(f"Caching to: {clip_dir}")

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14",
            pretrained="openai",
            cache_dir=str(clip_dir),
        )

        def get_device() -> str:
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        device = get_device() 
        model  = model.to(device).eval()

        # Quick smoke test
        import PIL.Image, numpy as np
        dummy_img   = PIL.Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        tokenizer   = open_clip.get_tokenizer("ViT-L-14")
        img_tensor  = preprocess(dummy_img).unsqueeze(0).to(device)
        txt_tensor  = tokenizer(["test"]).to(device)

        with torch.inference_mode():
            img_feat = model.encode_image(img_tensor)
            txt_feat = model.encode_text(txt_tensor)

        success(f"CLIP ViT-L/14 loaded and verified on {device}")
        success(f"Image features shape: {img_feat.shape}")
        return True

    except Exception as e:
        error(f"CLIP download failed: {e}")
        return False


# ── RAM++ ─────────────────────────────────────────────────────────────────────

def download_ram() -> bool:
    """
    Downloads RAM++ (Recognize Anything Plus) weights from Hugging Face into
    models/ram/ram_plus_swin_large_14m.pth  (~3 GB).

    Model card: https://huggingface.co/xinyu1205/recognize-anything-plus-model
    """
    banner("Downloading RAM++ weights")

    ram_dir = MODELS_DIR / "ram"
    ram_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = ram_dir / "ram_plus_swin_large_14m.pth"

    if checkpoint_path.exists():
        success(f"RAM++ weights already present at {checkpoint_path}")
        return True

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

        info("Downloading ram_plus_swin_large_14m.pth from xinyu1205/recognize-anything-plus-model ...")
        info("This is ~3 GB — may take several minutes depending on your connection.")

        downloaded = hf_hub_download(
            repo_id   = "xinyu1205/recognize-anything-plus-model",
            filename  = "ram_plus_swin_large_14m.pth",
            local_dir = str(ram_dir),
        )

        success(f"RAM++ saved to: {downloaded}")
        return True

    except LocalEntryNotFoundError as e:
        inner = e.__cause__ if isinstance(e.__cause__, HfHubHTTPError) else None
        error(f"Download failed (file not found locally or offline): {e}")
        if inner:
            error(f"  (caused by: {inner})")
        return False

    except HfHubHTTPError as e:
        error(f"Download failed (HTTP error): {e}")
        return False

    except Exception as e:
        error(f"Unexpected error downloading RAM++: {e}")
        return False


# ── SAM 3 smoke test ──────────────────────────────────────────────────────────

def verify_sam3() -> bool:
    """
    Quick smoke test to confirm SAM 3 can be imported and the checkpoint
    file can be loaded without running full inference.
    """
    banner("Verifying SAM 3 installation")

    sam3_checkpoint = MODELS_DIR / "sam3" / "sam3.pt"

    if not sam3_checkpoint.exists():
        error("SAM 3 weights not found — skipping verification.")
        return False

    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        info("Loading SAM 3 model from checkpoint...")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Locate BPE vocab asset — lives inside the sam3 package directory
        import sam3 as sam3_pkg
        bpe_path = Path(sam3_pkg.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"

        model = build_sam3_image_model(
            bpe_path        = str(bpe_path),
            device          = device,
            eval_mode       = True,
            checkpoint_path = str(sam3_checkpoint),
            load_from_HF    = False,
        )
        processor = Sam3Processor(model, device=device)

        success(f"SAM 3 loaded successfully on {device}")
        success("Processor ready")
        return True

    except ImportError:
        error("sam3 package not installed.")
        error("Run:  git clone https://github.com/facebookresearch/sam3.git")
        error("Then: cd sam3 && pip install -e .")
        return False

    except Exception as e:
        error(f"SAM 3 verification failed: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nMultimodel Visual Query — Model Setup")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Models dir   : {MODELS_DIR}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device   : {torch.cuda.get_device_name(0)}")

    sam3_ok = download_sam3()
    clip_ok = download_clip()
    ram_ok  = download_ram()

    if sam3_ok:
        verify_sam3()

    banner("Summary")
    print(f"  SAM 3 : {'✓ ready' if sam3_ok else '✗ failed — see errors above'}")
    print(f"  CLIP  : {'✓ ready' if clip_ok else '✗ failed — see errors above'}")
    print(f"  RAM++ : {'✓ ready' if ram_ok  else '✗ failed — see errors above'}")

    if sam3_ok and clip_ok and ram_ok:
        print("\n  All models ready. You can now run:  bash run.sh")
    else:
        print("\n  Fix the errors above before starting the server.")
        sys.exit(1)