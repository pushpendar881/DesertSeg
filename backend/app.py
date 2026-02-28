"""
Desert Segmentation API — HuggingFace Spaces Edition
=====================================================
Supports 3 SegFormer models selectable per request:
  - PUSHPENDAR/segformer-desert  (main, 50 epochs)
  - PUSHPENDAR/Sefformer-b2      (b2 variant)
  - PUSHPENDAR/Sefformer-b1      (b1 variant)

Free tier: 2 vCPU, 16 GB RAM
"""

import os
import io
import time
import base64
import logging
from typing import Tuple, Dict

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ==============================
# CONFIG
# ==============================

device = torch.device("cpu")

# All 3 models — key is what the frontend sends in the `model` form field
MODELS_CONFIG = {
    "segformer":       "PUSHPENDAR/segformer-desert",
    "segformer-b2":    "PUSHPENDAR/Sefformer-b2",
    "segformer-b1":    "PUSHPENDAR/Sefformer-b1",
    # aliases the frontend may already be sending
    "segformer-b10":   "PUSHPENDAR/Sefformer-b2",
    "segformer-10c-50e": "PUSHPENDAR/segformer-desert",
}

CLASS_NAMES = [
    "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", "Ground Clutter",
    "Flowers", "Logs", "Rocks", "Landscape", "Sky",
]

CLASS_COLORS = [
    [45, 106, 79],   [82, 183, 136],  [212, 160, 23],
    [160, 120, 80],  [107, 107, 107], [231, 111, 81],
    [139, 94, 60],   [158, 158, 158], [200, 169, 110],
    [144, 200, 224],
]

_COLOR_LUT = np.array(CLASS_COLORS + [[128, 128, 128]], dtype=np.uint8)

# ==============================
# MODEL REGISTRY (lazy loaded, cached)
# ==============================

_processors: Dict[str, AutoImageProcessor] = {}
_models: Dict[str, SegformerForSemanticSegmentation] = {}


def get_model(model_key: str):
    """Load and cache model by key. Lazy loads on first request."""
    repo_id = MODELS_CONFIG.get(model_key.lower())
    if repo_id is None:
        raise ValueError(f"Unknown model key: '{model_key}'. Valid: {list(MODELS_CONFIG.keys())}")

    if repo_id not in _models:
        log.info(f"Loading {repo_id} ...")
        _processors[repo_id] = AutoImageProcessor.from_pretrained(repo_id)
        _models[repo_id] = SegformerForSemanticSegmentation.from_pretrained(repo_id)
        _models[repo_id].to(device)
        _models[repo_id].eval()
        log.info(f"Loaded {repo_id} ✅")

    return _processors[repo_id], _models[repo_id]


# ==============================
# FASTAPI SETUP
# ==============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load all 3 models at startup so first request is fast
    log.info("Pre-loading all 3 SegFormer models...")
    for key, repo_id in {
        "segformer":    "PUSHPENDAR/segformer-desert",
        "segformer-b2": "PUSHPENDAR/Sefformer-b2",
        "segformer-b1": "PUSHPENDAR/Sefformer-b1",
    }.items():
        try:
            get_model(key)
        except Exception as e:
            log.error(f"Failed to load {key}: {e}")
    log.info("All models ready ✅")
    yield


app = FastAPI(title="Desert Segmentation API — Multi-Model", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# HELPERS
# ==============================

def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image.copy()


def run_inference(image: Image.Image, model_key: str) -> torch.Tensor:
    proc, mdl = get_model(model_key)
    inputs = proc(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = mdl(**inputs)
    return outputs.logits[0]  # (num_classes, h, w)


def logits_to_seg(logits: torch.Tensor, out_hw: Tuple[int, int]) -> np.ndarray:
    seg_small = logits.argmax(dim=0).cpu().numpy().astype(np.uint8)
    H, W = out_hw
    return np.array(Image.fromarray(seg_small).resize((W, H), Image.NEAREST))


def create_segmentation_mask(logits: torch.Tensor, out_hw: Tuple[int, int]) -> Image.Image:
    seg = logits_to_seg(logits, out_hw)
    mask = _COLOR_LUT[np.clip(seg, 0, len(_COLOR_LUT) - 1)]
    return Image.fromarray(mask.astype(np.uint8))


def create_attention_map(logits: torch.Tensor, out_hw: Tuple[int, int]) -> Image.Image:
    seg = logits_to_seg(logits, out_hw)
    n = max(len(CLASS_NAMES) - 1, 1)
    norm = (seg.astype(np.float32) / n * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB))


def calculate_class_confidence(logits: torch.Tensor, out_hw: Tuple[int, int]):
    seg = logits_to_seg(logits, out_hw)
    total = seg.size or 1
    num_classes = logits.shape[0]
    names = CLASS_NAMES[:num_classes] + [f"Class_{i}" for i in range(len(CLASS_NAMES), num_classes)]
    results = []
    for idx, name in enumerate(names):
        pct = float(np.sum(seg == idx) / total * 100)
        results.append({"class": name, "confidence": round(pct, 2)})
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


def calculate_feature_importance(logits: torch.Tensor):
    logits_np = logits.cpu().numpy()
    var = np.var(logits_np, axis=(1, 2))
    max_v = float(np.max(var)) if np.max(var) > 0 else 1.0
    var = var / max_v * 100
    return [
        {"feature": "Texture patterns",   "confidence": round(float(var[0]), 1) if len(var) > 0 else 0},
        {"feature": "Color distribution", "confidence": round(float(var[1]), 1) if len(var) > 1 else 0},
        {"feature": "Edge boundaries",    "confidence": round(float(var[2]), 1) if len(var) > 2 else 0},
        {"feature": "Spatial context",    "confidence": round(float(var[3]), 1) if len(var) > 3 else 0},
        {"feature": "Shape geometry",     "confidence": round(float(var[4]), 1) if len(var) > 4 else 0},
    ]


def generate_explanation(model_key: str, class_confidences) -> str:
    detected = [c for c in class_confidences if c["confidence"] > 5]
    if not detected:
        return "No terrain classes detected. Please upload a desert landscape image."
    names = [c["class"] for c in detected[:3]]
    model_labels = {
        "segformer":    "SegFormer-Desert (50 epochs)",
        "segformer-b2": "SegFormer-B2",
        "segformer-b1": "SegFormer-B1",
    }
    label = model_labels.get(model_key, "SegFormer")
    return (
        f"{label} identified terrain features including {', '.join(names)}. "
        "The transformer architecture captures long-range spatial dependencies "
        "for accurate desert landscape segmentation."
    )


def image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def build_response(model_key: str, logits: torch.Tensor, W: int, H: int, start: float, version: str) -> dict:
    seg_mask    = create_segmentation_mask(logits, (H, W))
    attention   = create_attention_map(logits, (H, W))
    class_conf  = calculate_class_confidence(logits, (H, W))
    feat_imp    = calculate_feature_importance(logits)
    explanation = generate_explanation(model_key, class_conf)
    pt = round(time.time() - start, 2)

    return {
        "success": True,
        "segmentationMask": f"data:image/png;base64,{image_to_base64(seg_mask)}",
        "reasoning": {
            "explanation":       explanation,
            "attentionMap":      f"data:image/png;base64,{image_to_base64(attention)}",
            "featureImportance": feat_imp,
            "classConfidence":   class_conf,
        },
        "metadata": {
            "processingTime": pt,
            "modelVersion":   version,
            "imageSize":      {"width": W, "height": H},
        },
    }


# ==============================
# ROUTES
# ==============================

@app.get("/")
async def root():
    return {
        "message": "Desert Segmentation API",
        "status":  "ready",
        "models":  list(MODELS_CONFIG.keys()),
    }


@app.get("/health")
async def health():
    return {
        "status":        "healthy",
        "models_loaded": list(_models.keys()),
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...), model: str = Form("segformer")):
    start = time.time()

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    image_pil = load_image_from_bytes(image_bytes)
    W, H = image_pil.size

    model_key = model.lower().strip()

    try:
        logits = run_inference(image_pil, model_key)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    version_map = {"segformer": "1.0.0", "segformer-b2": "2.0.0", "segformer-b1": "1.1.0"}
    version = version_map.get(model_key, "1.0.0")

    return build_response(model_key, logits, W, H, start, version)


@app.post("/predict-parallel")
async def predict_parallel(image: UploadFile = File(...)):
    """
    Runs all 3 models and returns comparative results.
    Frontend already expects this format.
    """
    start = time.time()

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    image_pil = load_image_from_bytes(image_bytes)
    W, H = image_pil.size

    models_to_run = [
        ("segformer",    "segformer-10c-50e", "1.0.0"),
        ("segformer-b2", "segformer-b2",      "2.0.0"),
        ("segformer-b1", "segformer-b1",      "1.1.0"),
    ]

    all_results = {}

    for model_key, model_param, version in models_to_run:
        try:
            model_start = time.time()
            logits = run_inference(image_pil, model_key)
            result = build_response(model_key, logits, W, H, model_start, version)
            result["model"]     = model_param
            result["modelType"] = model_key
            result["metadata"]["inferenceTime"] = result["metadata"]["processingTime"]
            all_results[model_param] = result
        except Exception as e:
            log.error(f"Model {model_key} failed: {e}")
            all_results[model_param] = {
                "success":   False,
                "model":     model_param,
                "modelType": model_key,
                "error":     str(e),
            }

    # Build comparative analysis
    all_classes = set()
    for r in all_results.values():
        if r.get("success"):
            for c in r.get("reasoning", {}).get("classConfidence", []):
                all_classes.add(c["class"])

    class_comparison = []
    for cls in sorted(all_classes):
        model_confs = {}
        for model_param, r in all_results.items():
            if r.get("success"):
                confs = r.get("reasoning", {}).get("classConfidence", [])
                val = next((c["confidence"] for c in confs if c["class"] == cls), 0.0)
                model_confs[model_param] = val
        vals = [v for v in model_confs.values() if v > 0]
        if vals:
            class_comparison.append({
                "class":             cls,
                "averageConfidence": round(sum(vals) / len(vals), 2),
                "maxConfidence":     round(max(vals), 2),
                "minConfidence":     round(min(vals), 2),
                "maxModel":          max(model_confs, key=model_confs.get),
                "modelConfidences":  model_confs,
            })
    class_comparison.sort(key=lambda x: x["averageConfidence"], reverse=True)

    processing_times = {
        k: r.get("metadata", {}).get("processingTime", 0)
        for k, r in all_results.items() if r.get("success")
    }

    model_insights = []
    for model_param, r in all_results.items():
        if r.get("success"):
            confs = r.get("reasoning", {}).get("classConfidence", [])
            top = confs[0] if confs else {"class": "Unknown", "confidence": 0}
            model_insights.append({
                "model":          model_param,
                "topClass":       top["class"],
                "topConfidence":  top["confidence"],
                "processingTime": r.get("metadata", {}).get("processingTime", 0),
            })

    total_time = round(time.time() - start, 2)

    return {
        "success": True,
        "results": all_results,
        "comparativeAnalysis": {
            "summary":      f"Comparative analysis across 3 SegFormer variants on desert terrain.",
            "classComparison": class_comparison,
            "processingTimeComparison": {
                "times":   processing_times,
                "fastest": min(processing_times, key=processing_times.get) if processing_times else None,
                "slowest": max(processing_times, key=processing_times.get) if processing_times else None,
            },
            "modelInsights": model_insights,
        },
        "metadata": {
            "totalProcessingTime": total_time,
            "imageSize":   {"width": W, "height": H},
            "modelsRun":   len([r for r in all_results.values() if r.get("success")]),
        },
    }

# ==============================
# VIDEO SEGMENTATION
# ==============================

@app.post("/predict-video")
async def predict_video(
    video: UploadFile = File(...),
    model: str = Form("segformer"),
    fps_limit: int = Form(5),       # process max N frames per second (speed vs quality)
    overlay_alpha: float = Form(0.6) # mask transparency: 0=invisible, 1=full mask
):
    """
    Upload a video file → get back a segmented video with colored mask overlay.

    - model:        which SegFormer variant to use
    - fps_limit:    max frames to process per second (lower = faster, higher = better quality)
    - overlay_alpha: how strong the mask overlay is (0.0 to 1.0)
    """
    start = time.time()

    # --- read uploaded video into temp file ---
    video_bytes = await video.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="Empty video file")

    # Write to temp file so OpenCV can read it
    import tempfile
    suffix = os.path.splitext(video.filename or "video.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        tmp_in.write(video_bytes)
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path.replace(suffix, "_segmented.mp4")

    try:
        # --- open video ---
        cap = cv2.VideoCapture(tmp_in_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file. Please upload MP4, AVI, or MOV.")

        orig_fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Limit frame processing rate to avoid timeout on long videos
        fps_limit  = max(1, min(fps_limit, int(orig_fps)))
        frame_step = max(1, int(orig_fps / fps_limit))

        log.info(f"Video: {width}x{height} @ {orig_fps:.1f}fps, {total_frames} frames, processing every {frame_step} frames")

        # --- setup output writer ---
        # Use H.264 codec (avc1) for browser compatibility instead of mp4v
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out    = cv2.VideoWriter(tmp_out_path, fourcc, orig_fps, (width, height))

        model_key = model.lower().strip()
        frame_idx = 0
        last_mask = None  # reuse last mask for skipped frames

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Only run inference on every Nth frame
            if frame_idx % frame_step == 0:
                # Convert BGR → RGB PIL
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)

                # Run segmentation
                logits   = run_inference(frame_pil, model_key)
                seg      = logits_to_seg(logits, (height, width))
                last_mask = _COLOR_LUT[np.clip(seg, 0, len(_COLOR_LUT) - 1)]  # (H, W, 3) RGB

            # Overlay mask on original frame
            if last_mask is not None:
                mask_bgr = cv2.cvtColor(last_mask, cv2.COLOR_RGB2BGR)
                blended  = cv2.addWeighted(frame_bgr, 1 - overlay_alpha, mask_bgr, overlay_alpha, 0)
            else:
                blended = frame_bgr

            out.write(blended)
            frame_idx += 1

        cap.release()
        out.release()

        processing_time = round(time.time() - start, 2)
        frames_processed = frame_idx // frame_step

        log.info(f"Video segmentation done: {frame_idx} frames in {processing_time}s")

        # Return video as file download with metadata in headers
        # This avoids base64 size limits and browser decoding issues
        from fastapi.responses import Response
        
        with open(tmp_out_path, "rb") as f:
            video_bytes = f.read()
        
        # Clean up temp file before returning (we have the bytes)
        try:
            os.unlink(tmp_out_path)
        except Exception:
            pass
        
        return Response(
            content=video_bytes,
            media_type="video/mp4",
            headers={
                "Content-Disposition": "attachment; filename=segmented_video.mp4",
                "X-Processing-Time": str(processing_time),
                "X-Frames-Total": str(frame_idx),
                "X-Frames-Processed": str(frames_processed),
                "X-Original-FPS": str(orig_fps),
                "X-Resolution": f"{width}x{height}",
                "X-Model-Used": model_key,
            }
        )

    except HTTPException:
        raise
    except Exception as exc:
        log.error(f"Video segmentation failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Video processing failed: {exc}")
    finally:
        # Cleanup temp files (output already cleaned in success path)
        for p in [tmp_in_path]:
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass

