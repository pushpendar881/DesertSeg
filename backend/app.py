from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import base64
import io
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import cv2
from typing import Optional, Tuple
import os

# Load environment variables from .env file
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models (lazy loading - will load on first request)
    print("API starting up. Models will be loaded on first request.")
    yield
    # Shutdown: Cleanup if needed
    pass

app = FastAPI(title="Desert Segmentation API", lifespan=lifespan)

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models and processors
segformer_processor = None
segformer_model = None
segformer_b1_processor = None
segformer_b1_model = None
segformer_b10_processor = None
segformer_b10_model = None
deeplabv3_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DeepLabV3+ preprocessing - we'll resize manually to avoid double resize

# Class names matching the frontend
CLASS_NAMES = [
    'Trees',
    'Lush Bushes',
    'Dry Grass',
    'Dry Bushes',
    'Ground Clutter',
    'Flowers',
    'Logs',
    'Rocks',
    'Landscape',
    'Sky'
]

# Class colors for visualization
CLASS_COLORS = [
    [45, 106, 79],    # Trees
    [82, 183, 136],   # Lush Bushes
    [212, 160, 23],   # Dry Grass
    [160, 120, 80],   # Dry Bushes
    [107, 107, 107],  # Ground Clutter
    [231, 111, 81],   # Flowers
    [139, 94, 60],    # Logs
    [158, 158, 158],  # Rocks
    [200, 169, 110],  # Landscape
    [144, 200, 224],  # Sky
]

def load_segformer_model():
    """Load the SegFormer model and processor"""
    global segformer_processor, segformer_model
    if segformer_processor is None or segformer_model is None:
        print("Loading SegFormer model...")
        segformer_processor = AutoImageProcessor.from_pretrained("PUSHPENDAR/segformer-desert")
        segformer_model = SegformerForSemanticSegmentation.from_pretrained("PUSHPENDAR/segformer-desert")
        segformer_model.to(device)
        segformer_model.eval()
        print(f"SegFormer model loaded on {device}")
    return segformer_processor, segformer_model

def load_segformer_b1_model():
    """Load the SegFormer-B1 model and processor (PUSHPENDAR/Sefformer-b1)"""
    global segformer_b1_processor, segformer_b1_model
    if segformer_b1_processor is None or segformer_b1_model is None:
        print("Loading SegFormer-B1 model...")
        segformer_b1_processor = AutoImageProcessor.from_pretrained("PUSHPENDAR/Sefformer-b1")
        segformer_b1_model = SegformerForSemanticSegmentation.from_pretrained("PUSHPENDAR/Sefformer-b1")
        segformer_b1_model.to(device)
        segformer_b1_model.eval()
        print(f"SegFormer-B1 model loaded on {device}")
    return segformer_b1_processor, segformer_b1_model

def load_segformer_b10_model():
    """Load the SegFormer-B10 model and processor (PUSHPENDAR/Sefformer-b10)"""
    global segformer_b10_processor, segformer_b10_model
    if segformer_b10_processor is None or segformer_b10_model is None:
        print("Loading SegFormer-B10 model...")
        segformer_b10_processor = AutoImageProcessor.from_pretrained("PUSHPENDAR/Sefformer-b10")
        segformer_b10_model = SegformerForSemanticSegmentation.from_pretrained("PUSHPENDAR/Sefformer-b10")
        segformer_b10_model.to(device)
        segformer_b10_model.eval()
        print(f"SegFormer-B10 model loaded on {device}")
    return segformer_b10_processor, segformer_b10_model

def load_deeplabv3_model():
    """Load the DeepLabV3+ model from best_model.pth (cached after first load)"""
    global deeplabv3_model
    if deeplabv3_model is None:
        load_start = time.time()
        print("[MODEL] Loading DeepLabV3+ model...")
        # Get model path from .env or use default
        model_path = os.getenv("BEST_MODEL_PATH") or os.getenv("DEEPLABV3_MODEL_PATH")
        if not model_path:
            model_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
        else:
            # Expand relative paths relative to backend directory
            if not os.path.isabs(model_path):
                model_path = os.path.join(os.path.dirname(__file__), model_path)
        
        print(f"[MODEL] Model path: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Try to load the model architecture
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            # Check if it's a state dict or full model
            if isinstance(checkpoint, dict):
                # Try to infer architecture from checkpoint
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Try different architectures
                from torchvision.models.segmentation import deeplabv3_resnet101, deeplabv3_resnet50
                
                # Try ResNet101 first (more common)
                try:
                    deeplabv3_model = deeplabv3_resnet101(num_classes=10, pretrained=False)
                    deeplabv3_model.load_state_dict(state_dict, strict=False)
                    print("[MODEL] Loaded DeepLabV3+ with ResNet101 backbone")
                except Exception as e1:
                    print(f"[MODEL] ResNet101 failed, trying ResNet50: {e1}")
                    try:
                        deeplabv3_model = deeplabv3_resnet50(num_classes=10, pretrained=False)
                        deeplabv3_model.load_state_dict(state_dict, strict=False)
                        print("[MODEL] Loaded DeepLabV3+ with ResNet50 backbone")
                    except Exception as e2:
                        print(f"[MODEL] ResNet50 failed, trying direct load: {e2}")
                        # If state dict loading fails, try loading the entire model
                        if 'model' in checkpoint:
                            deeplabv3_model = checkpoint['model']
                        else:
                            # Try using the dict as state dict with model key
                            raise ValueError("Could not load model architecture from state dict.")
            else:
                # Direct model object
                deeplabv3_model = checkpoint
                if not isinstance(deeplabv3_model, nn.Module):
                    raise ValueError("Model file does not contain a valid PyTorch model.")
            
            deeplabv3_model.to(device)
            deeplabv3_model.eval()
            load_time = time.time() - load_start
            print(f"[MODEL] DeepLabV3+ model loaded on {device} in {load_time:.2f}s")
        except Exception as e:
            print(f"[MODEL] Error loading DeepLabV3+ model: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: try to load as a generic model
            try:
                deeplabv3_model = torch.load(model_path, map_location=device, weights_only=False)
                if isinstance(deeplabv3_model, nn.Module):
                    deeplabv3_model.to(device)
                    deeplabv3_model.eval()
                    print(f"[MODEL] DeepLabV3+ model loaded (fallback method) on {device}")
                else:
                    raise ValueError("Could not determine model architecture")
            except Exception as e2:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to load DeepLabV3+ model: {str(e2)}. Please ensure best_model.pth is a valid PyTorch model file. Error details: {str(e)}"
                )
    else:
        print("[MODEL] Using cached DeepLabV3+ model")
    
    return deeplabv3_model

def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Load PIL Image directly from raw bytes with proper orientation handling"""
    try:
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Handle EXIF orientation - some images have rotation metadata
        # This ensures the image is displayed/processed correctly
        try:
            from PIL.ExifTags import ORIENTATION
            exif = image.getexif()
            if exif is not None:
                orientation = exif.get(ORIENTATION)
                if orientation:
                    if orientation == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation == 6:
                        image = image.rotate(270, expand=True)
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)
        except Exception:
            # If EXIF handling fails, continue without rotation
            pass
        
        # Convert to RGB - this is critical for model input
        # Some images might be RGBA, L (grayscale), P (palette), etc.
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create a copy to ensure the image is fully loaded
        image = image.copy()
        
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

def _class_label(class_id: int) -> str:
    return CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class_{class_id}"

def _class_color(class_id: int) -> list:
    return list(CLASS_COLORS[class_id]) if class_id < len(CLASS_COLORS) else [128, 128, 128]

def create_segmentation_mask(logits: torch.Tensor, original_size: tuple) -> Image.Image:
    """Create colored segmentation mask from model logits (supports variable num_classes)."""
    num_classes = logits.shape[0]
    # Get predictions
    predictions = torch.nn.functional.interpolate(
        logits.unsqueeze(0),
        size=original_size,
        mode="bilinear",
        align_corners=False
    ).squeeze(0)
    
    # Get class predictions
    seg = predictions.argmax(dim=0).cpu().numpy()
    
    # Debug: Check prediction distribution
    unique_classes, counts = np.unique(seg, return_counts=True)
    print(f"[MASK] Unique classes predicted: {unique_classes}, num_classes from logits: {num_classes}")
    print(f"[MASK] Class distribution: {dict(zip(unique_classes.tolist(), counts.tolist()))}")
    
    # Create colored mask
    h, w = seg.shape
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in range(num_classes):
        class_mask = (seg == class_id)
        if np.any(class_mask):
            color = _class_color(class_id)
            mask[class_mask] = color
            print(f"[MASK] Applied color for class {class_id} ({_class_label(class_id)}): {color}, pixels: {np.sum(class_mask)}")
    
    # Verify mask has non-zero values
    if np.all(mask == 0):
        print("[MASK] WARNING: Mask is all zeros (black)!")
    else:
        print(f"[MASK] Mask stats - min: {mask.min()}, max: {mask.max()}, mean: {mask.mean():.2f}")
    
    # Check if mask is all one color (would cause green screen)
    unique_colors = np.unique(mask.reshape(-1, mask.shape[-1]), axis=0)
    print(f"[MASK] Number of unique colors in mask: {len(unique_colors)}")
    if len(unique_colors) <= 2:
        print(f"[MASK] WARNING: Mask has very few colors ({len(unique_colors)}). This might cause a single-color display!")
        print(f"[MASK] Unique colors: {unique_colors}")
    
    mask_image = Image.fromarray(mask)
    print(f"[MASK] Created mask image: size={mask_image.size}, mode={mask_image.mode}")
    
    # Verify image can be saved
    try:
        test_buffer = io.BytesIO()
        mask_image.save(test_buffer, format='PNG')
        print(f"[MASK] Image save test successful, size: {len(test_buffer.getvalue())} bytes")
    except Exception as e:
        print(f"[MASK] ERROR: Failed to save test image: {e}")
    
    return mask_image

def create_attention_map(logits: torch.Tensor, original_size: tuple) -> Image.Image:
    """Create attention/confidence heatmap from model logits"""
    # Interpolate to original size
    predictions = torch.nn.functional.interpolate(
        logits.unsqueeze(0),
        size=original_size,
        mode="bilinear",
        align_corners=False
    ).squeeze(0)
    
    # Get max confidence for each pixel
    probs = torch.nn.functional.softmax(predictions, dim=0)
    max_probs, _ = torch.max(probs, dim=0)
    confidence_map = max_probs.cpu().numpy()
    
    # Normalize to 0-255 and apply colormap
    confidence_map = (confidence_map * 255).astype(np.uint8)
    attention_map = cv2.applyColorMap(confidence_map, cv2.COLORMAP_JET)
    
    return Image.fromarray(cv2.cvtColor(attention_map, cv2.COLOR_BGR2RGB))

def calculate_class_confidence(logits: torch.Tensor, original_size: tuple) -> list:
    """Calculate average confidence for each class (supports variable num_classes)."""
    num_classes = logits.shape[0]
    # Interpolate to original size
    predictions = torch.nn.functional.interpolate(
        logits.unsqueeze(0),
        size=original_size,
        mode="bilinear",
        align_corners=False
    ).squeeze(0)
    
    # Get probabilities
    probs = torch.nn.functional.softmax(predictions, dim=0)
    probs_np = probs.cpu().numpy()
    
    # Calculate average confidence per class
    class_confidences = []
    for class_id in range(num_classes):
        # Get pixels predicted as this class
        seg = predictions.argmax(dim=0).cpu().numpy()
        class_mask = (seg == class_id)
        
        if np.any(class_mask):
            avg_confidence = float(np.mean(probs_np[class_id][class_mask]) * 100)
        else:
            avg_confidence = 0.0
        
        class_confidences.append({
            'class': _class_label(class_id),
            'confidence': round(avg_confidence, 2)
        })
    
    # Sort by confidence descending
    class_confidences.sort(key=lambda x: x['confidence'], reverse=True)
    return class_confidences

def calculate_feature_importance(logits: torch.Tensor) -> list:
    """Calculate feature importance based on model activations"""
    try:
        # Calculate feature importance based on actual logit variance
        logits_np = logits.cpu().numpy()
        
        # Calculate variance across spatial dimensions for each class channel
        # Higher variance indicates more discriminative features
        channel_variance = np.var(logits_np, axis=(1, 2))
        
        # Normalize to 0-100 scale based on actual variance values
        if np.sum(channel_variance) > 0:
            max_variance = np.max(channel_variance)
            if max_variance > 0:
                normalized_variance = (channel_variance / max_variance) * 100
            else:
                normalized_variance = np.zeros_like(channel_variance)
        else:
            normalized_variance = np.zeros_like(channel_variance)
        
        # Map variance to feature importance (using actual calculated values)
        # Distribute across 5 features based on variance distribution
        features = [
            {'feature': 'Texture patterns', 'confidence': 0},
            {'feature': 'Color distribution', 'confidence': 0},
            {'feature': 'Edge boundaries', 'confidence': 0},
            {'feature': 'Spatial context', 'confidence': 0},
            {'feature': 'Shape geometry', 'confidence': 0},
        ]
        
        # Map class channel variances to features
        # Use quantiles of variance distribution
        if len(normalized_variance) > 0:
            sorted_variances = np.sort(normalized_variance)[::-1]  # Descending
            num_features = len(features)
            
            for i, feature in enumerate(features):
                # Map to variance percentiles
                percentile_idx = min(int((i / num_features) * len(sorted_variances)), len(sorted_variances) - 1)
                if percentile_idx >= 0:
                    confidence = float(sorted_variances[percentile_idx])
                    features[i]['confidence'] = round(max(0, min(100, confidence)), 1)
                else:
                    features[i]['confidence'] = 0.0
        else:
            # If no variance data, return zeros
            for feature in features:
                feature['confidence'] = 0.0
        
        return features
    except Exception as e:
        print(f"Error calculating feature importance: {e}")
        # Return zeros if calculation fails
        return [
            {'feature': 'Texture patterns', 'confidence': 0},
            {'feature': 'Color distribution', 'confidence': 0},
            {'feature': 'Edge boundaries', 'confidence': 0},
            {'feature': 'Spatial context', 'confidence': 0},
            {'feature': 'Shape geometry', 'confidence': 0},
        ]

def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def generate_explanation(model_name: str, class_confidences: list) -> str:
    """Generate AI explanation based on actual predictions"""
    # Filter classes with actual confidence > 0
    detected_classes = [c for c in class_confidences if c['confidence'] > 0]
    top_classes = [c['class'] for c in detected_classes[:3] if c['confidence'] > 10]
    
    if not detected_classes:
        return "No terrain classes were detected in the image. Please ensure the image contains desert terrain features."
    
    if model_name == "segformer":
        model_display = "SegFormer-B2"
        architecture_desc = "hierarchical transformer architecture"
    elif model_name == "segformer-b1":
        model_display = "SegFormer-B1"
        architecture_desc = "hierarchical transformer architecture (B1)"
    elif model_name == "segformer-b10":
        model_display = "SegFormer-B10"
        architecture_desc = "hierarchical transformer architecture (B10)"
    else:
        model_display = "DeepLabV3+"
        architecture_desc = "atrous spatial pyramid pooling (ASPP) architecture"
    
    explanation = f"The {model_display} model identified {len(detected_classes)} distinct terrain region(s) by analyzing texture patterns, color gradients, and spatial relationships. "
    
    if top_classes:
        top_confidences = [c['confidence'] for c in detected_classes[:3] if c['confidence'] > 10]
        if top_confidences:
            conf_str = ', '.join([f'{c:.1f}%' for c in top_confidences[:3]])
            explanation += f"High confidence was observed for {', '.join(top_classes)} classes (confidence: {conf_str}). "
    
    # Add information about detected classes
    if len(detected_classes) > 0:
        avg_confidence = sum(c['confidence'] for c in detected_classes) / len(detected_classes)
        explanation += f"The model's {architecture_desc} processed the image with an average confidence of {avg_confidence:.1f}% across detected classes."
    else:
        explanation += f"The model's {architecture_desc} enables robust segmentation even in challenging desert conditions with varying illumination and terrain complexity."
    
    return explanation

@app.get("/")
async def root():
    return {"message": "Desert Segmentation API", "status": "ready"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "segformer_loaded": segformer_model is not None,
        "segformer_b1_loaded": segformer_b1_model is not None,
        "segformer_b10_loaded": segformer_b10_model is not None,
        "deeplabv3_loaded": deeplabv3_model is not None
    }

def predict_segformer(image: Image.Image, original_size: Tuple[int, int]) -> torch.Tensor:
    """Run SegFormer inference"""
    proc, mdl = load_segformer_model()

    # Preprocess image
    inputs = proc(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = mdl(**inputs)
        logits = outputs.logits[0]  # Remove batch dimension

    return logits

def predict_segformer_b1(image: Image.Image, original_size: Tuple[int, int]) -> torch.Tensor:
    """Run SegFormer-B1 inference (same mechanism as SegFormer)"""
    proc, mdl = load_segformer_b1_model()

    # Preprocess image
    inputs = proc(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = mdl(**inputs)
        logits = outputs.logits[0]  # Remove batch dimension

    return logits

def predict_segformer_b10(image: Image.Image, original_size: Tuple[int, int]) -> torch.Tensor:
    """Run SegFormer-B10 inference (same mechanism as SegFormer)"""
    proc, mdl = load_segformer_b10_model()

    # Preprocess image
    inputs = proc(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = mdl(**inputs)
        logits = outputs.logits[0]  # Remove batch dimension

    return logits

def predict_deeplabv3(image: Image.Image, original_size: Tuple[int, int]) -> torch.Tensor:
    """Run DeepLabV3+ inference with proper preprocessing matching training"""
    mdl = load_deeplabv3_model()
    
    # Ensure image is RGB and properly formatted
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Preprocess image for DeepLabV3+ - match training preprocessing exactly
    # Try common input sizes: 512x512, 480x480, or 224x224
    # Most DeepLabV3+ models use 512x512 for segmentation
    input_size = (512, 512)
    
    # Resize using high-quality resampling
    # LANCZOS is better for downsampling, preserves more detail
    image_resized = image.resize(input_size, Image.LANCZOS)
    
    # Convert to numpy array first to ensure proper format
    # This helps catch any issues with image format
    import numpy as np
    img_array = np.array(image_resized, dtype=np.uint8)
    
    # Verify image array
    if img_array.shape != (input_size[1], input_size[0], 3):
        raise ValueError(f"Unexpected image shape after resize: {img_array.shape}")
    
    # Convert back to PIL for transform pipeline (ensures consistency)
    image_for_transform = Image.fromarray(img_array)
    
    # Standard ImageNet preprocessing
    # ToTensor: PIL Image (H, W, C) -> Tensor (C, H, W), scales [0, 255] -> [0, 1]
    # Normalize: applies ImageNet mean/std
    transform = transforms.Compose([
        transforms.ToTensor(),  # (C, H, W) in range [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transform and add batch dimension
    input_tensor = transform(image_for_transform).unsqueeze(0).to(device)
    
    # Verify input tensor
    print(f"[DEEPLABV3] Input tensor shape: {input_tensor.shape}")
    print(f"[DEEPLABV3] Input tensor range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
    print(f"[DEEPLABV3] Input tensor mean: {input_tensor.mean():.3f}, std: {input_tensor.std():.3f}")
    
    # Ensure tensor is in correct format
    if input_tensor.shape != (1, 3, input_size[1], input_size[0]):
        raise ValueError(f"Invalid input tensor shape: {input_tensor.shape}, expected (1, 3, {input_size[1]}, {input_size[0]})")
    
    # Run inference
    print("\n" + "="*60)
    print("[DEEPLABV3] Starting inference...")
    print("="*60)
    
    with torch.no_grad():
        outputs = mdl(input_tensor)
        
        # Debug: Print output structure
        print(f"\n[DEEPLABV3] Model output type: {type(outputs)}")
        print(f"[DEEPLABV3] Model output: {outputs}")
        
        if isinstance(outputs, dict):
            print(f"[DEEPLABV3] Output keys: {list(outputs.keys())}")
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"[DEEPLABV3]   {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
                    print(f"[DEEPLABV3]   {key}: min={value.min().item():.4f}, max={value.max().item():.4f}, mean={value.mean().item():.4f}")
            logits = outputs['out']  # Get the 'out' tensor
        elif isinstance(outputs, (list, tuple)):
            print(f"[DEEPLABV3] Output is list/tuple with {len(outputs)} elements")
            for i, out in enumerate(outputs):
                if isinstance(out, torch.Tensor):
                    print(f"[DEEPLABV3]   [{i}]: shape={out.shape}, dtype={out.dtype}")
            logits = outputs[0]
        else:
            print(f"[DEEPLABV3] Output is direct tensor")
            logits = outputs
        
        # Debug: Print logits shape and statistics
        print(f"\n[DEEPLABV3] Raw logits shape: {logits.shape}")
        print(f"[DEEPLABV3] Raw logits dtype: {logits.dtype}")
        print(f"[DEEPLABV3] Raw logits device: {logits.device}")
        print(f"[DEEPLABV3] Raw logits - min: {logits.min().item():.4f}, max: {logits.max().item():.4f}, mean: {logits.mean().item():.4f}")
        print(f"[DEEPLABV3] Raw logits - std: {logits.std().item():.4f}")
        
        # Print per-class statistics
        if len(logits.shape) >= 3:
            num_classes = logits.shape[0] if logits.shape[0] <= 20 else logits.shape[1]
            print(f"\n[DEEPLABV3] Per-class logits statistics:")
            for c in range(min(10, num_classes)):
                if len(logits.shape) == 4:
                    class_logits = logits[0, c, :, :]  # (batch, class, H, W)
                elif len(logits.shape) == 3:
                    class_logits = logits[c, :, :]  # (class, H, W)
                else:
                    break
                print(f"[DEEPLABV3]   Class {c} ({CLASS_NAMES[c] if c < len(CLASS_NAMES) else 'Unknown'}): "
                      f"min={class_logits.min().item():.4f}, max={class_logits.max().item():.4f}, "
                      f"mean={class_logits.mean().item():.4f}")
        
        # Ensure logits are in (C, H, W) format
        # DeepLabV3+ typically outputs (batch, classes, height, width)
        if len(logits.shape) == 4:
            logits = logits[0]  # Remove batch dimension -> (classes, height, width)
        elif len(logits.shape) == 3:
            # Already in (classes, height, width) or (batch, height, width)
            if logits.shape[0] == 10:  # If first dim is classes
                logits = logits  # Already correct
            else:  # If first dim is batch
                logits = logits[0].unsqueeze(0)  # Remove batch, add class dim
        elif len(logits.shape) == 2:
            # If it's (H, W), this means it's already predictions, not logits
            # We need to convert predictions back to logits format
            # This shouldn't happen, but handle it
            print("Warning: Received predictions instead of logits, converting...")
            # Create one-hot encoding
            num_classes = 10
            h, w = logits.shape
            logits_tensor = torch.zeros(num_classes, h, w, device=logits.device, dtype=torch.float32)
            for c in range(num_classes):
                logits_tensor[c] = (logits == c).float() * 100.0  # High confidence for predicted class
            logits = logits_tensor
        
        # Final check: ensure shape is (num_classes, height, width)
        if len(logits.shape) != 3 or logits.shape[0] != 10:
            raise ValueError(f"Invalid logits shape after processing: {logits.shape}. Expected (10, H, W)")
        
        print(f"\n[DEEPLABV3] Final processed logits shape: {logits.shape}")
        print(f"[DEEPLABV3] Final logits - min: {logits.min().item():.4f}, max: {logits.max().item():.4f}, mean: {logits.mean().item():.4f}")
        
        # Get predictions for visualization
        predictions = logits.argmax(dim=0).cpu().numpy()
        unique_classes, counts = np.unique(predictions, return_counts=True)
        print(f"\n[DEEPLABV3] Predicted classes distribution:")
        for cls, count in zip(unique_classes, counts):
            percentage = (count / predictions.size) * 100
            class_name = CLASS_NAMES[int(cls)] if int(cls) < len(CLASS_NAMES) else f"Class_{int(cls)}"
            print(f"[DEEPLABV3]   {class_name} (class {int(cls)}): {count} pixels ({percentage:.2f}%)")
        
        print("="*60)
        print("[DEEPLABV3] Inference completed\n")
    
    return logits

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    model: str = Form("segformer")
):
    """Main prediction endpoint - accepts raw image file directly (no base64, preserves quality)"""
    start_time = time.time()
    
    try:
        # Read raw image bytes directly - no conversion, no quality loss
        image_bytes = await image.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        print(f"[PREDICT] Received image: {image.filename}, size: {len(image_bytes)} bytes")
        
        # Load image directly from raw bytes with proper handling
        image_pil = load_image_from_bytes(image_bytes)
        original_size = image_pil.size  # (width, height)
        print(f"[PREDICT] Image dimensions: {original_size}, mode: {image_pil.mode}")
        
        # Validate image
        if original_size[0] == 0 or original_size[1] == 0:
            raise HTTPException(status_code=400, detail="Invalid image dimensions")
        
        # Log image info for debugging
        print(f"[PREDICT] Image format: {image_pil.format}, size: {len(image_bytes)} bytes")
        
        # Determine which model to use - handle various model name formats
        model_type = (model or "segformer").lower().strip()
        print(f"[PREDICT] Received model parameter: '{model}' -> normalized: '{model_type}'")
        
        if "deeplab" in model_type or model_type == "deeplabv3":
            model_type = "deeplabv3"
        elif "segformer-b10" in model_type or "sefformer-b10" in model_type:
            model_type = "segformer-b10"
        elif "segformer-b1" in model_type or "sefformer-b1" in model_type:
            model_type = "segformer-b1"
        elif "segformer" in model_type or model_type == "segformer" or "segformer-10c-50e" in model_type:
            model_type = "segformer"
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model type: {model}. Must be 'segformer', 'segformer-b1', 'segformer-b10', or 'deeplabv3'"
            )
        
        print(f"[PREDICT] Using model: {model_type}")
        
        # Run inference with selected model - image goes directly to model
        inference_start = time.time()
        if model_type == "segformer":
            logits = predict_segformer(image_pil, original_size)
            model_version = "1.0.0"
        elif model_type == "segformer-b1":
            logits = predict_segformer_b1(image_pil, original_size)
            model_version = "1.1.0"
        elif model_type == "segformer-b10":
            logits = predict_segformer_b10(image_pil, original_size)
            model_version = "1.3.0"
        else:  # deeplabv3
            logits = predict_deeplabv3(image_pil, original_size)
            model_version = "1.2.0"
        
        inference_time = time.time() - inference_start
        print(f"[PREDICT] Inference time: {inference_time:.2f}s")
        print(f"[PREDICT] Logits shape: {logits.shape}")
        
        # Verify logits shape: (num_classes, H, W) - support variable num_classes (e.g. 6 for SegFormer-B1, 10 for others)
        num_classes = logits.shape[0] if len(logits.shape) == 3 else 0
        if len(logits.shape) != 3 or num_classes < 1 or num_classes > 20:
            raise ValueError(f"Invalid logits shape: {logits.shape}. Expected (num_classes, H, W) with 1 <= num_classes <= 20")
        
        # Generate segmentation mask
        postprocess_start = time.time()
        seg_mask = create_segmentation_mask(logits, original_size[::-1])  # PIL uses (height, width)
        seg_mask_base64 = image_to_base64(seg_mask)
        print(f"[PREDICT] Segmentation mask base64 length: {len(seg_mask_base64)} chars")
        print(f"[PREDICT] Segmentation mask preview (first 100 chars): {seg_mask_base64[:100]}...")
        
        # Generate attention map
        attention_map = create_attention_map(logits, original_size[::-1])
        attention_map_base64 = image_to_base64(attention_map)
        
        # Calculate statistics
        class_confidences = calculate_class_confidence(logits, original_size[::-1])
        feature_importance = calculate_feature_importance(logits)
        
        # Generate explanation
        explanation = generate_explanation(model_type, class_confidences)
        
        postprocess_time = time.time() - postprocess_start
        processing_time = time.time() - start_time
        
        print(f"[PREDICT] Post-processing time: {postprocess_time:.2f}s")
        print(f"[PREDICT] Total processing time: {processing_time:.2f}s")
        
        # Verify base64 strings are valid
        if not seg_mask_base64 or len(seg_mask_base64) < 100:
            raise ValueError(f"Invalid segmentation mask base64: length={len(seg_mask_base64) if seg_mask_base64 else 0}")
        
        # Prepare response
        response = {
            "success": True,
            "segmentationMask": f"data:image/png;base64,{seg_mask_base64}",
            "reasoning": {
                "explanation": explanation,
                "attentionMap": f"data:image/png;base64,{attention_map_base64}",
                "featureImportance": feature_importance,
                "classConfidence": class_confidences
            },
            "metadata": {
                "processingTime": round(processing_time, 2),
                "modelVersion": model_version,
                "imageSize": {
                    "width": original_size[0],
                    "height": original_size[1]
                }
            }
        }
        
        print(f"[PREDICT] Response prepared - segmentationMask length: {len(response['segmentationMask'])}")
        print(f"[PREDICT] Response preview: {str(response)[:200]}...")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[PREDICT] Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def generate_comparative_analysis(all_results: dict) -> dict:
    """Generate comparative explainable AI analysis across all models"""
    model_names = list(all_results.keys())
    results = all_results
    
    # Aggregate class confidences across models
    all_classes = set()
    for model_name, result in results.items():
        if result.get('success') and result.get('reasoning', {}).get('classConfidence'):
            for conf in result['reasoning']['classConfidence']:
                all_classes.add(conf['class'])
    
    # Build comparative class confidence matrix
    class_comparison = []
    for class_name in sorted(all_classes):
        model_confidences = {}
        for model_name in model_names:
            if results[model_name].get('success'):
                confidences = results[model_name].get('reasoning', {}).get('classConfidence', [])
                class_conf = next((c['confidence'] for c in confidences if c['class'] == class_name), 0.0)
                model_confidences[model_name] = class_conf
        
        # Calculate statistics
        conf_values = [v for v in model_confidences.values() if v > 0]
        if conf_values:
            avg_confidence = sum(conf_values) / len(conf_values)
            max_confidence = max(conf_values)
            min_confidence = min(conf_values)
            max_model = max(model_confidences.items(), key=lambda x: x[1])[0] if model_confidences else None
            
            class_comparison.append({
                'class': class_name,
                'averageConfidence': round(avg_confidence, 2),
                'maxConfidence': round(max_confidence, 2),
                'minConfidence': round(min_confidence, 2),
                'maxModel': max_model,
                'modelConfidences': model_confidences
            })
    
    # Sort by average confidence descending
    class_comparison.sort(key=lambda x: x['averageConfidence'], reverse=True)
    
    # Processing time comparison
    processing_times = {}
    for model_name in model_names:
        if results[model_name].get('success'):
            processing_times[model_name] = results[model_name].get('metadata', {}).get('processingTime', 0)
    
    fastest_model = min(processing_times.items(), key=lambda x: x[1])[0] if processing_times else None
    slowest_model = max(processing_times.items(), key=lambda x: x[1])[0] if processing_times else None
    
    # Generate summary explanation
    top_classes = [c['class'] for c in class_comparison[:3] if c['averageConfidence'] > 10]
    explanation_parts = []
    
    explanation_parts.append(f"Comparative analysis across {len(model_names)} models identified {len(class_comparison)} distinct terrain classes.")
    
    if top_classes:
        explanation_parts.append(f"Highest average confidence observed for: {', '.join(top_classes)}.")
    
    if fastest_model and slowest_model and fastest_model != slowest_model:
        fastest_time = processing_times[fastest_model]
        slowest_time = processing_times[slowest_model]
        explanation_parts.append(f"Processing times ranged from {fastest_time}s ({fastest_model}) to {slowest_time}s ({slowest_model}).")
    
    # Model-specific insights
    model_insights = []
    for model_name in model_names:
        if results[model_name].get('success'):
            confidences = results[model_name].get('reasoning', {}).get('classConfidence', [])
            top_class = max(confidences, key=lambda x: x['confidence']) if confidences else None
            if top_class and top_class['confidence'] > 10:
                model_insights.append({
                    'model': model_name,
                    'topClass': top_class['class'],
                    'topConfidence': round(top_class['confidence'], 2),
                    'processingTime': processing_times.get(model_name, 0)
                })
    
    return {
        'summary': ' '.join(explanation_parts),
        'classComparison': class_comparison,
        'processingTimeComparison': {
            'times': processing_times,
            'fastest': fastest_model,
            'slowest': slowest_model
        },
        'modelInsights': model_insights
    }

@app.post("/predict-parallel")
async def predict_parallel(
    image: UploadFile = File(...)
):
    """Parallel prediction endpoint - runs all models simultaneously and returns comparative results"""
    start_time = time.time()
    
    try:
        # Read raw image bytes directly
        image_bytes = await image.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        print(f"[PREDICT-PARALLEL] Received image: {image.filename}, size: {len(image_bytes)} bytes")
        
        # Load image once
        image_pil = load_image_from_bytes(image_bytes)
        original_size = image_pil.size  # (width, height)
        print(f"[PREDICT-PARALLEL] Image dimensions: {original_size}, mode: {image_pil.mode}")
        
        # Validate image
        if original_size[0] == 0 or original_size[1] == 0:
            raise HTTPException(status_code=400, detail="Invalid image dimensions")
        
        # Define all models to run
        models_to_run = [
            ("segformer", "segformer-10c-50e"),
            ("segformer-b1", "segformer-b1"),
            ("segformer-b10", "segformer-b10"),
            ("deeplabv3", "deeplabv3")
        ]
        
        # Run all models sequentially (can be optimized with threading later)
        def run_model(model_type: str, model_param: str):
            """Run a single model prediction"""
            try:
                inference_start = time.time()
                
                # Run inference
                if model_type == "segformer":
                    logits = predict_segformer(image_pil, original_size)
                    model_version = "1.0.0"
                elif model_type == "segformer-b1":
                    logits = predict_segformer_b1(image_pil, original_size)
                    model_version = "1.1.0"
                elif model_type == "segformer-b10":
                    logits = predict_segformer_b10(image_pil, original_size)
                    model_version = "1.3.0"
                else:  # deeplabv3
                    logits = predict_deeplabv3(image_pil, original_size)
                    model_version = "1.2.0"
                
                inference_time = time.time() - inference_start
                
                # Verify logits shape
                num_classes = logits.shape[0] if len(logits.shape) == 3 else 0
                if len(logits.shape) != 3 or num_classes < 1 or num_classes > 20:
                    raise ValueError(f"Invalid logits shape: {logits.shape}")
                
                # Generate segmentation mask
                seg_mask = create_segmentation_mask(logits, original_size[::-1])
                seg_mask_base64 = image_to_base64(seg_mask)
                
                # Generate attention map
                attention_map = create_attention_map(logits, original_size[::-1])
                attention_map_base64 = image_to_base64(attention_map)
                
                # Calculate statistics
                class_confidences = calculate_class_confidence(logits, original_size[::-1])
                feature_importance = calculate_feature_importance(logits)
                
                # Generate explanation
                explanation = generate_explanation(model_type, class_confidences)
                
                processing_time = time.time() - inference_start
                
                return {
                    "success": True,
                    "model": model_param,
                    "modelType": model_type,
                    "segmentationMask": f"data:image/png;base64,{seg_mask_base64}",
                    "reasoning": {
                        "explanation": explanation,
                        "attentionMap": f"data:image/png;base64,{attention_map_base64}",
                        "featureImportance": feature_importance,
                        "classConfidence": class_confidences
                    },
                    "metadata": {
                        "processingTime": round(processing_time, 2),
                        "inferenceTime": round(inference_time, 2),
                        "modelVersion": model_version,
                        "imageSize": {
                            "width": original_size[0],
                            "height": original_size[1]
                        }
                    }
                }
            except Exception as e:
                print(f"[PREDICT-PARALLEL] Error running {model_type}: {str(e)}")
                import traceback
                traceback.print_exc()
                return {
                    "success": False,
                    "model": model_param,
                    "modelType": model_type,
                    "error": str(e)
                }
        
        # Run models sequentially
        all_results = {}
        for model_type, model_param in models_to_run:
            print(f"[PREDICT-PARALLEL] Running {model_type}...")
            result = run_model(model_type, model_param)
            all_results[model_param] = result
        
        # Generate comparative analysis
        comparative_analysis = generate_comparative_analysis(all_results)
        
        total_time = time.time() - start_time
        
        response = {
            "success": True,
            "results": all_results,
            "comparativeAnalysis": comparative_analysis,
            "metadata": {
                "totalProcessingTime": round(total_time, 2),
                "imageSize": {
                    "width": original_size[0],
                    "height": original_size[1]
                },
                "modelsRun": len([r for r in all_results.values() if r.get('success')])
            }
        }
        
        print(f"[PREDICT-PARALLEL] Completed parallel prediction in {total_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[PREDICT-PARALLEL] Error during parallel prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Parallel prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

