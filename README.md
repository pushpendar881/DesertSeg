# Desert Terrain Segmentation Application

A full-stack application for semantic segmentation of desert and off-road terrain images using deep learning models. The system provides real-time image segmentation with visualization of attention maps, feature importance, and class confidence scores.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Node.js](https://img.shields.io/badge/node.js-18+-green.svg)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Models](#models)
  - [Model 1: SegFormer-10C-50E (Nvidia-B-2)](#model-1-segformer-10c-50e-nvidia-b-2)
  - [Model 2: DeepLabV3+ (ResNET-100)](#model-2-deeplabv3-resnet-100)
  - [Model 3: SegFormer-C10 (Nvidia-B-1)](#model-3-segformer-c10-nvidia-b-1)
  - [Model 4: SegFormer-C6 (Nvidia-B1)](#model-4-segformer-c6-nvidia-b1)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This application performs semantic segmentation on desert and off-road terrain images, classifying pixels into 10 distinct categories:

| Class | Description |
|-------|-------------|
| Trees | Large vegetation and tree coverage |
| Lush Bushes | Green, healthy shrubs |
| Dry Grass | Dried or dead grass areas |
| Dry Bushes | Brown, dried shrubs |
| Ground Clutter | Debris and mixed ground elements |
| Flowers | Flowering vegetation |
| Logs | Fallen trees and wooden debris |
| Rocks | Stone and rocky terrain |
| Landscape | General terrain features |
| Sky | Sky and atmospheric regions |

---

## Features

- **Multi-Model Support**: Choose between different segmentation models
- **Real-time Inference**: Fast image processing with GPU acceleration support
- **Explainable AI**: Attention maps and feature importance visualization
- **Interactive UI**: Modern React-based frontend with drag-and-drop upload
- **RESTful API**: Well-documented FastAPI backend with Swagger UI
- **Cross-platform**: Runs on Windows, Linux, and macOS

---

## Architecture

```
┌─────────────────┐     HTTP/REST      ┌─────────────────┐
│                 │ ◄───────────────► │                 │
│   Next.js       │                    │   FastAPI       │
│   Frontend      │   JSON + Base64    │   Backend       │
│   (Port 3000)   │   Images           │   (Port 8000)   │
│                 │                    │                 │
└─────────────────┘                    └────────┬────────┘
                                                │
                                                ▼
                                       ┌─────────────────────┐
                                       │     ML Models       │
                                       │  - SegFormer-10C    │
                                       │  - DeepLabV3+       │
                                       │  - SegFormer-C10    │
                                       │  - SegFormer-C6     │
                                       └─────────────────────┘
```

---

## Models

### Model 1: SegFormer-10C-50E (Nvidia-B-2)

**Status**: ✅ Implemented

**Description**: SegFormer is a transformer-based semantic segmentation model that combines a hierarchical Transformer encoder with a lightweight MLP decoder.

**Model Details**:
| Property | Value |
|----------|-------|
| Architecture | SegFormer (Transformer-based) |
| Pretrained Source | `PUSHPENDAR/segformer-desert` (Hugging Face) |
| Input Size | Variable (auto-resized) |
| Output Classes | 10 |
| Framework | Hugging Face Transformers |

**Key Features**:
- Hierarchical Transformer encoder
- Lightweight All-MLP decoder
- No positional encoding needed
- Efficient attention mechanism

**Performance Metrics**:
<!-- Add your model performance metrics here -->
- Overall accuracy ~87%
- 
| Metric | Train | Test |
|--------|-------|------|
| mIoU | 65.2 | 30.2 |
| Mean Accuracy | 75.0 | 51.75 |

| Inference Metric | Value |
|------------------|-------|
| Inference Time (GPU) | ~2 ms |

**Training Details**:
- Dataset: Off-road Segmentation Training Dataset
- Training notebook: `Desert_Segmentation_SegFormer_Colab (2).ipynb`

---

### Model 2: DeepLabV3+ (ResNET-100)

**Status**: ✅ Implemented

**Description**: DeepLabV3+ is a state-of-the-art semantic segmentation model that employs atrous convolution with multiple rates (ASPP) and encoder-decoder architecture.

**Model Details**:
| Property | Value |
|----------|-------|
| Architecture | DeepLabV3+ |
| Backbone | ResNet-101 |
| Input Size | 512 × 512 |
| Output Classes | 10 |
| Framework | PyTorch |

**Key Features**:
- Atrous Spatial Pyramid Pooling (ASPP)
- Encoder-decoder structure with skip connections
- Multi-scale feature extraction
- Pretrained backbone support

**Performance Metrics**:
<!-- Add your model performance metrics here -->
| Metric | Train | Test |
|--------|-------|------|
| mIoU | 0.62 | TBD |
| Mean Accuracy | 0.85 | TBD |

| Inference Metric | Value |
|------------------|-------|
| Inference Time (GPU) | ~4 ms |

**Training Details**:
- Model weights file: `best_model.pth`
- Configure path via `.env` file: `BEST_MODEL_PATH=path/to/model.pth`

---

### Model 3: SegFormer-C10 (Nvidia-B-1)

**Status**: ✅ Implemented

**Description**: SegFormer variant trained with 10 classes using the Nvidia MiT-B1 backbone for 50 epochs. A lighter version optimized for faster inference while maintaining good accuracy.

**Model Details**:
| Property | Value |
|----------|-------|
| Architecture | SegFormer (Transformer-based) |
| Backbone | Nvidia MiT-B1 |
| Input Size | Variable (auto-resized) |
| Output Classes | 10 |
| Epochs | 50 |
| Framework | Hugging Face Transformers |

**Key Features**:
- Lightweight MiT-B1 backbone
- Faster inference than B2 variant
- Good balance of speed and accuracy
- 10-class semantic segmentation

**Performance Metrics**:
| Metric | Train | Test |
|--------|-------|------|
| mIoU | 61.72% | 29.32% |
| Mean Accuracy | 72.32 | - |


| Inference Metric | Value |
|------------------|-------|
| Inference Time (GPU) | ~2.48s |


**Training Details**:
- Epochs: 50
- Dataset: Off-road Segmentation Training Dataset
- Backbone: Nvidia MiT-B1

---

### Model 4: SegFormer-C6 (Nvidia-B1)

**Status**: ✅ Implemented

**Description**: SegFormer variant trained with 6 classes using the Nvidia MiT-B1 backbone for 80 epochs. Optimized for a reduced class set with extended training for better convergence.

**Model Details**:
| Property | Value |
|----------|-------|
| Architecture | SegFormer (Transformer-based) |
| Backbone | Nvidia MiT-B1 |
| Input Size | Variable (auto-resized) |
| Output Classes | 6 |
| Epochs | 80 |
| Framework | Hugging Face Transformers |

**Key Features**:
- Reduced 6-class segmentation
- Extended 80 epoch training
- Better class-specific accuracy
- Optimized for specific terrain categories

**Performance Metrics**:
| Metric | Train | Test |
|--------|-------|------|
| mIoU | 58.30 | ~26 |
| Mean Accuracy | 70 | - |


| Inference Metric | Value |
|------------------|-------|
| Inference Time (GPU) | ~2.24 |


**Training Details**:
- Epochs: 80
- Dataset: Off-road Segmentation Training Dataset
- Backbone: Nvidia MiT-B1
- Classes: 6 (reduced from 10)

---

## Model Comparison

| Model | Classes | Epochs | Backbone | Train mIoU | Test mIoU | Train Acc | Test Acc | Inference (GPU) |
|-------|---------|--------|----------|------------|-----------|-----------|----------|------------------|
| SegFormer-10C-50E | 10 | 50 | Nvidia-B2 | 65.2 | 30.2 | 75.0 | 51.75 | ~2 ms |
| DeepLabV3+ | 10 | 50 | ResNet-101 | 62.0 | TBD | 85.0 | TBD | ~4 ms |
| SegFormer-C10 | 10 | 50 | Nvidia-B1 | 61.72 | 29.32 | 72.32 | - | ~2.48s |
| SegFormer-C6 | 6 | 80 | Nvidia-B1 | 58.30 | ~26 | 70.0 | - | ~2.24s |

---

## Installation

### Prerequisites

- **Python** 3.8 or higher
- **Node.js** 18 or higher
- **pnpm** or **npm** package manager
- **CUDA** (optional, for GPU acceleration)

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SPIT-Hackathon-2026/--INIT--.git
   cd --INIT--
   ```

2. **Setup Backend**:
   ```bash
   cd backend
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   
   pip install -r requirements.txt
   python app.py
   ```

3. **Setup Frontend** (new terminal):
   ```bash
   cd frontend
   pnpm install   # or npm install
   pnpm dev       # or npm run dev
   ```

4. **Access Application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

For detailed setup instructions, see [SETUP.md](SETUP.md).

---

## Usage

### Web Interface

1. Navigate to http://localhost:3000
2. Upload a desert/off-road terrain image (JPG/PNG)
3. Select the desired model from the dropdown
4. View segmentation results, attention maps, and statistics

### API Usage

```python
import requests
import base64

# Read and encode image
with open("terrain.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Make prediction request
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "image": image_b64,
        "model": "segformer"  # or "deeplabv3"
    }
)

result = response.json()
print(result["segmentationMask"])
```

---

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Perform segmentation on an image |
| `GET` | `/health` | Health check endpoint |
| `GET` | `/` | API information |
| `GET` | `/docs` | Swagger UI documentation |

### POST /predict

**Request Body**:
```json
{
  "image": "base64_encoded_image_string",
  "model": "segformer"  // "segformer" | "deeplabv3"
}
```

**Response**:
```json
{
  "success": true,
  "segmentationMask": "data:image/png;base64,...",
  "reasoning": {
    "explanation": "Model analysis description",
    "attentionMap": "data:image/png;base64,...",
    "featureImportance": [...],
    "classConfidence": [...]
  },
  "metadata": {
    "processingTime": 1.23,
    "modelVersion": "1.0.0",
    "imageSize": {"width": 512, "height": 512}
  }
}
```

---

## Project Structure

```
--INIT--/
├── README.md                    # This file
├── SETUP.md                     # Detailed setup guide
├── Desert_Segmentation_*.ipynb  # Training notebooks
│
├── backend/                     # FastAPI backend
│   ├── app.py                   # Main API application
│   ├── requirements.txt         # Python dependencies
│   ├── run.bat / run.sh         # Startup scripts
│   └── README.md                # Backend documentation
│
├── frontend/                    # Next.js frontend
│   ├── app/                     # Next.js app router
│   │   ├── page.tsx             # Main page
│   │   └── api/predict/         # API proxy route
│   ├── components/              # React components
│   │   ├── demo-section.tsx     # Image upload & results
│   │   ├── dashboard-section.tsx
│   │   └── ui/                  # UI components (shadcn/ui)
│   ├── lib/                     # Utilities
│   └── public/                  # Static assets
│
└── Offroad_Segmentation_Training_Dataset/  # Training data
    ├── train/
    │   ├── Color_Images/
    │   └── Segmentation/
    └── val/
        ├── Color_Images/
        └── Segmentation/
```

---

## Environment Variables

### Backend (`backend/.env`)

| Variable | Description | Default |
|----------|-------------|---------|
| `BEST_MODEL_PATH` | Path to DeepLabV3+ model weights | `best_model.pth` |

### Frontend (`frontend/.env.local`)

| Variable | Description | Default |
|----------|-------------|---------|
| `BACKEND_URL` | Backend API URL | `http://localhost:8000` |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Team

**Team Name**: --INIT--

<!-- Add team members here -->
| Name | Role | GitHub |
|------|------|--------|
| Member 1 | TBD | [@username](https://github.com/username) |
| Member 2 | TBD | [@username](https://github.com/username) |
| Member 3 | TBD | [@username](https://github.com/username) |
| Member 4 | TBD | [@username](https://github.com/username) |

---

## Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)
- [Next.js](https://nextjs.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [shadcn/ui](https://ui.shadcn.com/)
- SPIT Hackathon 2026

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ for SPIT Hackathon 2026
</p>
