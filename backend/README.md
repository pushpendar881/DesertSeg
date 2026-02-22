# Desert Segmentation Backend API

FastAPI backend for desert terrain semantic segmentation using SegFormer model.

## Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Navigate to the backend directory:
```bash
cd backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file for configuration (optional):
```bash
# Create .env file with model path
echo "BEST_MODEL_PATH=best_model.pth" > .env
```

Or create `backend/.env` manually with:
```
# DeepLabV3+ model weights - path relative to backend/ or absolute
BEST_MODEL_PATH=best_model.pth
```

**Note:** The first time you run the application, it will download the SegFormer model from Hugging Face (~150MB). This may take a few minutes depending on your internet connection.

### Running the Server

**Option 1: Using Python directly**
```bash
python app.py
```

**Option 2: Using uvicorn directly (recommended for development)**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Option 3: Using the startup script**
- Windows: `run.bat`
- Linux/Mac: `bash run.sh` (make sure it's executable: `chmod +x run.sh`)

The API will be available at `http://localhost:8000`

### Testing the API

You can test the health endpoint:
```bash
curl http://localhost:8000/health
```

Or visit `http://localhost:8000/docs` in your browser to see the interactive API documentation.

## API Endpoints

### POST /predict
Accepts a base64-encoded image and returns segmentation results.

**Request Body:**
```json
{
  "image": "base64_encoded_image_string",
  "model": "segformer"
}
```

**Response:**
```json
{
  "success": true,
  "segmentationMask": "data:image/png;base64,...",
  "reasoning": {
    "explanation": "...",
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

### GET /health
Health check endpoint.

### GET /
Root endpoint with API information.

