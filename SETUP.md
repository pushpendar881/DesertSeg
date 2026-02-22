# Desert Segmentation Application - Setup Guide

This guide will help you set up and run both the frontend and backend components of the Desert Segmentation application.

## Architecture Overview

- **Frontend**: Next.js application (TypeScript/React) - serves the UI and acts as a proxy to the backend
- **Backend**: FastAPI Python application - handles image processing and model inference using SegFormer

## Prerequisites

- **Node.js** 18+ and npm/pnpm
- **Python** 3.8+
- **pip** package manager

## Setup Instructions

### 1. Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
```

3. Activate the virtual environment:
   - **Windows**: `venv\Scripts\activate`
   - **Linux/Mac**: `source venv/bin/activate`

4. Install Python dependencies:
```bash
pip install -r requirements.txt
```

5. Start the backend server:
```bash
python app.py
```

The backend will start on `http://localhost:8000`. The first run will download the SegFormer model (~150MB), which may take a few minutes.

**Verify backend is running:**
- Visit `http://localhost:8000/health` in your browser
- Or visit `http://localhost:8000/docs` for interactive API documentation

### 2. Frontend Setup

1. Open a new terminal and navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
# or
pnpm install
```

3. Start the development server:
```bash
npm run dev
# or
pnpm dev
```

The frontend will start on `http://localhost:3000`

### 3. Using the Application

1. Open your browser and navigate to `http://localhost:3000`
2. Upload a JPG image of a desert scene
3. The application will automatically:
   - Send the image to the backend API
   - Process it through the SegFormer model
   - Display the segmentation mask, attention map, and statistics

## Environment Variables (Optional)

You can configure the backend URL by creating a `.env.local` file in the `frontend` directory:

```env
BACKEND_URL=http://localhost:8000
```

If not set, it defaults to `http://localhost:8000`.

## Troubleshooting

### Backend Issues

- **Model download fails**: Check your internet connection. The model is downloaded from Hugging Face on first run.
- **CUDA/GPU errors**: The code will automatically fall back to CPU if CUDA is not available. GPU is optional but recommended for faster inference.
- **Port 8000 already in use**: Change the port in `backend/app.py` or use: `uvicorn app:app --port 8001`

### Frontend Issues

- **Cannot connect to backend**: Make sure the backend is running on port 8000, or update the `BACKEND_URL` environment variable.
- **CORS errors**: The backend is configured to allow all origins. In production, update the CORS settings in `backend/app.py`.

## Development Notes

- The backend uses lazy loading - the model is loaded on first request or at startup
- The frontend includes a fallback to mock data if the backend is unavailable (development mode only)
- Both servers support hot-reload during development

## Production Deployment

For production:
1. Set `NODE_ENV=production` for the frontend
2. Update CORS origins in `backend/app.py` to your frontend domain
3. Use a production WSGI server like Gunicorn for the backend
4. Configure proper environment variables

