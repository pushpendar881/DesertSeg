#!/bin/bash
# Startup script for the backend API

echo "Starting Desert Segmentation Backend API..."
echo "Make sure you have installed dependencies: pip install -r requirements.txt"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Run the FastAPI server
python app.py

