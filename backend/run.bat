@echo off
REM Startup script for Windows

echo Starting Desert Segmentation Backend API...
echo Make sure you have installed dependencies: pip install -r requirements.txt
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

REM Run the FastAPI server
python app.py

