@echo off
title Pill Detection AI Server

echo ============================================================
echo   Pill Detection AI — RF-DETR nano ONNX
echo ============================================================
echo.

echo [1/2] Installing / verifying dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERROR: pip install failed. Make sure Python is on PATH.
    pause
    exit /b 1
)

echo [2/2] Starting server on http://localhost:8000
echo.
echo  Open your browser at:  http://localhost:8000
echo  Press Ctrl+C to stop the server.
echo.

C:\Users\asus\AppData\Local\Programs\Python\Python310\python.exe server.py

pause
