@echo off
title Amazon Review Predictor

cd /d "D:\Amazon"

echo Starting Amazon Review Predictor...
echo.

"D:\Qwen 2.5 7B\env\python.exe" app.py

echo.
echo Server stopped. Press any key to exit...
pause >nul
