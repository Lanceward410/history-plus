@echo off
title History Plus Backend Launcher
echo ================================================
echo        History Plus Backend Launcher
echo ================================================
echo.

:: Check if Python is installed
echo [1/4] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org/downloads
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)
echo ✓ Python found

:: Navigate to backend directory
echo.
echo [2/4] Navigating to backend directory...
cd /d "%~dp0backend"
if %errorlevel% neq 0 (
    echo ERROR: Could not find backend directory
    echo Make sure this script is in the History Plus extension folder
    pause
    exit /b 1
)
echo ✓ Backend directory found

:: Install dependencies
echo.
echo [3/4] Installing dependencies...
if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found
    pause
    exit /b 1
)

pip install -r requirements.txt >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Some dependencies may have failed to install
    echo Attempting to continue anyway...
) else (
    echo ✓ Dependencies installed
)

:: Start the Flask server
echo.
echo [4/4] Starting History Plus backend server...
echo.
echo ================================================
echo  Backend starting on http://localhost:5000
echo  Press Ctrl+C to stop the server
echo ================================================
echo.

python main_app.py

echo.
echo Backend stopped.
pause 