@echo off
TITLE RoadSentinel Demo Launcher
COLOR 0A

echo ===================================================
echo        RoadSentinel Traffic Auditor Launcher
echo ===================================================
echo.

:: 1. Launch FastAPI Backend
echo [*] Starting FastAPI Backend on http://127.0.0.1:8765...
start "RoadSentinel Backend" cmd /k "cd /d C:\Users\Admin\Desktop\BTECH\RoadSentinel && C:\Projects\RoadSentinel\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8765"

:: 2. Wait 3 seconds for backend initialization
timeout /t 3 /nobreak >nul

:: 3. Launch React / Vite Frontend
echo [*] Starting Frontend Server on http://127.0.0.1:5173...
start "RoadSentinel Frontend" cmd /k "cd /d C:\Users\Admin\Desktop\BTECH\RoadSentinel\frontend && set VITE_API_BASE_URL=http://127.0.0.1:8765 && npm run dev"

:: 4. Open Browser
echo [*] Launching dashboard in browser...
timeout /t 2 /nobreak >nul
start http://127.0.0.1:5173

echo.
echo ===================================================
echo Both services are active. Close the open windows to stop.
echo ===================================================
pause