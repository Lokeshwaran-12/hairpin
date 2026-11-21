@echo off
title Hairpin Guard AI - System Launcher
echo ======================================================================
echo           HAIRPIN GUARD AI - TRAFFIC MONITORING SYSTEM
echo ======================================================================
echo.

echo Starting Python AI Backend Service (Port 8000)...
start "Hairpin AI Backend (Port 8000)" cmd /k "python backend_service.py"

timeout /t 3 /nobreak > nul

echo Starting Node.js Web Gateway (Port 3000)...
start "Hairpin Web UI (Port 3000)" cmd /k "node server.js"

echo.
echo All services launched!
echo - Web Dashboard: http://localhost:3000
echo - FastAPI Docs:  http://localhost:8000/docs
echo ======================================================================
