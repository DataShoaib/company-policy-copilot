@echo off
REM One-click local dev launcher for the Company Policy Copilot.
REM Runs the API on :8001 and the Streamlit client on :8502 so it coexists
REM with any other project already using 8000/8501. Requires Docker
REM (Redis + Qdrant server mode) already running, or start them first:
REM   docker compose up -d redis qdrant
setlocal
cd /d "%~dp0"

REM ---- config ----
set "VENV=.venv-2\Scripts\python.exe"
set "API_PORT=8001"
set "UI_PORT=8500"
set "API_URL=http://localhost:8001"

REM ---- 1. start backend ----
start "hr-api" cmd /c "%VENV% -m uvicorn hr_rag.api.main:app --host 127.0.0.1 --port %API_PORT%" 1>uvicorn_out.log 2>uvicorn_err.log

echo Waiting for API on :%API_PORT% ...
:await_api
timeout /t 2 >nul
curl -s -o nul -w "%%{http_code}" http://127.0.0.1:%API_PORT%/health 2>nul | findstr /r "200" >nul
if errorlevel 1 goto await_api

echo API is up. Starting UI on :%UI_PORT% ...
set "API_URL=%API_URL%"
start "UI" cmd /k "%VENV% -m streamlit run frontend/app.py --server.port %UI_PORT% --server.headless true"
echo.
echo Policy Copilot:  http://localhost:%UI_PORT%
echo API:             %API_URL%/docs
endlocal