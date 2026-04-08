@echo off
REM Run API from project root (no need to cd backend)
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8001
