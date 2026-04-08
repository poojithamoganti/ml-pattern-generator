@echo off
REM Must run with cwd=backend so "app" package resolves (or use start-backend.bat from repo root)
cd /d "%~dp0"
call .venv\Scripts\activate.bat
cd backend
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
