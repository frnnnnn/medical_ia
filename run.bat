@echo off
REM Ejecutar la app web (Windows)
if not exist .venv (
  echo No existe el entorno .venv. Ejecuta primero windows_setup.bat
  pause
  exit /b 1
)
call .venv\Scripts\activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
