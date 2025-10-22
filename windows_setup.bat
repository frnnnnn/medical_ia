@echo off
REM Crear entorno virtual e instalar dependencias (Windows)
py -m venv .venv
call .venv\Scripts\activate
py -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Entorno listo. Para activarlo en futuras sesiones: call .venv\Scripts\activate
echo.
pause
