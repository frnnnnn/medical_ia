@echo off
REM === Crea un entorno con Python 3.12 explícitamente ===
REM Requiere tener instalado Python 3.12 (https://www.python.org/downloads/release/python-312x/)
where py 1>nul 2>nul || (echo No se encontró 'py'. Instala Python desde python.org y reintenta. & pause & exit /b 1)
py -3.12 -V || (echo No se encontró Python 3.12. Instálalo y reintenta. & pause & exit /b 1)

py -3.12 -m venv .venv
call .venv\Scripts\activate
py -3.12 -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Entorno con Python 3.12 listo. Para activarlo: call .venv\Scripts\activate
echo.
pause
