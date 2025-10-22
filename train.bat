@echo off
if not exist .venv (
  echo No existe el entorno .venv. Ejecuta primero windows_setup.bat o windows_setup_py312.bat
  pause
  exit /b 1
)
call .venv\Scripts\activate
py -m models.train_insurance
py -m models.train_diabetes
py -m models.rf_feature_importance
echo.
echo Entrenamiento completado. Revisa la carpeta reports\
echo.
pause
