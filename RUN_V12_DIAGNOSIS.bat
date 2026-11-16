@echo off
REM Diagnosis of the v1.2.0 collapses: two bisection runs and one more v1.2.0 seed (about 40 minutes).
cd /d "%~dp0"
chcp 65001 >nul
python scripts\98e_v12_diagnosis.py
echo.
echo Results: results\statistics\V12_DIAGNOSIS_v1.json
pause
