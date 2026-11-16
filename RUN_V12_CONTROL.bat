@echo off
REM v1.2.0 control runs: five seeds on the entity-masked training split (about 1 hour on the RTX 4060).
REM Resumable: run again if it stops.
cd /d "%~dp0"
chcp 65001 >nul
python scripts\98c_v12_control_runs.py
echo.
echo Done. Results: results\statistics\POST_REVIEW_V12_CONTROL_v1.json
pause
