@echo off
REM Diagnostic: seed 42 on the v1.1.0 (unmasked) training split, same configuration (about 13 minutes).
cd /d "%~dp0"
chcp 65001 >nul
python scripts\98d_recheck_v11_s42.py
pause
