@echo off
REM Post-review analyses for the LRE manuscript (2026-11-16).
REM Run from the tr-factbench-v0.1.0-preview folder. The script is resumable:
REM if it stops, run this file again and finished runs are skipped.
cd /d "%~dp0"
chcp 65001 >nul
echo === 1/2 Turkish marker strata (zeyrek, a few minutes) ===
python scripts\98b_post_review_turkish_strata.py
if errorlevel 1 (
  echo zeyrek analysis failed. Install with: python -m pip install zeyrek
  pause
  exit /b 1
)
echo.
echo === 2/2 Encoder runs: 5 full-input seeds, 3 claim-only, 3 question+claim (GPU, several hours) ===
python scripts\98_post_review_runs.py
echo.
echo Done. Results: results\statistics\POST_REVIEW_TURKISH_STRATA_v1.json
echo                results\statistics\POST_REVIEW_ENCODER_RUNS_v1.json
pause
