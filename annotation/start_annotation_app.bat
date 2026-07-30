@echo off
cd /d "%~dp0\.."
python -m streamlit run annotation\review_app.py
