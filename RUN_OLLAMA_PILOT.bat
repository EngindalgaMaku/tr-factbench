@echo off
REM Pilot: Turkish-Gemma-9b-T1 (local, Ollama) on 24 Gold-480 examples, zero-shot (about 5-10 minutes).
REM The model is downloaded on first use (about 5.8 GB).
cd /d "%~dp0"
chcp 65001 >nul
ollama pull hf.co/ytu-ce-cosmos/Turkish-Gemma-9b-T1-GGUF:Q4_K_M
python scripts\98f_ollama_pilot.py hf.co/ytu-ce-cosmos/Turkish-Gemma-9b-T1-GGUF:Q4_K_M turkish-gemma-9b-t1-q4km zero_shot
pause
