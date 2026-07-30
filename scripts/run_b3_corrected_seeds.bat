@echo off
REM ============================================================
REM B3 — Düzeltilmiş Seed Eğitimleri
REM
REM Sorun: B2_M03_CONVBERT S43-S44 yanlış checkpoint (cased yerine mc4-cased olmalı)
REM        B2_M02_BERT128K S43-S44 yanlış checkpoint (uncased yerine 128k-cased olmalı)
REM
REM Bu script doğru model_name ile 4 run'ı sırayla çalıştırır.
REM GPU gereklidir. Her run ~25-35 dakika sürer.
REM ============================================================

cd /d "%~dp0.."

echo ============================================================
echo B3 Düzeltilmiş Seed Eğitimleri — Başlıyor
echo ============================================================

REM ── ConvBERT mc4-cased S43 ────────────────────────────────
echo.
echo [1/4] B3_M03_CONVBERT_ALLLINEAR_LORA16_S43
python scripts/10_run_experiment.py ^
  --run_id B3_M03_CONVBERT_ALLLINEAR_LORA16_S43 ^
  --model_name dbmdz/convbert-base-turkish-mc4-cased ^
  --method lora ^
  --lora_r 16 ^
  --lora_alpha 32 ^
  --learning_rate 0.0002 ^
  --seed 43
if %errorlevel% neq 0 (
    echo [ERROR] B3_M03_CONVBERT_S43 basarisiz!
    exit /b 1
)

REM ── ConvBERT mc4-cased S44 ────────────────────────────────
echo.
echo [2/4] B3_M03_CONVBERT_ALLLINEAR_LORA16_S44
python scripts/10_run_experiment.py ^
  --run_id B3_M03_CONVBERT_ALLLINEAR_LORA16_S44 ^
  --model_name dbmdz/convbert-base-turkish-mc4-cased ^
  --method lora ^
  --lora_r 16 ^
  --lora_alpha 32 ^
  --learning_rate 0.0002 ^
  --seed 44
if %errorlevel% neq 0 (
    echo [ERROR] B3_M03_CONVBERT_S44 basarisiz!
    exit /b 1
)

REM ── BERT-128k-cased S43 ──────────────────────────────────
echo.
echo [3/4] B3_M02_BERT128K_ALLLINEAR_LORA16_S43
python scripts/10_run_experiment.py ^
  --run_id B3_M02_BERT128K_ALLLINEAR_LORA16_S43 ^
  --model_name dbmdz/bert-base-turkish-128k-cased ^
  --method lora ^
  --lora_r 16 ^
  --lora_alpha 32 ^
  --learning_rate 0.0002 ^
  --seed 43
if %errorlevel% neq 0 (
    echo [ERROR] B3_M02_BERT128K_S43 basarisiz!
    exit /b 1
)

REM ── BERT-128k-cased S44 ──────────────────────────────────
echo.
echo [4/4] B3_M02_BERT128K_ALLLINEAR_LORA16_S44
python scripts/10_run_experiment.py ^
  --run_id B3_M02_BERT128K_ALLLINEAR_LORA16_S44 ^
  --model_name dbmdz/bert-base-turkish-128k-cased ^
  --method lora ^
  --lora_r 16 ^
  --lora_alpha 32 ^
  --learning_rate 0.0002 ^
  --seed 44
if %errorlevel% neq 0 (
    echo [ERROR] B3_M02_BERT128K_S44 basarisiz!
    exit /b 1
)

echo.
echo ============================================================
echo TUM B3 RUNLARI TAMAMLANDI
echo Sonraki adim: python scripts/37_b3_postprocess.py
echo ============================================================