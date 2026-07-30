@echo off
REM Reproducibility sentinel - reruns the locked H1 final-model configuration.
REM Acceptance rule: absolute Macro-F1 difference <= 0.010 on both development sets.
REM Reference H1_DROP10_S45: DEV960-GEN=0.9014, DEV600-STR=0.8675.
REM This is a fresh training run, not inference on saved weights.

python scripts/10_run_experiment.py ^
  --run_id H1_SENTINEL_DROP10_S45_RERUN ^
  --model_name dbmdz/electra-base-turkish-cased-discriminator ^
  --method lora ^
  --lora_r 16 ^
  --lora_alpha 32 ^
  --lora_dropout 0.10 ^
  --lora_target_modules all-linear ^
  --learning_rate 2e-4 ^
  --seed 45

if %ERRORLEVEL% neq 0 (
  echo [HATA] Sentinel kosusu basarisiz oldu.
  exit /b 1
)

python scripts/13_collect_results.py
python scripts/35_generate_lock_yaml.py

echo.
echo [OK] Sentinel kosu tamamlandi: H1_SENTINEL_DROP10_S45_RERUN
echo      Referans DEV960-GEN: 0.9014  DEV600-STR: 0.8675
echo      Kabul esigi: her iki sette mutlak Macro-F1 farki ^<= 0.010
echo      Sonucu final_config.lock.yaml dosyasindan kontrol edin.
