@echo off
REM ModernBERT-TR-Base-1k F-serisi eşlenmiş 3-seed koşusu
REM F1_M12_MODERNBERT_LORA16_S42/43/44
REM Konfigürasyon: LoRA r=16, alpha=32, all-linear, dropout=0.05, lr=2e-4
setlocal enabledelayedexpansion
cd /d "%~dp0\.."

echo ============================================================
echo MODERNBERT-TR F1 3-SEED ALIGNED RUN
echo ============================================================

echo [1/3] ModernBERT-TR S42
python scripts/10_run_experiment.py --run_id F1_M12_MODERNBERT_LORA16_S42 --model_name ytu-ce-cosmos/modernbert-tr-base-1k --method lora --lora_r 16 --lora_alpha 32 --lora_target_modules all-linear --lora_dropout 0.10 --learning_rate 2e-4 --seed 42
if %errorlevel% neq 0 ( echo HATA: S42 basarisiz & exit /b 1 )

echo [2/3] ModernBERT-TR S43
python scripts/10_run_experiment.py --run_id F1_M12_MODERNBERT_LORA16_S43 --model_name ytu-ce-cosmos/modernbert-tr-base-1k --method lora --lora_r 16 --lora_alpha 32 --lora_target_modules all-linear --lora_dropout 0.10 --learning_rate 2e-4 --seed 43
if %errorlevel% neq 0 ( echo HATA: S43 basarisiz & exit /b 1 )

echo [3/3] ModernBERT-TR S44
python scripts/10_run_experiment.py --run_id F1_M12_MODERNBERT_LORA16_S44 --model_name ytu-ce-cosmos/modernbert-tr-base-1k --method lora --lora_r 16 --lora_alpha 32 --lora_target_modules all-linear --lora_dropout 0.10 --learning_rate 2e-4 --seed 44
if %errorlevel% neq 0 ( echo HATA: S44 basarisiz & exit /b 1 )

echo ============================================================
echo MODERNBERT-TR 3 KOSU TAMAMLANDI
echo ============================================================