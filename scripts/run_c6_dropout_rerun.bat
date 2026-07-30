@echo off
REM C6 Dropout Ablasyon Yeniden Calisturma — 3 kosu (~45 dak)
REM ELECTRA-TR r=16 all-linear, seed=42
REM dropout: 0.00 / 0.05 / 0.10
setlocal enabledelayedexpansion
cd /d "%~dp0\.."

echo ============================================================
echo C6 DROPOUT ABLASYON RERUN (mevcut src/train.py)
echo ============================================================

echo [1/3] dropout=0.00
python scripts/10_run_experiment.py --run_id G1_C6_M04_ELECTRA_LORA16_DROP0_S42 --model_name dbmdz/electra-base-turkish-cased-discriminator --method lora --lora_r 16 --lora_alpha 32 --lora_target_modules query,key,value,dense,out_proj --lora_dropout 0.00 --learning_rate 2e-4 --seed 42
if %errorlevel% neq 0 ( echo HATA: DROP0 basarisiz & exit /b 1 )

echo [2/3] dropout=0.05
python scripts/10_run_experiment.py --run_id G1_C6_M04_ELECTRA_LORA16_DROP5_S42 --model_name dbmdz/electra-base-turkish-cased-discriminator --method lora --lora_r 16 --lora_alpha 32 --lora_target_modules query,key,value,dense,out_proj --lora_dropout 0.05 --learning_rate 2e-4 --seed 42
if %errorlevel% neq 0 ( echo HATA: DROP5 basarisiz & exit /b 1 )

echo [3/3] dropout=0.10
python scripts/10_run_experiment.py --run_id G1_C6_M04_ELECTRA_LORA16_DROP10_S42 --model_name dbmdz/electra-base-turkish-cased-discriminator --method lora --lora_r 16 --lora_alpha 32 --lora_target_modules query,key,value,dense,out_proj --lora_dropout 0.10 --learning_rate 2e-4 --seed 42
if %errorlevel% neq 0 ( echo HATA: DROP10 basarisiz & exit /b 1 )

echo ============================================================
echo C6 DROPOUT ABLASYON TAMAMLANDI
echo ============================================================