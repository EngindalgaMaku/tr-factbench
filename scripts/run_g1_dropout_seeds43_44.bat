@echo off
REM G1_C6 dropout ablasyon — seed 43 ve 44 (6 kos, ~90 dak)
cd /d "%~dp0\.."

python scripts/10_run_experiment.py --run_id G1_C6_M04_ELECTRA_LORA16_DROP0_S43 --model_name dbmdz/electra-base-turkish-cased-discriminator --method lora --lora_r 16 --lora_alpha 32 --lora_target_modules all-linear --lora_dropout 0.00 --learning_rate 2e-4 --seed 43
if %errorlevel% neq 0 exit /b 1

python scripts/10_run_experiment.py --run_id G1_C6_M04_ELECTRA_LORA16_DROP0_S44 --model_name dbmdz/electra-base-turkish-cased-discriminator --method lora --lora_r 16 --lora_alpha 32 --lora_target_modules all-linear --lora_dropout 0.00 --learning_rate 2e-4 --seed 44
if %errorlevel% neq 0 exit /b 1

python scripts/10_run_experiment.py --run_id G1_C6_M04_ELECTRA_LORA16_DROP5_S43 --model_name dbmdz/electra-base-turkish-cased-discriminator --method lora --lora_r 16 --lora_alpha 32 --lora_target_modules all-linear --lora_dropout 0.05 --learning_rate 2e-4 --seed 43
if %errorlevel% neq 0 exit /b 1

python scripts/10_run_experiment.py --run_id G1_C6_M04_ELECTRA_LORA16_DROP5_S44 --model_name dbmdz/electra-base-turkish-cased-discriminator --method lora --lora_r 16 --lora_alpha 32 --lora_target_modules all-linear --lora_dropout 0.05 --learning_rate 2e-4 --seed 44
if %errorlevel% neq 0 exit /b 1

python scripts/10_run_experiment.py --run_id G1_C6_M04_ELECTRA_LORA16_DROP10_S43 --model_name dbmdz/electra-base-turkish-cased-discriminator --method lora --lora_r 16 --lora_alpha 32 --lora_target_modules all-linear --lora_dropout 0.10 --learning_rate 2e-4 --seed 43
if %errorlevel% neq 0 exit /b 1

python scripts/10_run_experiment.py --run_id G1_C6_M04_ELECTRA_LORA16_DROP10_S44 --model_name dbmdz/electra-base-turkish-cased-discriminator --method lora --lora_r 16 --lora_alpha 32 --lora_target_modules all-linear --lora_dropout 0.10 --learning_rate 2e-4 --seed 44
if %errorlevel% neq 0 exit /b 1

echo G1 dropout tum seedler tamamlandi