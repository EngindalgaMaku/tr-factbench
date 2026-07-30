@echo off
REM G1_C6 dropout nihai model secimi — DROP0 ve DROP10, seed 45-46 (4 kos, ~60 dak)
REM DROP5 elendi, bu kosular 5-seed kararlılık kaniti icin yapiliyor.
cd /d "%~dp0\.."

python scripts/10_run_experiment.py --run_id G1_C6_M04_ELECTRA_LORA16_DROP0_S45 --model_name dbmdz/electra-base-turkish-cased-discriminator --method lora --lora_r 16 --lora_alpha 32 --lora_target_modules all-linear --lora_dropout 0.00 --learning_rate 2e-4 --seed 45
if %errorlevel% neq 0 exit /b 1

python scripts/10_run_experiment.py --run_id G1_C6_M04_ELECTRA_LORA16_DROP0_S46 --model_name dbmdz/electra-base-turkish-cased-discriminator --method lora --lora_r 16 --lora_alpha 32 --lora_target_modules all-linear --lora_dropout 0.00 --learning_rate 2e-4 --seed 46
if %errorlevel% neq 0 exit /b 1

python scripts/10_run_experiment.py --run_id G1_C6_M04_ELECTRA_LORA16_DROP10_S45 --model_name dbmdz/electra-base-turkish-cased-discriminator --method lora --lora_r 16 --lora_alpha 32 --lora_target_modules all-linear --lora_dropout 0.10 --learning_rate 2e-4 --seed 45
if %errorlevel% neq 0 exit /b 1

python scripts/10_run_experiment.py --run_id G1_C6_M04_ELECTRA_LORA16_DROP10_S46 --model_name dbmdz/electra-base-turkish-cased-discriminator --method lora --lora_r 16 --lora_alpha 32 --lora_target_modules all-linear --lora_dropout 0.10 --learning_rate 2e-4 --seed 46
if %errorlevel% neq 0 exit /b 1

echo G1 nihai model seedleri tamamlandi (DROP0 ve DROP10, S45-46)