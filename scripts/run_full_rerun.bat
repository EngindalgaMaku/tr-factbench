@echo off
REM Tablo A icin tam yeniden calisturma — 18 kosu, ~3.5 saat
REM Her kosu tamamlaninca bir sonrakine gec. Hata durumunda dur.
setlocal enabledelayedexpansion
cd /d "%~dp0\.."

echo ============================================================
echo FULL TABLE-A RERUN STARTING
echo ============================================================

REM === ELECTRA-TR B1_NEW seeds 42/43/44 ===
echo [1/18] ELECTRA S42
python scripts/10_run_experiment.py --run_id F1_M04_ELECTRA_LORA16_S42 --model_name dbmdz/electra-base-turkish-cased-discriminator --method lora --lora_r 16 --lora_alpha 32 --lora_target_modules query,key,value,dense,out_proj --lora_dropout 0.05 --learning_rate 2e-4 --seed 42
if %errorlevel% neq 0 ( echo HATA: S42 basarisiz & exit /b 1 )

echo [2/18] ELECTRA S43
python scripts/10_run_experiment.py --run_id F1_M04_ELECTRA_LORA16_S43 --model_name dbmdz/electra-base-turkish-cased-discriminator --method lora --lora_r 16 --lora_alpha 32 --lora_target_modules query,key,value,dense,out_proj --lora_dropout 0.05 --learning_rate 2e-4 --seed 43
if %errorlevel% neq 0 ( echo HATA: S43 basarisiz & exit /b 1 )

echo [3/18] ELECTRA S44
python scripts/10_run_experiment.py --run_id F1_M04_ELECTRA_LORA16_S44 --model_name dbmdz/electra-base-turkish-cased-discriminator --method lora --lora_r 16 --lora_alpha 32 --lora_target_modules query,key,value,dense,out_proj --lora_dropout 0.05 --learning_rate 2e-4 --seed 44
if %errorlevel% neq 0 ( echo HATA: S44 basarisiz & exit /b 1 )

REM === ConvBERT-mc4-cased seeds 42/43/44 ===
echo [4/18] ConvBERT S42
python scripts/10_run_experiment.py --run_id F1_M03_CONVBERT_LORA16_S42 --model_name dbmdz/convbert-base-turkish-mc4-cased --method lora --lora_r 16 --lora_alpha 32 --lora_target_modules query,key,value,dense,out_proj --lora_dropout 0.05 --learning_rate 2e-4 --seed 42
if %errorlevel% neq 0 ( echo HATA: ConvBERT S42 basarisiz & exit /b 1 )

echo [5/18] ConvBERT S43
python scripts/10_run_experiment.py --run_id F1_M03_CONVBERT_LORA16_S43 --model_name dbmdz/convbert-base-turkish-mc4-cased --method lora --lora_r 16 --lora_alpha 32 --lora_target_modules query,key,value,dense,out_proj --lora_dropout 0.05 --learning_rate 2e-4 --seed 43
if %errorlevel% neq 0 ( echo HATA: ConvBERT S43 basarisiz & exit /b 1 )

echo [6/18] ConvBERT S44
python scripts/10_run_experiment.py --run_id F1_M03_CONVBERT_LORA16_S44 --model_name dbmdz/convbert-base-turkish-mc4-cased --method lora --lora_r 16 --lora_alpha 32 --lora_target_modules query,key,value,dense,out_proj --lora_dropout 0.05 --learning_rate 2e-4 --seed 44
if %errorlevel% neq 0 ( echo HATA: ConvBERT S44 basarisiz & exit /b 1 )

REM === BERT-128k-cased seeds 42/43/44 ===
echo [7/18] BERT-128k S42
python scripts/10_run_experiment.py --run_id F1_M02_BERT128K_LORA16_S42 --model_name dbmdz/bert-base-turkish-128k-cased --method lora --lora_r 16 --lora_alpha 32 --lora_target_modules query,key,value,dense --lora_dropout 0.05 --learning_rate 2e-4 --seed 42
if %errorlevel% neq 0 ( echo HATA: BERT128K S42 basarisiz & exit /b 1 )

echo [8/18] BERT-128k S43
python scripts/10_run_experiment.py --run_id F1_M02_BERT128K_LORA16_S43 --model_name dbmdz/bert-base-turkish-128k-cased --method lora --lora_r 16 --lora_alpha 32 --lora_target_modules query,key,value,dense --lora_dropout 0.05 --learning_rate 2e-4 --seed 43
if %errorlevel% neq 0 ( echo HATA: BERT128K S43 basarisiz & exit /b 1 )

echo [9/18] BERT-128k S44
python scripts/10_run_experiment.py --run_id F1_M02_BERT128K_LORA16_S44 --model_name dbmdz/bert-base-turkish-128k-cased --method lora --lora_r 16 --lora_alpha 32 --lora_target_modules query,key,value,dense --lora_dropout 0.05 --learning_rate 2e-4 --seed 44
if %errorlevel% neq 0 ( echo HATA: BERT128K S44 basarisiz & exit /b 1 )

REM === BERT-32k LoRA r=32 seeds 42/43/44 ===
echo [10/18] BERT-32k LoRA-r32 S42
python scripts/10_run_experiment.py --run_id F1_M01_BERT32K_LORA32_S42 --model_name dbmdz/bert-base-turkish-cased --method lora --lora_r 32 --lora_alpha 64 --lora_target_modules query,key,value,dense --lora_dropout 0.05 --learning_rate 2e-4 --seed 42
if %errorlevel% neq 0 ( echo HATA: BERT32K-r32 S42 basarisiz & exit /b 1 )

echo [11/18] BERT-32k LoRA-r32 S43
python scripts/10_run_experiment.py --run_id F1_M01_BERT32K_LORA32_S43 --model_name dbmdz/bert-base-turkish-cased --method lora --lora_r 32 --lora_alpha 64 --lora_target_modules query,key,value,dense --lora_dropout 0.05 --learning_rate 2e-4 --seed 43
if %errorlevel% neq 0 ( echo HATA: BERT32K-r32 S43 basarisiz & exit /b 1 )

echo [12/18] BERT-32k LoRA-r32 S44
python scripts/10_run_experiment.py --run_id F1_M01_BERT32K_LORA32_S44 --model_name dbmdz/bert-base-turkish-cased --method lora --lora_r 32 --lora_alpha 64 --lora_target_modules query,key,value,dense --lora_dropout 0.05 --learning_rate 2e-4 --seed 44
if %errorlevel% neq 0 ( echo HATA: BERT32K-r32 S44 basarisiz & exit /b 1 )

REM === BERT-32k Full Fine-tune seeds 43/44/45 (S42 = D2_M01_BERT32K_FULL_FT mevcut) ===
echo [13/18] BERT-32k Full-FT S43
python scripts/10_run_experiment.py --run_id F1_M01_BERT32K_FULLFT_S43 --model_name dbmdz/bert-base-turkish-cased --method full_ft --learning_rate 2e-5 --seed 43
if %errorlevel% neq 0 ( echo HATA: BERT32K-FT S43 basarisiz & exit /b 1 )

echo [14/18] BERT-32k Full-FT S44
python scripts/10_run_experiment.py --run_id F1_M01_BERT32K_FULLFT_S44 --model_name dbmdz/bert-base-turkish-cased --method full_ft --learning_rate 2e-5 --seed 44
if %errorlevel% neq 0 ( echo HATA: BERT32K-FT S44 basarisiz & exit /b 1 )

echo [15/18] BERT-32k Full-FT S45
python scripts/10_run_experiment.py --run_id F1_M01_BERT32K_FULLFT_S45 --model_name dbmdz/bert-base-turkish-cased --method full_ft --learning_rate 2e-5 --seed 45
if %errorlevel% neq 0 ( echo HATA: BERT32K-FT S45 basarisiz & exit /b 1 )

REM === mBERT seeds 42/43/44 ===
echo [16/18] mBERT S42
python scripts/10_run_experiment.py --run_id F1_M07_MBERT_LORA16_S42 --model_name google-bert/bert-base-multilingual-cased --method lora --lora_r 16 --lora_alpha 32 --lora_target_modules query,key,value,dense --lora_dropout 0.05 --learning_rate 2e-4 --seed 42
if %errorlevel% neq 0 ( echo HATA: mBERT S42 basarisiz & exit /b 1 )

echo [17/18] mBERT S43
python scripts/10_run_experiment.py --run_id F1_M07_MBERT_LORA16_S43 --model_name google-bert/bert-base-multilingual-cased --method lora --lora_r 16 --lora_alpha 32 --lora_target_modules query,key,value,dense --lora_dropout 0.05 --learning_rate 2e-4 --seed 43
if %errorlevel% neq 0 ( echo HATA: mBERT S43 basarisiz & exit /b 1 )

echo [18/18] mBERT S44
python scripts/10_run_experiment.py --run_id F1_M07_MBERT_LORA16_S44 --model_name google-bert/bert-base-multilingual-cased --method lora --lora_r 16 --lora_alpha 32 --lora_target_modules query,key,value,dense --lora_dropout 0.05 --learning_rate 2e-4 --seed 44
if %errorlevel% neq 0 ( echo HATA: mBERT S44 basarisiz & exit /b 1 )

echo ============================================================
echo TUM 18 KOSU TAMAMLANDI
echo ============================================================