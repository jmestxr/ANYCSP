@echo off
setlocal

set DATA_DIR_3CNF="data\test\MAXSAT\3CNF"
set DATA_DIR_4CNF="data\test\MAXSAT\4CNF"
set DATA_DIR_5CNF="data\test\MAXSAT\5CNF"
set MODEL_DIR="models\MAXSAT"
set TIMEOUT=1200
set NUM_BOOST=1

for %%f in (%DATA_DIR_3CNF%\*.cnf) do (
    python evaluate.py --data_path "%%f" --model_dir %MODEL_DIR% --num_boost %NUM_BOOST% --timeout %TIMEOUT% --verbose
)

for %%f in (%DATA_DIR_4CNF%\*.cnf) do (
    python evaluate.py --data_path "%%f" --model_dir %MODEL_DIR% --num_boost %NUM_BOOST% --timeout %TIMEOUT% --verbose
)

for %%f in (%DATA_DIR_5CNF%\*.cnf) do (
    python evaluate.py --data_path "%%f" --model_dir %MODEL_DIR% --num_boost %NUM_BOOST% --timeout %TIMEOUT% --verbose
)

endlocal