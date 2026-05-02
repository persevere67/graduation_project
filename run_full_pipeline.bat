@echo off
setlocal

echo [1/6] Extracting news embeddings...
python preprocess\preprocess_news.py
if errorlevel 1 goto :fail

echo.
echo [2/6] Building federated training data...
python preprocess\preprocess_behavior.py
if errorlevel 1 goto :fail

echo.
echo [3/6] Building dev evaluation data...
python preprocess\preprocess_dev.py
if errorlevel 1 goto :fail

echo.
echo [4/6] Training centralized baseline...
python baseline_centralized.py
if errorlevel 1 goto :fail

echo.
echo [5/6] Running offline evaluation...
python evaluation.py
if errorlevel 1 goto :fail

echo.
echo [6/6] Starting federated training...
python federated_main.py
if errorlevel 1 goto :fail

echo.
echo Full pipeline finished successfully.
goto :end

:fail
echo.
echo Full pipeline failed. Please check the previous step output.
exit /b 1

:end
endlocal
