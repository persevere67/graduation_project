@echo off
setlocal

echo [1/3] Checking processed data...
python tests\check_data.py
if errorlevel 1 goto :fail

echo.
echo [2/3] Checking dataset pipeline...
python tests\test_dataset.py
if errorlevel 1 goto :fail

echo.
echo [3/3] Running offline evaluation...
python evaluation.py
if errorlevel 1 goto :fail

echo.
echo Quickstart finished successfully.
goto :end

:fail
echo.
echo Quickstart failed. Please check Python dependencies and required files.
exit /b 1

:end
endlocal
