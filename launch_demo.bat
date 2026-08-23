@echo off
echo =========================================================================
echo RxVLM Clinical OCR ^& Knowledge Graph Dashboard Launcher
echo =========================================================================
echo.

netstat -ano | findstr :8501 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [INFO] Streamlit server is already active on port 8501.
) else (
    echo [INFO] Starting Streamlit server on port 8501...
    start /b streamlit run app.py --server.headless true --server.port 8501
    ping 127.0.0.1 -n 4 >nul
)

echo [INFO] Opening http://localhost:8501 in default browser...
start http://localhost:8501
echo.
echo [SUCCESS] Demo Dashboard launched!
