@echo off
:: Whole federation on this laptop: server + dashboard + 3 hospitals. Extra args pass through.
if not exist .venv\Scripts\python.exe (
    echo .venv not found. Run setup.bat first.
    pause
    exit /b 1
)
.venv\Scripts\python.exe run_demo.py %*
