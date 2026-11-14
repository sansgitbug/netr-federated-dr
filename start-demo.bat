@echo off
:: Server side for a multi-machine run: Flower server (port 8080) + dashboard (port 8501).
:: Allow inbound TCP 8080 in Windows Firewall, then run start-client.bat on each hospital laptop.
if not exist .venv\Scripts\python.exe (
    echo .venv not found. Run setup.bat first.
    pause
    exit /b 1
)
set /p CLIENTS="How many hospitals will connect? [3]: "
if "%CLIENTS%"=="" set CLIENTS=3
start "Flower Server" cmd /k ".venv\Scripts\python.exe server\flower_server.py --clients %CLIENTS%"
.venv\Scripts\python.exe -m streamlit run server\dashboard.py --server.port 8501
