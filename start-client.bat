@echo off
:: Hospital side for a multi-machine run. Needs data_zips\hospital_<ID>_data.zip on this laptop.
if not exist .venv\Scripts\python.exe (
    echo .venv not found. Run setup.bat first.
    pause
    exit /b 1
)
set /p HOSPITAL_ID="Hospital ID (1, 2, 3...): "
set /p SERVER_IP="Server IP address (e.g. 192.168.1.10): "
.venv\Scripts\python.exe client.py --hospital-id %HOSPITAL_ID% --server %SERVER_IP%:8080
pause
