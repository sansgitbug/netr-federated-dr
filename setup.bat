@echo off
:: One-time setup: create .venv and install pinned requirements (Python 3.11).
py -3.11 -m venv .venv || (echo Python 3.11 not found & pause & exit /b 1)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Setup done. Next: python scripts\prepare_data.py --help
pause
