@echo off
REM Starts the FIXED dashboard (Claude/fixed/app_fixed.py) with the project's .venv.
REM Run f09_forecast_fixed.py first. Nothing outside the Claude folder is written.
cd /d "%~dp0..\.."
".venv\Scripts\python.exe" -m streamlit run "Claude\fixed\app_fixed.py"
pause
