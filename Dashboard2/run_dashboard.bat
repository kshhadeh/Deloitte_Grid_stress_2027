@echo off
REM Starts Dashboard2 with the project's .venv (double-click this file).
cd /d "%~dp0.."
".venv\Scripts\python.exe" -m streamlit run "Dashboard2\app.py"
pause
