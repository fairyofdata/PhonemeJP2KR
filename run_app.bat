@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo Starting Korean Pronunciation Coach...
streamlit run app.py
pause
