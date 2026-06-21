@echo off
cd /d "%~dp0\Main"
call ..\.venv\Scripts\activate.bat
echo Starting Korean Pronunciation Correction App...
streamlit run app.py
pause
