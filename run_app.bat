@echo off
REM Launch the pronunciation coach locally (double-click this file).
REM First run creates .venv and installs requirements (~2 GB, torch included);
REM later runs start in seconds. Pass --update to reinstall requirements.
REM
REM Messages are ASCII on purpose: cmd.exe reads batch files by byte offset,
REM so non-ASCII text here desynchronises the parser and corrupts later lines.
setlocal
cd /d "%~dp0"

set "VENV=.venv"
set "PY=%VENV%\Scripts\python.exe"
set "STAMP=%VENV%\.requirements.stamp"

REM --- find a Python launcher for the first run --------------------------------
set "BOOTSTRAP="
where py >nul 2>&1 && set "BOOTSTRAP=py -3"
if not defined BOOTSTRAP where python >nul 2>&1 && set "BOOTSTRAP=python"
if not defined BOOTSTRAP if not exist "%PY%" (
    echo [!] Python was not found.
    echo     Install Python 3.11+ from https://www.python.org/downloads/
    echo     and tick "Add python.exe to PATH", then run this file again.
    goto :fail
)

REM --- create the virtual environment on first run -----------------------------
if not exist "%PY%" (
    echo [1/3] Creating the virtual environment in .venv ...
    %BOOTSTRAP% -m venv "%VENV%"
    if errorlevel 1 (
        echo [!] Could not create the virtual environment.
        goto :fail
    )
)

REM --- install dependencies when missing, outdated, or --update ----------------
set "NEEDS_INSTALL="
if not exist "%STAMP%" set "NEEDS_INSTALL=1"
if /i "%~1"=="--update" set "NEEDS_INSTALL=1"
if not defined NEEDS_INSTALL call :newer requirements.txt "%STAMP%" && set "NEEDS_INSTALL=1"
if defined NEEDS_INSTALL (
    echo [2/3] Installing dependencies. The first run downloads about 2 GB
    echo       and can take 10 minutes or more ...
    "%PY%" -m pip install --upgrade pip
    "%PY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [!] Dependency installation failed. Check your network and retry.
        goto :fail
    )
    echo installed> "%STAMP%"
)

REM --- LLM feedback needs a key; the analysis itself works without one ---------
if not defined GEMINI_API_KEY if not exist ".streamlit\secrets.toml" (
    echo [i] GEMINI_API_KEY is not set: coaching text and JP-KR translation
    echo     stay unavailable. Scoring and the phoneme analysis work as usual.
    echo     To enable them: set GEMINI_API_KEY=your-key
)

echo [3/3] Starting the app. Your browser will open shortly.
echo       Press Ctrl+C in this window to stop it.
"%PY%" -m streamlit run app.py
if errorlevel 1 goto :fail
endlocal
exit /b 0

REM --- :newer <file> <reference> -> errorlevel 0 when <file> is newer ----------
:newer
for /f %%t in ('powershell -NoProfile -Command "try { if ((Get-Item '%~1').LastWriteTime -gt (Get-Item '%~2').LastWriteTime) { 1 } else { 0 } } catch { 0 }"') do set "NEWER=%%t"
if "%NEWER%"=="1" exit /b 0
exit /b 1

:fail
echo.
echo See the Installation section of README.md if the problem persists.
pause
endlocal
exit /b 1
