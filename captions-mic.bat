@echo off
REM Live Captions from microphone input (default input device).
cd /d "%~dp0"
start "" pythonw captions.py --source mic --transparent-bg %*
