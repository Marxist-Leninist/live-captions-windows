@echo off
REM Live Captions from system audio + microphone mixed.
cd /d "%~dp0"
start "" pythonw captions.py --source both --transparent-bg %*
