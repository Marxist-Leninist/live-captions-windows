@echo off
REM Live Captions launcher (no console window).
REM Drop optional args after --: e.g. captions.bat -- --model small.en
cd /d "%~dp0"
start "" pythonw captions.py --transparent-bg %*
