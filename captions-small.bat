@echo off
REM small.en (~460 MB) - notably better accuracy, still realtime on GTX 1050 Ti
cd /d "%~dp0"
start "" pythonw captions.py --source both --transparent-bg --model small.en %*
