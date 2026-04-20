@echo off
REM medium.en (~1.5 GB) - best quality that fits 1050 Ti int8. ~1.5-2x realtime.
REM First run downloads ~1.5 GB model.
cd /d "%~dp0"
start "" pythonw captions.py --source both --transparent-bg --model medium.en %*
