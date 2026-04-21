@echo off
REM SenseVoice-Small (Alibaba) - non-Whisper, non-autoregressive encoder
REM Source: source + mic mixed, transparent bg, GPU-only.
REM   WER ~7.5%% English, ~1 GB VRAM, 1-2x realtime on 1050 Ti
cd /d "%~dp0"
start "" pythonw captions.py --source both --transparent-bg --engine sensevoice --font-size 54 %*
