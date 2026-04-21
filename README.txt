Live Captions - WASAPI loopback + faster-whisper
=================================================

RUN
  captions.bat         system audio (default, for Discord/YouTube/etc.)
  captions-mic.bat     microphone only
  captions-both.bat    system audio + mic MIXED (meetings: you + everyone else)

  python captions.py --source {system,mic,both}
  python captions.py --alpha 0.4              # 0.0 invisible .. 1.0 opaque
  python captions.py --transparent-bg         # floating text, no backdrop
  python captions.py --model small.en         # more accurate, ~3x slower
  python captions.py --cpu                    # force CPU

CONTROLS
  F9      toggle overlay show/hide (global while app running)
  Esc     quit (overlay focused)
  Right-click overlay  quit
  Drag    left-click-hold on caption text to move window

NOTES
- Captions come from whatever your default output device is playing.
  If you change speakers/headphones, restart the app.
- VAD based on RMS - first syllables can be clipped on very quiet audio.
  Adjust RMS_THRESHOLD (captions.py, default 0.006) if needed.
- Model is cached to %USERPROFILE%\.cache\huggingface\hub after first run.
- Pascal GPUs (GTX 10xx) use int8_float32 on CUDA (fp16 not efficient).

ENGINES
  whisper (default)  faster-whisper CT2 — small.en / medium.en / distil-large-v3
  sensevoice         Alibaba SenseVoice-Small, non-Whisper non-autoregressive
                     architecture, similar WER with different hallucination profile
  parakeet           NVIDIA Parakeet-TDT-0.6B — OOMs on 4 GB VRAM, DO NOT USE

WHISPER MODELS (English-only variants are fastest)
  tiny.en                                      ~40MB    WER ~12%, 10x realtime
  base.en                                      ~140MB   WER ~11%, 7x realtime
  Systran/faster-distil-whisper-small.en       ~330MB   WER  ~9%, 5-6x realtime
  Systran/faster-distil-whisper-medium.en      ~760MB   WER  ~8%, 2-3x realtime
  Systran/faster-distil-whisper-large-v3       ~1.5GB   WER ~7.4%, 1.5x realtime

SENSEVOICE (single model, encoder-only)
  iic/SenseVoiceSmall                          ~900MB   WER ~7.5%, ~1.4x realtime
                                                         non-Whisper, 234M params

SWITCHING
  - Use a different .bat launcher (captions-small.bat, captions-sensevoice.bat, etc.)
  - Or:  python captions.py --engine sensevoice
         python captions.py --engine whisper --model tiny.en
  - Only one instance at a time; kill previous before launching new.
  - First run of a new model downloads weights to the HF / ModelScope cache.
