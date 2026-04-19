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

MODELS (English-only variants are fastest)
  tiny.en    ~40MB   fast, ok accuracy
  base.en    ~140MB  [default] best speed/accuracy tradeoff
  small.en   ~460MB  notably better, still realtime on 1050 Ti
  medium.en  ~1.5GB  great, slower, may lag on long bursts
