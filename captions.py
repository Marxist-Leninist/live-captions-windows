#!/usr/bin/env python3
"""
Live Captions for Windows (system audio).

Pipeline:
  WASAPI loopback capture  ->  VAD/silence segmenter  ->  faster-whisper  ->  overlay

Hotkeys (when overlay has focus, or globally via click):
  F9   toggle visible
  Esc  quit

Usage:
  python captions.py                # GPU (CUDA float16)
  python captions.py --cpu          # force CPU (int8)
  python captions.py --model small.en
"""
import argparse
import os
import queue
import sys
import threading
import time
import traceback
from collections import deque

# Early stderr/stdout capture so pythonw crashes land in captions.log.
try:
    _logfh = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "captions.log"), "a", buffering=1)
    sys.stderr = _logfh
    sys.stdout = _logfh
    print(f"\n=== captions.py started at {time.strftime('%Y-%m-%d %H:%M:%S')} pid={os.getpid()} ===")
except Exception:
    pass

import numpy as np
import pyaudiowpatch as pyaudio
from faster_whisper import WhisperModel
import tkinter as tk

TARGET_SR = 16000
CHUNK_MS = 30
VAD_SILENCE_MS = 500
MAX_BUFFER_S = 4.0          # shorter = lower latency; small/medium models need this
MIN_SPEAK_S = 0.4
RMS_THRESHOLD = 0.006  # float32 RMS (normalized)
MAX_QUEUE_BACKLOG = 1       # drop older segments if transcribe can't keep up


def resample_mono(raw_int16: np.ndarray, src_sr: int, src_channels: int) -> np.ndarray:
    a = raw_int16.astype(np.float32) / 32768.0
    if src_channels > 1:
        a = a.reshape(-1, src_channels).mean(axis=1)
    if src_sr != TARGET_SR:
        n_target = int(len(a) * TARGET_SR / src_sr)
        if n_target > 0:
            a = np.interp(
                np.linspace(0, len(a), n_target, endpoint=False),
                np.arange(len(a)),
                a,
            ).astype(np.float32)
    return a


class CaptureThread(threading.Thread):
    daemon = True

    def __init__(self, out_q: queue.Queue, source: str = "system"):
        super().__init__()
        self.out_q = out_q
        self.running = True
        self.pa = pyaudio.PyAudio()
        self.err: str | None = None
        self.source = source  # "system" or "mic"

    def _find_loopback(self):
        try:
            default_spk = self.pa.get_default_output_device_info()
            want = default_spk["name"]
        except Exception:
            want = None
        match = None
        for i in range(self.pa.get_device_count()):
            d = self.pa.get_device_info_by_index(i)
            if d.get("isLoopbackDevice"):
                if want and d["name"].startswith(want):
                    return d
                if match is None:
                    match = d
        return match

    def _find_mic(self):
        try:
            return self.pa.get_default_input_device_info()
        except Exception:
            pass
        for i in range(self.pa.get_device_count()):
            d = self.pa.get_device_info_by_index(i)
            if d.get("maxInputChannels", 0) > 0 and not d.get("isLoopbackDevice"):
                return d
        return None

    def run(self):
        dev = self._find_mic() if self.source == "mic" else self._find_loopback()
        if dev is None:
            self.err = f"no device for source={self.source!r}"
            return
        sr = int(dev["defaultSampleRate"])
        ch = int(dev["maxInputChannels"])
        frames = int(sr * CHUNK_MS / 1000)
        try:
            stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=ch,
                rate=sr,
                input=True,
                frames_per_buffer=frames,
                input_device_index=dev["index"],
            )
        except Exception as e:
            self.err = f"open stream failed: {e}"
            return
        while self.running:
            try:
                data = stream.read(frames, exception_on_overflow=False)
            except Exception as e:
                self.err = f"read err: {e}"
                break
            raw = np.frombuffer(data, dtype=np.int16)
            self.out_q.put(resample_mono(raw, sr, ch))
        stream.stop_stream()
        stream.close()
        self.pa.terminate()


class MixerThread(threading.Thread):
    """Pair chunks from two queues and sum into one. Gaps filled with silence."""
    daemon = True

    def __init__(self, in_a: queue.Queue, in_b: queue.Queue, out_q: queue.Queue):
        super().__init__()
        self.in_a = in_a
        self.in_b = in_b
        self.out_q = out_q
        self.running = True

    def run(self):
        while self.running:
            try:
                a = self.in_a.get(timeout=0.5)
            except queue.Empty:
                a = None
            try:
                b = self.in_b.get(timeout=0.05)
            except queue.Empty:
                b = None
            if a is None and b is None:
                continue
            if a is None:
                a = np.zeros_like(b)
            if b is None:
                b = np.zeros_like(a)
            n = min(len(a), len(b))
            if n == 0:
                continue
            # Both sources already scaled to [-1, 1]; sum and clip to avoid overflow.
            mix = a[:n] + b[:n]
            np.clip(mix, -1.0, 1.0, out=mix)
            self.out_q.put(mix.astype(np.float32))


class VADThread(threading.Thread):
    """Accumulate audio; emit a segment when silence > VAD_SILENCE_MS after speech,
    or when buffer reaches MAX_BUFFER_S."""
    daemon = True

    def __init__(self, in_q: queue.Queue, out_q: queue.Queue):
        super().__init__()
        self.in_q = in_q
        self.out_q = out_q
        self.running = True

    def run(self):
        buf: list[np.ndarray] = []
        buf_samples = 0
        silent_samples = 0
        speech_samples = 0
        silence_limit = int(TARGET_SR * VAD_SILENCE_MS / 1000)
        max_samples = int(TARGET_SR * MAX_BUFFER_S)
        min_speech = int(TARGET_SR * MIN_SPEAK_S)
        last_chunk_time = time.time()

        while self.running:
            try:
                chunk = self.in_q.get(timeout=0.5)
                last_chunk_time = time.time()
            except queue.Empty:
                # Watchdog: if we have buffered speech and capture has stalled for >2s,
                # flush what we have instead of letting the overlay freeze on stale text.
                if speech_samples >= min_speech and time.time() - last_chunk_time > 2.0:
                    try:
                        self.out_q.put(np.concatenate(buf))
                    except Exception:
                        pass
                    buf = []
                    buf_samples = silent_samples = speech_samples = 0
                continue
            try:
                rms = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0
            except Exception:
                rms = 0.0
            is_speech = rms > RMS_THRESHOLD
            buf.append(chunk)
            buf_samples += len(chunk)
            if is_speech:
                speech_samples += len(chunk)
                silent_samples = 0
            else:
                silent_samples += len(chunk)

            flush = False
            if speech_samples >= min_speech and silent_samples >= silence_limit:
                flush = True
            elif buf_samples >= max_samples and speech_samples >= min_speech:
                flush = True
            elif buf_samples >= max_samples and speech_samples < min_speech:
                # Drop the buffer, no speech in it.
                buf = []
                buf_samples = silent_samples = speech_samples = 0
                continue

            if flush:
                try:
                    audio = np.concatenate(buf)
                    self.out_q.put(audio)
                except Exception as _e:
                    print(f"[VAD flush err] {_e}")
                buf = []
                buf_samples = silent_samples = speech_samples = 0


class TranscribeThread(threading.Thread):
    daemon = True

    def __init__(self, in_q: queue.Queue, ui_cb, model_name: str, force_cpu: bool, compute_type: str | None = None):
        super().__init__()
        self.in_q = in_q
        self.cb = ui_cb
        self.running = True
        self.model_name = model_name
        self.force_cpu = force_cpu
        self.compute_type = compute_type

    def run(self):
        # GTX 1050 Ti / other Pascal cards lack efficient fp16 in CTranslate2.
        # int8   — fastest on Pascal (pure 8-bit, ~30% quicker than int8_float32)
        # int8_float32 — quantized weights + fp32 accumulation (slightly higher quality)
        if self.compute_type:
            devices = [("cuda", self.compute_type)] if not self.force_cpu else [("cpu", self.compute_type)]
        elif self.force_cpu:
            devices = [("cpu", "int8")]
        else:
            devices = [("cuda", "int8"), ("cuda", "int8_float32"), ("cpu", "int8")]
        model = None
        last_err = None
        for dev, ct in devices:
            try:
                self.cb(f"Loading {self.model_name} on {dev}...")
                model = WhisperModel(self.model_name, device=dev, compute_type=ct)
                self.cb(f"Ready ({dev}/{ct}). Play something.")
                break
            except Exception as e:
                last_err = e
                self.cb(f"{dev} load failed: {e}")
                continue
        if model is None:
            self.cb(f"Failed to load model: {last_err}")
            return

        while self.running:
            try:
                audio = self.in_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if audio is None or len(audio) < 1600:
                continue
            # Drop backlog: if more segments are already waiting, skip stale ones
            # and only transcribe the newest. Prevents captions falling seconds
            # behind live audio when the model is slower than realtime.
            dropped = 0
            while self.in_q.qsize() > MAX_QUEUE_BACKLOG:
                try:
                    audio = self.in_q.get_nowait()
                    dropped += 1
                except queue.Empty:
                    break
            if dropped:
                print(f"[transcribe] dropped {dropped} stale segments to catch up")
            try:
                segments, _info = model.transcribe(
                    audio,
                    language="en",
                    beam_size=1,
                    vad_filter=False,
                    condition_on_previous_text=False,
                )
                text = " ".join(s.text.strip() for s in segments).strip()
            except Exception as e:
                print(f"[transcribe err] {e}\n{traceback.format_exc()}")
                self.cb(f"[transcribe err: {e}]")
                # Try to reload model rather than die silently.
                try:
                    del model
                    model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
                    print("[transcribe] reloaded on cpu/int8 after error")
                except Exception as e2:
                    print(f"[transcribe] reload failed: {e2}")
                continue
            if text:
                self.cb(text, finalize=True)


class CaptionUI:
    def __init__(self, alpha: float = 0.55, transparent_bg: bool = False):
        self.root = tk.Tk()
        self.root.title("Live Captions")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        # With transparent_bg the black backdrop becomes fully see-through via
        # -transparentcolor; keep window alpha at 1.0 so the text itself stays 100%
        # opaque. Only apply the alpha arg when NOT using transparent_bg.
        if transparent_bg:
            self.root.attributes("-alpha", 1.0)
        else:
            self.root.attributes("-alpha", max(0.05, min(1.0, alpha)))
        self.root.configure(bg="#000000")
        if transparent_bg:
            try:
                self.root.attributes("-transparentcolor", "#000000")
            except tk.TclError:
                pass
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        w, h = int(sw * 0.7), 130
        x = (sw - w) // 2
        y = 20
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self._reassert_topmost()

        self.label = tk.Label(
            self.root,
            text="Starting...",
            font=("Segoe UI", 18, "bold"),
            fg="#FFFFFF",
            bg="#000000",
            wraplength=w - 20,
            justify="center",
        )
        self.label.pack(expand=True, fill="both", padx=10, pady=10)

        self.root.bind("<Escape>", lambda _e: self.shutdown())
        self.root.bind_all("<F9>", lambda _e: self.toggle())
        self.root.protocol("WM_DELETE_WINDOW", self.shutdown)

        self.label.bind("<Button-1>", self._drag_start)
        self.label.bind("<B1-Motion>", self._drag)
        self.label.bind("<Button-3>", lambda _e: self.shutdown())

        self.history: deque[str] = deque(maxlen=2)
        self.visible = True
        self.shutdown_cb = None

    def _drag_start(self, e):
        self._dx = e.x_root - self.root.winfo_x()
        self._dy = e.y_root - self.root.winfo_y()

    def _drag(self, e):
        self.root.geometry(f"+{e.x_root - self._dx}+{e.y_root - self._dy}")

    def _reassert_topmost(self):
        try:
            self.root.attributes("-topmost", False)
            self.root.attributes("-topmost", True)
            self.root.lift()
        except tk.TclError:
            return
        self.root.after(1000, self._reassert_topmost)

    def toggle(self):
        if self.visible:
            self.root.withdraw()
        else:
            self.root.deiconify()
        self.visible = not self.visible

    def set_text(self, text: str, finalize: bool = False):
        def apply():
            if finalize:
                self.history.append(text)
                combined = " ".join(self.history)
            else:
                combined = text
            self.label.config(text=combined)
        self.root.after(0, apply)

    def shutdown(self):
        if self.shutdown_cb:
            try: self.shutdown_cb()
            except: pass
        try: self.root.destroy()
        except: pass

    def run(self):
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="base.en",
                    help="faster-whisper model. Base: tiny.en, base.en, small.en, medium.en. "
                         "Distilled (2x faster): Systran/faster-distil-whisper-small.en, "
                         "Systran/faster-distil-whisper-medium.en-v2")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    ap.add_argument("--compute-type", default=None,
                    help="ctranslate2 compute type. Default auto (int8 on GPU, int8 on CPU). "
                         "Options: int8 (fastest), int8_float32 (more stable), float32 (reference). "
                         "Pascal GPUs cannot use float16/bfloat16.")
    ap.add_argument("--source", choices=["system", "mic", "both"], default="system",
                    help="'system' = WASAPI loopback; 'mic' = default input; 'both' = mix of system+mic")
    ap.add_argument("--alpha", type=float, default=0.55,
                    help="overlay opacity 0.0 (invisible) to 1.0 (opaque). Default 0.55")
    ap.add_argument("--transparent-bg", action="store_true",
                    help="make the overlay background fully transparent (text floats, no backdrop)")
    args = ap.parse_args()

    audio_q: queue.Queue = queue.Queue()
    segment_q: queue.Queue = queue.Queue()

    ui = CaptionUI(alpha=args.alpha, transparent_bg=args.transparent_bg)

    captures: list[CaptureThread] = []
    mixer: MixerThread | None = None
    if args.source == "both":
        q_sys: queue.Queue = queue.Queue()
        q_mic: queue.Queue = queue.Queue()
        captures.append(CaptureThread(q_sys, source="system"))
        captures.append(CaptureThread(q_mic, source="mic"))
        mixer = MixerThread(q_sys, q_mic, audio_q)
    else:
        captures.append(CaptureThread(audio_q, source=args.source))
    capture = captures[0]  # for err reporting below
    vad = VADThread(audio_q, segment_q)
    transcribe = TranscribeThread(segment_q, ui.set_text, args.model, args.cpu, args.compute_type)

    def stop():
        for c in captures:
            c.running = False
        if mixer:
            mixer.running = False
        vad.running = False
        transcribe.running = False

    ui.shutdown_cb = stop

    for c in captures:
        c.start()
    if mixer:
        mixer.start()
    vad.start()
    transcribe.start()

    # Supervisor: if any worker dies, respawn it so the overlay never goes stale.
    def supervisor():
        nonlocal vad, transcribe
        while any(c.running for c in captures) or vad.running or transcribe.running:
            time.sleep(2.0)
            for i, c in enumerate(list(captures)):
                if c.running and not c.is_alive():
                    print(f"[supervisor] capture[{i}] ({c.source}) died, respawning")
                    target_q = audio_q if args.source != "both" else (q_sys if c.source == "system" else q_mic)
                    nc = CaptureThread(target_q, source=c.source)
                    captures[i] = nc
                    nc.start()
            if vad.running and not vad.is_alive():
                print("[supervisor] VAD died, respawning")
                seg_q = segment_q
                new_vad = VADThread(audio_q, seg_q)
                new_vad.start()
                vad = new_vad
            if transcribe.running and not transcribe.is_alive():
                print("[supervisor] transcribe died, respawning")
                new_t = TranscribeThread(segment_q, ui.set_text, args.model, args.cpu, args.compute_type)
                new_t.start()
                transcribe = new_t
    sup_thread = threading.Thread(target=supervisor, daemon=True)
    sup_thread.start()

    time.sleep(0.3)
    errs = [c.err for c in captures if c.err]
    if errs:
        ui.set_text("Audio err: " + " | ".join(errs), finalize=True)

    try:
        ui.run()
    finally:
        stop()


if __name__ == "__main__":
    main()
