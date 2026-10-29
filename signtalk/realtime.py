"""
Live loop: camera -> MediaPipe Tasks -> motion gate -> model -> phrase buffer -> speech.

Three threads, bounded queues, capture never blocks:

    capture --frame_q--> inference --audio_q--> audio

Laptop:
    python realtime.py --show
    python realtime.py --model aslcitizen --show

First time on a camera (writes calibration.json, auto-loaded afterwards):
    python realtime.py --calibrate

Raspberry Pi 5:
    python realtime.py --width 640 --height 480 --fps 15 --threads 3 --no-face

Needs the three MediaPipe .task bundles in models/ (see README). Face is optional;
--no-face is a large speedup and loses mouth-shape cues.

CLI change vs v1: --model is now the model FAMILY {popsign,aslcitizen};
the file path moved to --model-path.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from queue import Empty, Queue

import cv2
import numpy as np

import mpdetect
from calibrate import DEFAULT_PATH as CALIB_PATH, load_calibration, run_calibration
from classifier import TokenAdmitter
from engine import DEFAULT_REJECT, Engine
from gate import MotionGate
from models import CHOICES, load_model
from phrase import PhraseBuffer
from pipeline import DropStats, Speaker, put_drop_oldest
from session_log import SessionLogger

# Back-compat re-exports (v1 imported these from realtime).
build_detectors = mpdetect.build_detectors
detect_frame = mpdetect.detect_frame


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=CHOICES, default="popsign")
    p.add_argument("--model-path", default=None, help="override the .onnx path")
    p.add_argument("--labels", default=None,
                   help="popsign: sign->index JSON; aslcitizen: {index: gloss} JSON")
    p.add_argument("--aslcitizen-norm", choices=("train", "valid"), default="train",
                   help="shoulder-normalisation statistics: 'train' mirrors the training "
                        "code (incl. zero padding); 'valid' = real frames only (experiment)")
    p.add_argument("--tasks", default="models", help="dir holding *.task bundles")
    p.add_argument("--wavs", default="wavs", help="pre-rendered WAVs, one per word")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--video", default=None, help="read this file instead of a camera (testing)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--no-face", action="store_true", help="skip FaceLandmarker (faster)")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--show", action="store_true", help="preview window + debug overlay")
    # phrase / speech
    p.add_argument("--speak-immediately", action="store_true",
                   help="speak each word as it is committed (words stack up) instead of "
                        "buffering a phrase")
    p.add_argument("--phrase-gap", type=float, default=1.5,
                   help="seconds of stillness before the buffered phrase is spoken")
    # admission
    p.add_argument("--top1-threshold", type=float, default=0.30,
                   help="confident-commit probability (and tick-mode 'stable')")
    p.add_argument("--top2-threshold", type=float, default=0.10,
                   help="band admission floor for top-1 and the runner-up")
    p.add_argument("--player", default="aplay", help="command used to play a WAV (-q <file> appended)")
    p.add_argument("--reject-threshold", type=float, default=DEFAULT_REJECT,
                   help="top-1 probability floor; below it nothing is emitted")
    # gate / calibration
    p.add_argument("--calibrate", action="store_true",
                   help="sample 5 s of stillness, write calibration.json, exit")
    p.add_argument("--config", default=CALIB_PATH, help="calibration file (auto-loaded if present)")
    p.add_argument("--start-velocity", type=float, default=None)
    p.add_argument("--stop-velocity", type=float, default=None)
    # logging
    p.add_argument("--session-dir", default="sessions")
    p.add_argument("--no-log", action="store_true")
    return p.parse_args(argv)


# --- threads ----------------------------------------------------------------

class Capture(threading.Thread):
    """Reads frames; never blocks on downstream. Camera: drop-oldest. File: back-pressure."""

    def __init__(self, args, frame_q: Queue, stats: DropStats, stop: threading.Event):
        super().__init__(daemon=True, name="capture")
        self.args, self.q, self.stats, self.stop = args, frame_q, stats, stop
        self.is_file = args.video is not None
        self.cap = cv2.VideoCapture(args.video if self.is_file else args.camera)
        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            self.cap.set(cv2.CAP_PROP_FPS, args.fps)
        self.opened = self.cap.isOpened()
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or float(args.fps)
        self.finished = threading.Event()

    def run(self):
        t0, i = time.monotonic(), 0
        while not self.stop.is_set():
            ok, bgr = self.cap.read()
            if not ok:
                break
            ts_ms = (i / self.src_fps * 1000.0) if self.is_file else (time.monotonic() - t0) * 1000.0
            i += 1
            if self.is_file:                      # offline: don't drop, wait for inference
                while not self.stop.is_set():
                    try:
                        self.q.put((bgr, ts_ms), timeout=0.1)
                        break
                    except Exception:
                        pass
            else:
                put_drop_oldest(self.q, (bgr, ts_ms), self.stats)
        self.cap.release()
        self.finished.set()


class Inference(threading.Thread):
    def __init__(self, args, frame_q, show_q, engine, det, phrase, speaker, logger, stop, cap):
        super().__init__(daemon=True, name="inference")
        self.args, self.frame_q, self.show_q = args, frame_q, show_q
        self.engine, self.det, self.phrase = engine, det, phrase
        self.speaker, self.logger, self.stop, self.cap = speaker, logger, stop, cap
        self.ts = mpdetect.MonotonicTs()
        self.frames = 0
        self.t_start = time.monotonic()
        self.last_top3 = None
        self.infer_ms: list[float] = []
        self.track_ms: list[float] = []
        self.transcript: list[str] = []

    def _emit_words(self, words):
        if not words:
            return
        print(f"   >> SAY: {' '.join(words)}")
        self.transcript.extend(words)
        if self.speaker:
            self.speaker.say_phrase(words)

    def run(self):
        a, eng = self.args, self.engine
        last_report = time.monotonic()
        while not self.stop.is_set():
            try:
                bgr, ts_ms = self.frame_q.get(timeout=0.1)
            except Empty:
                if self.cap.finished.is_set() and self.frame_q.empty():
                    break
                self._poll_phrase()
                continue
            t0 = time.perf_counter()
            frame = mpdetect.detect_frame(mpdetect.to_mp_image(bgr), self.ts(ts_ms), self.det)
            self.track_ms.append((time.perf_counter() - t0) * 1000)
            self.frames += 1
            seg = eng.push(frame)
            if seg is not None:
                self.infer_ms.append(seg.infer_ms)
                self.last_top3 = seg.top[:3]
                tok = seg.token
                head = (f"[{seg.window.shape[0]:3d}f] "
                        + "  ".join(f"{l}:{s:.2f}" for l, s in seg.top[:3]))
                if tok:
                    print(f"{head}   -> {tok.label.upper()} ({tok.via})  [{seg.infer_ms:.0f} ms]")
                    if a.speak_immediately:
                        self._emit_words([tok.label])
                    else:
                        self.phrase.add(tok.label)
                else:
                    print(f"{head}   ({'rejected < floor' if seg.rejected else 'held'})")
                if self.logger:
                    self.logger.log_segment(
                        seg.window, seg.top, seg.velocity_trace,
                        emitted=tok.label if tok else None, via=tok.via if tok else None,
                        extra={"model": a.model, "infer_ms": round(seg.infer_ms, 2),
                               "rejected": seg.rejected})
            self._poll_phrase()

            if a.show:
                from overlay import draw_overlay
                fps = self.frames / max(1e-6, time.monotonic() - self.t_start)
                draw_overlay(bgr, frame, eng.gate.state, eng.gate.last_velocity,
                             eng.gate.start_velocity, self.last_top3, fps)
                put_drop_oldest(self.show_q, bgr)

            now = time.monotonic()
            if now - last_report >= 5.0:
                print(f"       ... {self.frames / (now - self.t_start):.1f} fps  "
                      f"(capture dropped {self.cap_stats.dropped})")
                last_report = now
        self._poll_phrase(force=True)
        self.stop.set()

    def _poll_phrase(self, force=False):
        if self.args.speak_immediately:
            return
        busy = self.engine.gate.state == "recording"
        words = self.phrase.flush() if force else self.phrase.poll(busy=busy)
        self._emit_words(words)


# --- main -------------------------------------------------------------------

def main(argv=None) -> None:
    args = parse_args(argv)

    cfg = load_calibration(args.config, args.fps) if not args.calibrate else None
    gate_kw = {}
    if cfg:
        gate_kw = {"start_velocity": cfg["start_velocity"], "stop_velocity": cfg["stop_velocity"]}
        print(f"[config] gate from {args.config}: start={cfg['start_velocity']} stop={cfg['stop_velocity']}")
    if args.start_velocity is not None:
        gate_kw["start_velocity"] = args.start_velocity
    if args.stop_velocity is not None:
        gate_kw["stop_velocity"] = args.stop_velocity

    stop = threading.Event()
    frame_q: Queue = Queue(maxsize=4)
    show_q: Queue = Queue(maxsize=1)
    cap_stats = DropStats()
    cap = Capture(args, frame_q, cap_stats, stop)
    if not cap.opened:
        raise SystemExit(f"cannot open {'video ' + args.video if args.video else 'camera ' + str(args.camera)}")

    det = mpdetect.build_detectors(Path(args.tasks), not args.no_face)

    if args.calibrate:
        cap.start()
        ts = mpdetect.MonotonicTs()

        def read_frame():
            try:
                bgr, ts_ms = frame_q.get(timeout=1.0)
            except Empty:
                return None
            return mpdetect.detect_frame(mpdetect.to_mp_image(bgr), ts(ts_ms), det)

        try:
            run_calibration(read_frame, out_path=args.config,
                            camera_info={"width": args.width, "height": args.height,
                                         "target_fps": args.fps, "face": not args.no_face})
        finally:
            stop.set()
            det.close()
        return

    model = load_model(args.model, args.model_path, args.labels, args.threads, args.aslcitizen_norm)
    warm = model.warmup()
    print(f"[warmup] one dummy inference: {warm * 1000:.0f} ms ({model.name})")

    gate = MotionGate(**gate_kw)
    engine = Engine(model, gate, TokenAdmitter(args.top1_threshold, args.top2_threshold),
                    args.reject_threshold)
    speaker = None if args.no_audio else Speaker(Path(args.wavs), player=args.player)
    phrase = PhraseBuffer(flush_after_s=args.phrase_gap)
    logger = None if args.no_log else SessionLogger(args.session_dir, meta={
        "argv": vars(args), "model": model.name, "labels_are_real": getattr(model, "labels_are_real", True),
        "gate": {"start_velocity": gate.start_velocity, "stop_velocity": gate.stop_velocity,
                 "start_frames": gate.start_frames, "stop_frames": gate.stop_frames,
                 "min_frames": gate.min_frames, "max_frames": gate.max_frames}})

    inf = Inference(args, frame_q, show_q, engine, det, phrase, speaker, logger, stop, cap)
    inf.cap_stats = cap_stats

    print(f"running  {args.width}x{args.height}  model={model.name}  face={'off' if args.no_face else 'on'}"
          f"  vocab={model.vocab_size}  speech={'immediate' if args.speak_immediately else f'phrase({args.phrase_gap}s)'}"
          f"  reject<{args.reject_threshold}    ctrl-c to stop\n")
    if logger:
        print(f"[log] {logger.dir}")

    cap.start()
    inf.start()
    try:
        while not stop.is_set():
            if args.show:
                try:
                    cv2.imshow("signtalk", show_q.get(timeout=0.05))
                except Empty:
                    pass
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        inf.join(timeout=3.0)
        if args.show:
            cv2.destroyAllWindows()
        if speaker:
            speaker.close(drain_s=3.0)
        if logger:
            logger.close()
        det.close()
        def _pct(xs, q):
            return f"{np.percentile(xs, q):.1f}" if xs else "n/a"
        print(f"\nframes={inf.frames}  tracking ms p50/p95={_pct(inf.track_ms,50)}/{_pct(inf.track_ms,95)}"
              f"  classify ms p50/p95={_pct(inf.infer_ms,50)}/{_pct(inf.infer_ms,95)}"
              f"  capture-dropped={cap_stats.dropped}"
              + (f"  audio-dropped={speaker.stats.dropped}" if speaker else ""))
        print(f"transcript: {' '.join(inf.transcript) or '(nothing)'}")


if __name__ == "__main__":
    main()
