# signtalk

Live ASL sign → speech. Laptop first, Raspberry Pi 5 second, same code.

## What's verified

Cloned `nlevites/signchat` (MIT) and confirmed the weights are **committed, not gitignored**:

| | |
|---|---|
| File | `asl-signs.onnx`, 22 MB |
| Input | `[T, 543, 3]` float32 — **variable T**, no fixed window |
| Output | `[250]` **raw logits** (apply softmax yourself) |
| Opset | 17, produced by tf2onnx 1.17.0 |
| Latency | 3.6 ms @ T=16 → 7.4 ms @ T=64 (dev CPU) |

The classifier is free. MediaPipe is ~95% of your compute budget.

## The spec that matters

Row order is **`[Face(468)] [LHand(21)] [Pose(33)] [RHand(21)]`** = 543.

Note pose sits *between* the hands — not the order you'd guess.

**Missing landmarks must be `NaN`, never `0`.** Normalization is baked into
the model: it normalizes finite values per clip and reads NaN as "not
detected". Writing zeros moves "not detected" into the data distribution and
quietly destroys accuracy. This is the one mistake that produces a pipeline
that runs fine and predicts garbage.

Other conventions:
- `x, y` normalized to [0,1]; `z` in image-width units, smaller = closer
- FaceLandmarker returns 478 (468 + 10 iris) — **truncate to the first 468**
- Handedness is relative to the **user**, not the camera. No flip needed.

## Files

| File | Role |
|---|---|
| `landmarks.py` | MediaPipe → (543,3) assembly, NaN padding, ring buffer |
| `gate.py` | Motion gate — sign start/end detection |
| `classifier.py` | ONNX wrapper, softmax, stable/band token admission |
| `realtime.py` | Webcam loop + audio |
| `test_pipeline.py` | End-to-end check, no camera needed |

Run `python test_pipeline.py` first — it passes on this machine and proves the
model and wiring before you add hardware variables.

## Setup

```bash
pip install onnxruntime numpy opencv-python mediapipe

# MediaPipe Tasks bundles into models/
wget -P models https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
wget -P models https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
wget -P models https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

# Pre-render the vocabulary once — zero TTS latency at runtime
pip install piper-tts
mkdir -p wavs
python -c "import json;[print(w) for w in json.load(open('models/sign_to_prediction_index_map.json'))]" \
  | while read w; do echo "$w" | piper -m en_US-lessac-medium -f "wavs/$w.wav"; done
```

```bash
python realtime.py --show                      # laptop
python realtime.py --width 640 --height 480 --no-face --threads 3   # Pi 5
```

## Tuning

Everything that decides whether this *feels* live lives in two files.

`gate.py` — thresholds in shoulder-widths per frame, so they're invariant to
how far the signer sits from the camera, but **not** to frame rate. Retune when
you move to the Pi:

```
start_velocity = 0.020    stop_velocity = 0.008
start_frames   = 3        stop_frames   = 8
min_frames     = 10       max_frames    = 90
```

`classifier.py` — admission, ported from signchat's `admit.ts`:

```
TOP1_THRESHOLD = 0.30     # "stable": this score + same label twice running
TOP2_THRESHOLD = 0.10     # "band":  mid confidence with a credible runner-up
refractory_s   = 1.0      # stops one long sign firing the same word repeatedly
```

Run with `--show` to watch `state` and `v` live, then set `start_velocity` just
above your resting jitter.

## Know this about the vocabulary

The 250 words are **PopSign's**, and PopSign exists to teach *parents* signs for
talking to their Deaf toddlers. So you get `potty`, `owie`, `kitty`, `nap`,
`napkin`, `puppy`, `yucky` — and no `work`, `help`, `name`, `understand`,
`hospital`, or `phone`.

It's a real 250-word ASL model and it demos well. It is **not** a vocabulary an
adult would hold a conversation in. If your friend is an actual user rather than
an audience, plan on the fine-tune path: freeze the trunk, retrain the final
layer on ~30 clips each of the words they actually need.

## Pi 5 notes

- `pip install onnxruntime` has ARM64 wheels — no build step
- **Pi 5 has no 3.5 mm jack.** USB audio dongle, I2S DAC, or BT speaker.
- Active cooling, or you'll thermal throttle mid-demo
- Start with `--no-face`: it's the largest single cost and only matters for
  signs distinguished by mouth shape
- If MediaPipe still can't keep up, the Hailo AI HAT is the next lever

## Provenance

Model and the admission logic from [`nlevites/signchat`](https://github.com/nlevites/signchat) (MIT),
trained on Google's Kaggle `asl-signs` / PopSign 250 dataset.
Python pipeline here is new.
