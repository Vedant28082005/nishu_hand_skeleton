"""Model selection shared by realtime.py and eval_clips.py (--model {popsign,aslcitizen})."""

from __future__ import annotations

from pathlib import Path

CHOICES = ("popsign", "aslcitizen")

DEFAULTS = {
    "popsign": {"model": "models/asl-signs.onnx",
                "labels": "models/sign_to_prediction_index_map.json"},
    "aslcitizen": {"model": "models/aslcitizen_stgcn.onnx",
                   "labels": "models/aslcitizen_gloss.json"},
}


def load_model(name: str, model_path: str | None = None, labels_path: str | None = None,
               threads: int = 2, aslcitizen_norm: str = "train"):
    d = DEFAULTS[name]
    model_path = model_path or d["model"]
    labels_path = labels_path or d["labels"]
    if not Path(model_path).exists():
        hint = ("run scripts/fetch_models.sh" if name == "popsign"
                else "run `python export_stgcn.py` (needs PyTorch once) to create it")
        raise SystemExit(f"model file not found: {model_path}  ({hint})")
    if name == "popsign":
        from classifier import SignClassifier
        return SignClassifier(model_path, labels_path, threads=threads)
    from aslcitizen import AslCitizenClassifier
    return AslCitizenClassifier(model_path, labels_path, threads=threads, norm=aslcitizen_norm)
