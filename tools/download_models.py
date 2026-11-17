"""Download the MediaPipe model files into ./models.

Usage:
    python tools/download_models.py            # face landmarker + EfficientDet-Lite0
    python tools/download_models.py --all      # also EfficientDet-Lite2 (slower, better at phones)
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
BASE = "https://storage.googleapis.com/mediapipe-models"
CLIP = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main"

MODELS = {
    "face_landmarker.task": f"{BASE}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "efficientdet_lite0.tflite": f"{BASE}/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite",
    # CLIP ViT-B/32 (quantized ONNX) for the phone-vs-calculator check
    "clip/vision_model_quantized.onnx": f"{CLIP}/onnx/vision_model_quantized.onnx",
    "clip/text_model_quantized.onnx": f"{CLIP}/onnx/text_model_quantized.onnx",
    "clip/tokenizer.json": f"{CLIP}/tokenizer.json",
}
OPTIONAL_MODELS = {
    "efficientdet_lite2.tflite": f"{BASE}/object_detector/efficientdet_lite2/float32/latest/efficientdet_lite2.tflite",
}


def download(name: str, url: str, force: bool) -> None:
    target = MODELS_DIR / name
    if target.exists() and not force:
        print(f"  ok      {name} (already present)")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    print(f"  fetch   {name} ...", end="", flush=True)
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(target)
    print(f" {target.stat().st_size / 1e6:.1f} MB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="also download optional models")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    wanted = dict(MODELS)
    if args.all:
        wanted.update(OPTIONAL_MODELS)
    print(f"Downloading models into {MODELS_DIR}")
    for name, url in wanted.items():
        download(name, url, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
