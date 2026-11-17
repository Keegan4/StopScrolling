# PyInstaller spec: one-folder build of StopScrolling with all models bundled.
# Build with:  python packaging/build.py   (downloads models, builds, self-tests, archives)

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).parent
MODELS = ROOT / "models"

datas, binaries, hiddenimports = [], [], []
# mediapipe loads its native library via importlib.resources from mediapipe.tasks.c;
# onnxruntime / tokenizers ship native extensions + data PyInstaller can't fully infer.
for pkg in ("mediapipe", "onnxruntime", "tokenizers"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

model_files = [
    "face_landmarker.task",
    "efficientdet_lite0.tflite",
    "efficientdet_lite2.tflite",
    "clip/vision_model_quantized.onnx",
    "clip/text_model_quantized.onnx",
    "clip/tokenizer.json",
]
for rel in model_files:
    src = MODELS / rel
    if not src.exists():
        raise SystemExit(f"missing model {src} - run: python tools/download_models.py --all")
    datas.append((str(src), str(Path("models") / rel).replace("\\", "/").rsplit("/", 1)[0]))

a = Analysis(
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + ["stopscrolling.alerts.worker", "PIL.ImageTk", "tkinter"],
    # (matplotlib must stay: mediapipe.tasks.python.vision imports it at load time)
    excludes=["pytest", "IPython", "jax", "torch", "tensorflow", "onnxruntime.quantization"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="StopScrolling",
    console=True,  # shows the status log; closing it quits the app
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="StopScrolling", upx=False)
