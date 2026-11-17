"""Build a standalone StopScrolling folder for the current OS and archive it.

    python packaging/build.py            # -> dist/StopScrolling-<version>-<os>-<arch>.zip / .tar.gz

PyInstaller cannot cross-compile, so run this once on each OS (CI does this for releases).
Steps: download models -> PyInstaller -> copy user folders/readme -> --self-test -> archive.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
APP_DIR = DIST / "StopScrolling"


def version() -> str:
    sys.path.insert(0, str(ROOT / "src"))
    from stopscrolling import __version__

    return __version__


def platform_tag() -> str:
    system = {"Windows": "windows", "Darwin": "macos", "Linux": "linux"}.get(platform.system(), platform.system().lower())
    arch = platform.machine().lower()
    arch = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(arch, arch)
    return f"{system}-{arch}"


def run(*cmd: str) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> int:
    run(sys.executable, "tools/download_models.py", "--all")
    run(sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--distpath", str(DIST),
        "--workpath", str(ROOT / "build"), "packaging/stopscrolling.spec")

    # User-editable folders + docs next to the executable.
    for sub in ("memes", "sounds"):
        dest = APP_DIR / "assets" / sub
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "assets" / sub / "README.txt", dest / "README.txt")
    shutil.copy2(ROOT / "packaging" / "README-release.txt", APP_DIR / "README.txt")

    exe = APP_DIR / ("StopScrolling.exe" if platform.system() == "Windows" else "StopScrolling")
    run(str(exe), "--self-test")

    name = f"StopScrolling-{version()}-{platform_tag()}"
    if platform.system() == "Windows":
        archive = shutil.make_archive(str(DIST / name), "zip", DIST, "StopScrolling")
    else:  # tar keeps symlinks and the executable bit
        archive = str(DIST / f"{name}.tar.gz")
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(APP_DIR, arcname="StopScrolling")
    size = Path(archive).stat().st_size / 1e6
    print(f"built {archive} ({size:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
