"""Bootstrap the backend environment and run common VS Code tasks.

This script lets a clean checkout run directly from VS Code. It creates the
project virtual environment, installs backend requirements when needed, and
then launches the selected command using that environment's Python.
"""
from __future__ import annotations

import os
from importlib import metadata
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
VENV = ROOT / ".venv"
REQUIREMENTS = BACKEND / "requirements.txt"


def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def install_backend_if_needed(python: Path) -> None:
    """Install requirements only if the local environment is incomplete."""
    required = ("fastapi", "uvicorn", "tensorflow", "pandas", "numpy", "scikit-learn", "joblib", "pydantic", "pytest", "httpx")
    check = subprocess.run(
        [str(python), "-c", "from importlib import metadata; names=" + repr(required) + "; [metadata.distribution(name) for name in names]"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if check.returncode == 0:
        return

    print("Setting up Trip Decider's Python environment. This may take a few minutes on first run.", flush=True)
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], cwd=ROOT, check=True)
    subprocess.run([str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS)], cwd=ROOT, check=True)


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {"backend", "train", "test"}:
        print("Usage: python vscode_runner.py {backend|train|test}", file=sys.stderr)
        return 2

    mode = sys.argv[1]
    if not venv_python().exists():
        print("Creating Trip Decider's project virtual environment…", flush=True)
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], cwd=ROOT, check=True)

    python = venv_python()
    install_backend_if_needed(python)

    if mode == "backend":
        port = os.environ.get("TRIP_DECIDER_BACKEND_PORT", "8000")
        command = [
            str(python), "-m", "uvicorn", "app.main:app", "--app-dir", str(BACKEND),
            "--reload", "--host", "127.0.0.1", "--port", port,
        ]
    elif mode == "train":
        command = [str(python), str(BACKEND / "scripts" / "train_model.py")]
    else:
        command = [str(python), "-m", "pytest", str(BACKEND / "tests"), "-q"]

    os.chdir(ROOT)
    if os.name == "nt":
        return subprocess.call(command, cwd=ROOT)
    os.execv(str(python), command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
