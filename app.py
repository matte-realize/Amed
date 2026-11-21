"""Vercel entrypoint: exposes the FastAPI app from src/hospital_twin as `app`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from hospital_twin.app import app  # noqa: E402,F401
