"""Streamlit root entrypoint for Premier League Match Outcome & Probability Predictor.

Ensures proper sys.path resolution and launches app/dashboard.py.
Compatible with Streamlit Community Cloud deployment defaults.
"""

import os
import sys
from pathlib import Path

# Set root directory and src directory strictly at the top of sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
root_str = str(PROJECT_ROOT)
src_str = str(PROJECT_ROOT / "src")

for p in [src_str, root_str]:
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

# Evict any namespace package 'src' that lacks __file__ (e.g. /mount/src on Streamlit Cloud)
if "src" in sys.modules and getattr(sys.modules["src"], "__file__", None) is None:
    del sys.modules["src"]

from app.dashboard import main

if __name__ == "__main__":
    main()
