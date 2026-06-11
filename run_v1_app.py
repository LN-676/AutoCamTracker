"""Run AutoCamTracker V1 from the repository root.

This wrapper is convenient for VSCode's "Run Python File" action because it
sets the working directory and import path before launching the Tkinter app.
"""

from __future__ import annotations

from pathlib import Path
import os
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
V1_DIR = PROJECT_ROOT / "code" / "V1"

os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(V1_DIR))

from app import main


if __name__ == "__main__":
    main()
