#!/usr/bin/env python3
"""Sens sight worker entrypoint. All logic lives in the sight/ package."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sight.server import main

if __name__ == "__main__":
    main()
