#!/usr/bin/env python3
"""
Correctness tests — delegates to tests/run_correctness.py.

Usage:
    python test_correctness.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))

from run_correctness import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
