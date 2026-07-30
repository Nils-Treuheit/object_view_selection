#!/usr/bin/env python3
"""
Smoke tests — delegates to tests/run_smoke.py.

Usage:
    python test_smoke.py --data_root /path/to/bottle
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))

from run_smoke import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
