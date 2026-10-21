"""Run unchanged submission tests against the current standalone kernel."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

import perf_takehome as optimized

# Keep the public suite's import bound to the current standalone submission.
sys.modules["perf_takehome"] = optimized
runpy.run_path(str(ROOT / "tests" / "submission_tests.py"), run_name="__main__")
