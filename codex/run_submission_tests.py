"""Run the repository's unchanged submission tests against codex.KernelBuilder."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

from codex import perf_takehome_under1000 as optimized

# submission_tests imports the root module name.  Supplying our independent
# module through the import cache avoids copying over the user's root solution.
sys.modules["perf_takehome"] = optimized
runpy.run_path(str(ROOT / "tests" / "submission_tests.py"), run_name="__main__")
