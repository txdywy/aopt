"""Bounded, reproducible teacher-priority sweep; only executed scores count."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cases = list(itertools.product(("", "3,4", "31"), (64, 256, 1024, 4096), (1, 2, 4), (False, True)))

    def run(case):
        groups, weight, height, alu_stride = case
        command = [sys.executable, str(root / "codex/experiment_paired_base.py"),
                   "--arity", "4", "--round", "4", "--seeds", "1",
                   "--teacher-weight", str(weight), "--height-weight", str(height),
                   "--keep-final", "--compact-return", "--recycle-stride",
                   "--relax-jump", "--no-packet-repair"]
        command += ["--group-ids", groups] if groups else ["--groups", "0"]
        if alu_stride:
            command.append("--alu-stride")
        try:
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=45)
            record = {"command": command, "returncode": result.returncode}
            if result.returncode == 0:
                record["result"] = json.loads(result.stdout)
            else:
                record["error"] = result.stderr[-2000:]
        except subprocess.TimeoutExpired:
            record = {"command": command, "error": "timeout"}
        return record

    args.output.parent.mkdir(parents=True, exist_ok=True)
    best = None
    with args.output.open("w") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for case, record in zip(cases, pool.map(run, cases)):
            out.write(json.dumps(record) + "\n")
            out.flush()
            cycles = record.get("result", {}).get("cycles", [])
            score = max(cycles) if cycles else None
            if score is not None and (best is None or score < best):
                best = score
            print(json.dumps({"case": case, "cycles": score, "best": best}), flush=True)


if __name__ == "__main__":
    main()
