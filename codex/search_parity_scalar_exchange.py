"""Try parity-vectorization plus one VALU->ALU scalar cell exchange.

The exchange preserves aggregate ALU/VALU pressure (8 scalar lanes for one
vector op) while changing only the cross-engine dependency shape.  It is a
cheap way to turn the exact mirror-bit ALU/LOAD core into a candidate that
still has a 959-cycle projected VALU/FLOW calendar.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import subprocess


BASE_PARITY = "16:14,21:13,21:14,31:11,31:12,31:13,31:14"
BASE_HASH1 = "18:15"


def encode(values):
    return ",".join(f"{g}:{r}" for g, r in sorted(values))


def trial(spec):
    parity, kind, cell = spec
    env = os.environ.copy()
    env.update(
        {
            "VECTOR_PARITY_SET": f"{BASE_PARITY},{parity}",
            "TARGET": "959",
            "ENGINES": "valu,flow",
            "GREEDY_TRIALS": os.environ.get("PARITY_GREEDY_TRIALS", "1"),
            "GREEDY_START": os.environ.get("PARITY_GREEDY_START", "2103"),
            "GREEDY_DIRECTION": "forward",
            "OUT_PREFIX": "/tmp/parity-scalar-unused",
        }
    )
    if kind == "hash1":
        env["SCALAR_HASH1_JOIN"] = f"{BASE_HASH1},{cell}"
    elif kind == "hash23":
        env["SCALAR_HASH23_JOIN"] = cell
    elif kind == "hash5":
        env["SCALAR_HASH5_JOIN"] = cell
    elif kind == "finalshift":
        env["SCALAR_FINAL_SHIFT"] = cell.split(":", 1)[0]
    elif kind == "finaljoin":
        env["SCALAR_FINAL_JOIN"] = cell.split(":", 1)[0]
    else:
        raise ValueError(kind)
    p = subprocess.run(
        ["python3", "-m", "codex.solve_projected_engines"],
        cwd=os.getcwd(), env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=60, check=False,
    )
    scores = [int(x) for x in re.findall(r"greedy score=(\d+)", p.stdout)]
    return min(scores, default=10**9), kind, cell, parity, p.stdout


def main():
    # Start with phase-near cells.  The broad scalar sets can be expanded once
    # a useful phase exchange is found.
    parity_cells = os.environ.get("PARITY_CELLS", "10:2;22:3;4:1;4:2;7:2;9:0;13:0;15:0").split(";")
    cells = os.environ.get("SCALAR_CELLS", "").split(",")
    if not cells or cells == [""]:
        cells = [f"{g}:{r}" for g in range(32) for r in (0, 1, 2, 3, 8, 9, 14, 15)]
    kinds = tuple(os.environ.get("SCALAR_KINDS", "hash1,hash23,hash5").split(","))
    specs = [(parity, kind, cell) for parity in parity_cells for kind in kinds for cell in cells]
    workers = int(os.environ.get("PARITY_WORKERS", "4"))
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(trial, spec) for spec in specs]
        for number, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
            except Exception as error:
                print(f"trial_error={error}", flush=True)
                continue
            results.append(result)
            score, kind, cell, parity, output = result
            print(f"trial={number}/{len(specs)} score={score} parity={parity} kind={kind} cell={cell}", flush=True)
            if score <= 959:
                print(output, flush=True)
    for score, kind, cell, parity, _ in sorted(results)[:40]:
        print(f"best score={score} parity={parity} kind={kind} cell={cell}", flush=True)


if __name__ == "__main__":
    main()
