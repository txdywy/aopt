"""Search one-cell vector-parity exchanges using the joint V/F greedy oracle."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import subprocess


DEFAULT = {(31, 11), (31, 12), (31, 13), (31, 14), (16, 14), (21, 13), (21, 14)}


def encode(values):
    return ",".join(f"{g}:{r}" for g, r in sorted(values))


def trial(values):
    env = os.environ.copy()
    env.update(
        {
            "VECTOR_PARITY_SET": encode(values),
            "TARGET": "959",
            "ENGINES": "valu,flow",
            "GREEDY_TRIALS": os.environ.get("PARITY_GREEDY_TRIALS", "1"),
            "GREEDY_START": os.environ.get("PARITY_GREEDY_START", "2103"),
            "GREEDY_DIRECTION": os.environ.get("PARITY_GREEDY_DIRECTION", "forward"),
            "OUT_PREFIX": "/tmp/parity-search-unused",
        }
    )
    p = subprocess.run(
        ["python3", "-m", "codex.solve_projected_engines"],
        cwd=os.getcwd(), env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=60, check=False,
    )
    scores = [int(x) for x in re.findall(r"greedy score=(\d+)", p.stdout)]
    best = min(scores, default=10**9)
    return best, encode(values), p.stdout


def main():
    candidates = [
        DEFAULT | {(g, r)}
        for g in range(32)
        for r in range(4)
        if (g, r) not in DEFAULT
    ]
    workers = int(os.environ.get("PARITY_WORKERS", "4"))
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(trial, values) for values in candidates]
        for number, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
            except Exception as error:
                print(f"trial_error={error}", flush=True)
                continue
            results.append(result)
            score, parity, output = result
            print(f"trial={number}/{len(candidates)} score={score} parity={parity}", flush=True)
            if score <= 959:
                print(output, flush=True)
    for result in sorted(results)[:20]:
        print(f"best score={result[0]} parity={result[1]}", flush=True)


if __name__ == "__main__":
    main()
