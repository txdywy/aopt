"""Search launch phases and workspace colors against the joint Flow/Load CP.

The first traversal assigns physical mux workspaces before scheduling.  Every
legal reuse therefore inserts WAW/WAR edges between otherwise independent
groups.  This tool mutates the software-pipeline launch phases, interval-colors
the first traversal onto the same eight physical workspaces, and asks the
exact projected Flow+Load solver for a short infeasibility certificate.

Candidates that survive the short solve are printed and saved for a longer
all-engine run.  The subprocess boundary intentionally gives every CP-SAT
model a clean global kernel configuration and makes the search safely
parallel.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import heapq
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target


def normalize(offsets: list[int]) -> tuple[int, ...]:
    low = min(offsets)
    return tuple(value - low for value in offsets)


def color(
    offsets: tuple[int, ...],
    *,
    seed: int,
    count: int = 8,
) -> tuple[int, ...] | None:
    """Randomized optimal interval coloring for first-traversal live ranges."""
    rng = random.Random(seed)
    assignment = [-1] * kernel.N_GROUPS
    active: list[tuple[int, int]] = []
    free = list(range(count))
    tie_noise = {group: rng.random() for group in range(kernel.N_GROUPS)}
    for group in sorted(
        range(kernel.N_GROUPS),
        key=lambda item: (offsets[item], tie_noise[item]),
    ):
        start = offsets[group]
        while active and active[0][0] < start:
            _, workspace = heapq.heappop(active)
            free.append(workspace)
        if not free:
            return None
        workspace = rng.choice(free)
        free.remove(workspace)
        assignment[group] = workspace
        end = start + (4 if group in kernel.FIRST_CACHE_SET else 3)
        heapq.heappush(active, (end, workspace))
    return tuple(assignment)


def make_candidates() -> list[tuple[tuple[int, ...], tuple[int, ...], int]]:
    target_count = int(os.environ.get("OFFSET_CANDIDATES", "96"))
    max_offset = int(os.environ.get("MAX_OFFSET", "26"))
    seed = int(os.environ.get("OFFSET_SEED", "20260716"))
    rng = random.Random(seed)
    start = tuple(kernel.FULL_ROUND_OFFSETS)
    candidates: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

    def add(offsets: tuple[int, ...], color_seed: int) -> None:
        assignment = color(offsets, seed=color_seed)
        if assignment is None:
            return
        key = (offsets, assignment)
        if key in seen:
            return
        seen.add(key)
        candidates.append((offsets, assignment, color_seed))

    # Recoloring alone changes false dependencies and is essentially free.
    for recolor in range(min(24, target_count)):
        add(start, seed + recolor)

    attempts = 0
    while len(candidates) < target_count and attempts < 100 * target_count:
        attempts += 1
        trial = list(start)
        mutation_count = rng.choices(
            (1, 2, 3, 4, 6, 8, 12),
            weights=(8, 8, 6, 5, 3, 2, 1),
        )[0]
        for _ in range(mutation_count):
            group = rng.randrange(kernel.N_GROUPS)
            radius = rng.choice((1, 2, 3, 4, 6, 8))
            trial[group] = max(
                0,
                min(max_offset, trial[group] + rng.randint(-radius, radius)),
            )
        offsets = normalize(trial)
        if max(offsets) > max_offset:
            continue
        add(offsets, seed + attempts * 17)
    return candidates


def solve_candidate(
    item: tuple[int, tuple[int, ...], tuple[int, ...], int],
) -> dict[str, object]:
    index, offsets, assignment, color_seed = item
    env = dict(os.environ)
    env.update(
        {
            "FULL_ROUND_OFFSETS": ",".join(map(str, offsets)),
            "WORKSPACE_ASSIGNMENT": ",".join(map(str, assignment)),
            "TARGET": os.environ.get("SEARCH_TARGET", "962"),
            "ENGINES": "flow,load",
            "MICROSLOT_ENGINES": "flow,load",
            "TIME_LIMIT": os.environ.get("OFFSET_TIME_LIMIT", "5"),
            "WORKERS": os.environ.get("OFFSET_CP_WORKERS", "1"),
            "OUT_PREFIX": f"/tmp/aopt-offset-{index}",
        }
    )
    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, "-m", "codex.solve_projected_engines"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=float(env["TIME_LIMIT"]) + 20,
        check=False,
    )
    elapsed = time.monotonic() - started
    status = "ERROR"
    for line in completed.stdout.splitlines():
        if line.startswith("status="):
            status = line.split("=", 1)[1]
    return {
        "index": index,
        "status": status,
        "elapsed": elapsed,
        "returncode": completed.returncode,
        "offsets": offsets,
        "assignment": assignment,
        "color_seed": color_seed,
        "tail": completed.stdout.splitlines()[-8:],
    }


def main() -> None:
    configure_target()
    candidates = make_candidates()
    workers = int(os.environ.get("SEARCH_WORKERS", "4"))
    print(
        f"candidates={len(candidates)} workers={workers} "
        f"target={os.environ.get('SEARCH_TARGET', '962')}",
        flush=True,
    )
    results: list[dict[str, object]] = []
    # Each job already launches an isolated Python/CP-SAT subprocess.  A
    # thread pool avoids macOS sandbox semaphore probes from
    # ProcessPoolExecutor while preserving the intended process-level
    # parallelism.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                solve_candidate,
                (index, offsets, assignment, color_seed),
            ): index
            for index, (offsets, assignment, color_seed) in enumerate(candidates)
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result["status"] != "INFEASIBLE":
                print("survivor=" + json.dumps(result), flush=True)
            elif len(results) % 8 == 0:
                print(
                    f"tested={len(results)}/{len(candidates)} "
                    f"infeasible={sum(r['status'] == 'INFEASIBLE' for r in results)}",
                    flush=True,
                )

    rank = {"OPTIMAL": 0, "FEASIBLE": 1, "UNKNOWN": 2, "INFEASIBLE": 3}
    results.sort(
        key=lambda result: (
            rank.get(str(result["status"]), 9),
            float(result["elapsed"]),
            int(result["index"]),
        )
    )
    output = Path(
        os.environ.get("OFFSET_SEARCH_OUT", "/tmp/aopt-offset-search.json")
    )
    output.write_text(json.dumps(results, indent=2))
    summary: dict[str, int] = {}
    for result in results:
        status = str(result["status"])
        summary[status] = summary.get(status, 0) + 1
    print(f"summary={summary} output={output}", flush=True)
    for result in results[:12]:
        print("ranked=" + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
