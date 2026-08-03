"""Fast Hall-window search for the 959-cycle mixed lowering space.

This is a structural filter, not a scheduler: every candidate is rebuilt from
the real semantic DAG, and its exact earliest/latest Hall overload is computed
for all physical engines.  Candidates with non-positive Hall cuts are then
small enough to hand to ``solve_full_dag`` instead of spending CP-SAT time on
obviously impossible resource mixes.

Run after sourcing a structural base environment, for example::

    source codex/sub959_candidate_projected964.env
    python3 -m codex.search_h959_mix
"""

from __future__ import annotations

from collections import Counter
import importlib
import gc
import os
import random
import subprocess
import sys

import numpy as np

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from problem import SLOT_LIMITS


HORIZON = int(os.environ.get("TARGET", "959"))
ALL_GROUPS = tuple(range(kernel.N_GROUPS))
ELIGIBLE_SCALAR = (2, 8, 13, 15, 19, 22, 24, 26, 30, 31)
ALL_DYN = ",".join(
    f"{group}:{round_}"
    for group in range(7)
    for round_ in range(4, 16)
)
ALL_RESTORE = ",".join(map(str, ALL_GROUPS))


def hall_for_engine(ops, earliest, latest, engine: str) -> tuple[int, int, int]:
    matrix = np.zeros((HORIZON, HORIZON), dtype=np.int32)
    for index, op in enumerate(ops):
        if op.engine != engine:
            continue
        left, right = earliest[index], latest[index]
        if 0 <= left <= right < HORIZON:
            matrix[left, right] += 1
    contained = matrix[::-1].cumsum(axis=0)[::-1].cumsum(axis=1)
    capacity = SLOT_LIMITS[engine]
    best = (-10**9, 0, 0)
    for left in range(HORIZON):
        lengths = np.arange(1, HORIZON - left + 1, dtype=np.int32)
        overloads = contained[left, left:] - capacity * lengths
        offset = int(np.argmax(overloads))
        candidate = (int(overloads[offset]), left, left + offset)
        if candidate[0] > best[0]:
            best = candidate
    return best


def evaluate(settings: dict[str, str]) -> tuple[tuple, dict] | None:
    for name, value in settings.items():
        os.environ[name] = value
    # Reinitialize all module-level lowering defaults before applying the
    # candidate environment.  map_variant_schedule holds the same module
    # object, so its configure_target() sees the reloaded definitions.
    importlib.reload(kernel)
    configure_target()
    kernel.SCHEDULE_EXACT_CYCLES = None
    builder = kernel.KernelBuilder()
    try:
        builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration, ValueError):
        if not hasattr(builder, "dag_ops"):
            return None
    ops = real_tail_ops(builder.dag_ops)
    children = [[] for _ in ops]
    indegree = [len(op.parents) for op in ops]
    for child, op in enumerate(ops):
        for parent, lag in op.parents.items():
            children[parent].append((child, lag))
            indegree[child] -= 0
    ready = [i for i, degree in enumerate(indegree) if degree == 0]
    topo = []
    while ready:
        parent = ready.pop()
        topo.append(parent)
        for child, _ in children[parent]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(topo) != len(ops):
        return None
    earliest = [0] * len(ops)
    for child in topo:
        earliest[child] = max(
            (earliest[parent] + lag for parent, lag in ops[child].parents.items()),
            default=0,
        )
    tail = [0] * len(ops)
    for parent in reversed(topo):
        tail[parent] = max(
            (lag + tail[child] for child, lag in children[parent]),
            default=0,
        )
    latest = [HORIZON - 1 - value for value in tail]
    halls = {
        engine: hall_for_engine(ops, earliest, latest, engine)
        for engine in ("alu", "valu", "load", "flow")
    }
    counts = Counter(op.engine for op in ops)
    maximum = max(value[0] for value in halls.values())
    positive = sum(max(0, value[0]) for value in halls.values())
    key = (maximum, positive, sum(counts[e] for e in ("alu", "valu", "load", "flow")))
    return key, {"settings": settings.copy(), "halls": halls, "counts": dict(counts)}


def main() -> None:
    trials = int(os.environ.get("MIX_TRIALS", "200"))
    rng = random.Random(int(os.environ.get("MIX_SEED", "959")))
    best = (10**9, 10**9, 10**9)
    best_payload = None
    strict_checked = 0
    strict_limit = int(os.environ.get("STRICT_CHECK_LIMIT", "0"))
    strict_time = float(os.environ.get("STRICT_CHECK_TIME", "3"))
    for trial in range(trials):
        # Keep the high-value phase cuts fixed and vary only the resource
        # exchange knobs.  Prefixes are used because each vector cell has the
        # same semantic shape but different late-tail pressure.
        node_count = rng.randrange(8, 29)
        node_set = ",".join(f"{group}:15" for group in range(node_count))
        scalar_count = rng.randrange(0, 6)
        scalar_groups = sorted(rng.sample(ELIGIBLE_SCALAR, scalar_count))
        n = rng.randrange(150, 181)
        # Start from the candidate's seven MADD-heavy groups and remove a
        # random number of pairs to create Flow slack where a scalar lowering
        # needs it.  Limits are only meaningful for round 4.
        limits = {3: 8, 4: 8, 5: 8, 6: 8, 7: 8, 9: 8, 10: 8}
        for group in rng.sample(tuple(limits), rng.randrange(0, 5)):
            limits[group] = rng.randrange(0, 9)
        overrides = ["1:4:0"] + [f"{group}:4:{limits[group]}" for group in sorted(limits)]
        gather_group = rng.choice((None, 3, 5, 6, 8, 10, 11, 13, 15))
        gather = "1:4" if gather_group is None else f"1:4,{gather_group}:4"
        settings = {
            "VECTOR_RESTORE_LAYOUT_SET": ALL_RESTORE,
            "VECTOR_DYNAMIC_XOR_SET": ALL_DYN,
            "VECTOR_NODE_XOR_SET": node_set,
            "HASH_SCALAR_EXTRA_COUNT": str(n),
            "HYBRID_MADD_OVERRIDES": ",".join(overrides),
            "SCALAR_SECOND_PATH_GROUPS": ",".join(map(str, scalar_groups)),
            "GATHER_LEVEL4": gather,
        }
        # A complete final-group direct lookup is the only lowering found so
        # far that removes the six-load latest=H-11 trap without spending
        # Flow capacity.  Keep it enabled for the H959 balancing search; the
        # remaining problem is then the much smaller ALU/VALU exchange.
        if os.environ.get("MIX_DIRECT", "22:8"):
            settings["DIRECT_BRANCH_LOOKUPS"] = os.environ.get(
                "MIX_DIRECT", "22:8"
            )
        result = evaluate(settings)
        # Each trial rebuilds a ~20k-node semantic DAG and a dense Hall
        # matrix.  CPython normally delays collection of the temporary
        # object graph long enough for long random searches to be killed by
        # the OOM reaper; collect between trials so the search is genuinely
        # reproducible rather than silently stopping after a dozen samples.
        gc.collect()
        if result is None:
            continue
        key, payload = result
        if (
            strict_limit
            and key[0] <= 0
            and strict_checked < strict_limit
        ):
            strict_checked += 1
            strict_env = os.environ.copy()
            strict_env.update(payload["settings"])
            strict_env.update(
                {
                    "TARGET": str(HORIZON),
                    "TIME_LIMIT": str(strict_time),
                    "WORKERS": os.environ.get("STRICT_WORKERS", "1"),
                    "REPAIR_HINT": "0",
                    "BASELINE_HINT": "0",
                    "MICROSLOT_ENGINES": "alu,valu,load,flow",
                    "SATURATE_MICROSLOTS": "1",
                }
            )
            try:
                completed = subprocess.run(
                    [sys.executable, "-m", "codex.solve_full_dag"],
                    env=strict_env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=max(5.0, strict_time + 5.0),
                    check=False,
                )
                status_line = next(
                    (
                        line.strip()
                        for line in completed.stdout.splitlines()
                        if line.startswith("status=")
                    ),
                    "status=NO_RESULT",
                )
            except subprocess.TimeoutExpired:
                status_line = "status=TIMEOUT"
            print(
                f"strict_trial={strict_checked} key={key} {status_line} "
                f"settings={settings}",
                flush=True,
            )
            if status_line.startswith(("status=FEASIBLE", "status=OPTIMAL")):
                print("STRICT_FEASIBLE", flush=True)
                break
        if key < best:
            best = key
            best_payload = payload
            print(
                f"trial={trial} best={best} halls={payload['halls']} "
                f"counts={payload['counts']} settings={settings}",
                flush=True,
            )
            if best[0] <= 0:
                print("HALL_FEASIBLE", flush=True)
                break
    print(f"final_best={best} payload={best_payload}", flush=True)


if __name__ == "__main__":
    main()
