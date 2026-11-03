"""Local search over a jointly feasible projected VALU/FLOW calendar.

This is a small research harness for the near-960 candidate.  It never
changes the kernel; it only proposes swaps/moves in a pair schedule and
checks the *weighted projection* of the complete DAG.  A schedule accepted by
this tool therefore remains a valid schedule for every path whose endpoints
are VALU/FLOW operations, including paths through omitted engines.
"""

from __future__ import annotations

from collections import Counter
import heapq
import json
import os
from pathlib import Path

import codex.perf_takehome_under1000 as kernel
from codex.map_variant_schedule import configure_target, real_tail_ops
from problem import SLOT_LIMITS


def build_projection(ops, selected_set):
    children = [[] for _ in ops]
    indegree = [0] * len(ops)
    for child, op in enumerate(ops):
        indegree[child] = len(op.parents)
        for parent, lag in op.parents.items():
            children[parent].append((child, lag))
    ready = [i for i, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)
    topo = []
    while ready:
        parent = heapq.heappop(ready)
        topo.append(parent)
        for child, _ in children[parent]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)
    if len(topo) != len(ops):
        raise ValueError("kernel DAG is cyclic")
    earliest = [0] * len(ops)
    for child in topo:
        for parent, lag in ops[child].parents.items():
            earliest[child] = max(earliest[child], earliest[parent] + lag)
    tail = [0] * len(ops)
    for parent in reversed(topo):
        tail[parent] = max(
            (lag + tail[child] for child, lag in children[parent]),
            default=0,
        )
    position = {index: rank for rank, index in enumerate(topo)}
    frontier = [{} for _ in ops]
    projected = {}
    for child in topo:
        incoming = {}
        for parent, lag in ops[child].parents.items():
            sources = ({parent: 0} if parent in selected_set else frontier[parent])
            for source, distance in sources.items():
                incoming[source] = max(incoming.get(source, -1), distance + lag)
        if child in selected_set:
            projected[child] = incoming
            frontier[child] = {child: 0}
        else:
            frontier[child] = incoming

    # A full weighted transitive reduction is compact and exact for this
    # strict pair model: every retained edge is a hard precedence edge.
    reduced = {}
    ancestors = {}
    for child in topo:
        if child not in selected_set:
            continue
        kept = {}
        implied = {}
        for parent, lag in sorted(
            projected[child].items(), key=lambda item: position[item[0]], reverse=True
        ):
            if implied.get(parent, -1) >= lag:
                continue
            kept[parent] = lag
            implied[parent] = max(implied.get(parent, -1), lag)
            for ancestor, distance in ancestors[parent].items():
                implied[ancestor] = max(implied.get(ancestor, -1), distance + lag)
        reduced[child] = kept
        ancestors[child] = implied
    return reduced, tail, earliest


def validate(cycles, ops, selected, projected, horizon):
    usage = Counter((cycles[i], ops[i].engine) for i in selected)
    if any(demand > SLOT_LIMITS[engine] for (_, engine), demand in usage.items()):
        return False
    bad = []
    for child in selected:
        for parent, lag in projected[child].items():
            if cycles[child] < cycles[parent] + lag:
                bad.append((cycles[parent] + lag - cycles[child], parent, child, lag))
    if bad:
        if os.environ.get("DEBUG", "0") == "1":
            print("violations", sorted(bad, reverse=True)[:20], flush=True)
        return False
    # The reported score includes tails of paths through omitted engines.
    score = max(cycles[i] + horizon_tail[i] for i in selected) + 1
    return score


def load(path):
    payload = json.loads(Path(path).read_text())
    return payload, {int(i): int(c) for i, c in payload["cycles"].items()}


def dump(prefix, payloads, cycles_by_engine, score):
    for engine, payload in payloads.items():
        result = dict(payload)
        result["horizon"] = score
        result["cycles"] = {
            str(index): cycle for index, cycle in cycles_by_engine[engine].items()
        }
        Path(f"{prefix}-{engine}.json").write_text(json.dumps(result))


def main():
    configure_target()
    kernel.SCHEDULE_EXACT_CYCLES = None
    builder = kernel.KernelBuilder()
    try:
        builder.build_kernel(10, 2047, 256, 16)
    except (AssertionError, StopIteration):
        if not hasattr(builder, "dag_ops"):
            raise
    ops = real_tail_ops(builder.dag_ops)
    payloads = {}
    cycles_by_engine = {}
    for engine, raw_path in (
        ("valu", os.environ["VALU"]),
        ("flow", os.environ["FLOW"]),
    ):
        payload, cycles = load(raw_path)
        payloads[engine] = payload
        cycles_by_engine[engine] = cycles
    cycles = {}
    for values in cycles_by_engine.values():
        if set(cycles) & set(values):
            raise ValueError("overlapping engine hints")
        cycles.update(values)
    selected = set(cycles)
    if any(ops[i].engine not in cycles_by_engine for i in selected):
        raise ValueError("input contains an unexpected engine")
    projected, horizon_tail, earliest = build_projection(ops, selected)
    globals()["horizon_tail"] = horizon_tail
    before = validate(cycles, ops, selected, projected, 959)
    if not before:
        raise ValueError("input pair schedule is not projected-feasible")

    # Explicit moves are useful for deterministic experiments.  A move is
    # encoded as old-id:new-cycle.  The optional search below explores every
    # single relocation and every two-operation cycle swap around the tail.
    raw_moves = os.environ.get("MOVES", "")
    trial = dict(cycles)
    for item in raw_moves.split(","):
        if not item:
            continue
        index, cycle = (int(value) for value in item.split(":", 1))
        trial[index] = cycle
    after = validate(trial, ops, selected, projected, 959)
    print(f"input_score={before} moved_score={after or 'INVALID'}")
    if after:
        prefix = Path(os.environ.get("OUT_PREFIX", "/tmp/aopt-pair-local"))
        by_engine = {
            engine: {
                index: trial[index]
                for index in indices
            }
            for engine, indices in (
                ("valu", [i for i in selected if ops[i].engine == "valu"]),
                ("flow", [i for i in selected if ops[i].engine == "flow"]),
            )
        }
        dump(prefix, payloads, by_engine, after)

    if os.environ.get("SEARCH", "0") != "1":
        return
    # Search only the final 12 cycles and the early/empty slots.  The pair is
    # almost fully saturated, so any useful improvement must eject one of the
    # few tail jobs into an actual hole and refill its old slot.
    current = dict(cycles)
    best_score = before
    best = dict(current)
    limit = int(os.environ.get("SEARCH_TAIL", "12"))
    tail_cycles = sorted({current[i] for i in selected if current[i] >= 959 - limit})
    all_cycles = sorted({*range(0, 16), *tail_cycles})
    candidates = [i for i in selected if current[i] >= 959 - limit]
    for first in candidates:
        old_cycle = current[first]
        engine = ops[first].engine
        counts = Counter(current[i] for i in selected if ops[i].engine == engine)
        for target in all_cycles:
            if target == old_cycle or counts[target] >= SLOT_LIMITS[engine]:
                continue
            trial = dict(current)
            trial[first] = target
            score = validate(trial, ops, selected, projected, 959)
            if score and score < best_score:
                best_score, best = score, trial
                print(f"relocation score={score} id={first} {old_cycle}->{target}", flush=True)
    # Also try pairwise swaps of tail jobs on the same engine.  This catches
    # free-slot chains where one move alone violates an immediate edge.
    for left_pos, first in enumerate(candidates):
        for second in candidates[left_pos + 1:]:
            if ops[first].engine != ops[second].engine:
                continue
            if current[first] == current[second]:
                continue
            trial = dict(current)
            trial[first], trial[second] = current[second], current[first]
            score = validate(trial, ops, selected, projected, 959)
            if score and score < best_score:
                best_score, best = score, trial
                print(f"swap score={score} ids={first},{second}", flush=True)
    if best_score < before:
        prefix = Path(os.environ.get("SEARCH_OUT", "/tmp/aopt-pair-search"))
        by_engine = {
            engine: {i: best[i] for i in selected if ops[i].engine == engine}
            for engine in ("valu", "flow")
        }
        dump(prefix, payloads, by_engine, best_score)
    print(f"search_best={best_score}", flush=True)


if __name__ == "__main__":
    main()
