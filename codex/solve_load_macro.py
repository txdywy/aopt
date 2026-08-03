"""Research solver for the projected LOAD graph with symmetry macros.

Most tree-gather loads occur in groups of eight with identical projected
parents.  Treating their permutations as distinct makes the ordinary CP model
spend almost all of its time rediscovering the same schedule.  This utility
packs such classes into a four-cycle, two-wide macro and uses a compact model
to search for a calendar near the 959-cycle target.  It is deliberately
research-only: every emitted expansion is checked against the original graph.
"""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path

from ortools.sat.python import cp_model


def main() -> None:
    graph_path = Path(os.environ.get("GRAPH", ".search959/d1-graph.json"))
    payload = json.loads(graph_path.read_text())
    ops = payload["ops"]
    op_by_index = {int(op["index"]): op for op in ops}
    capacity = int(payload.get("capacity", 2))
    horizon = int(os.environ.get("TARGET", payload.get("horizon", 959)))

    # Only merge nontrivial classes with exactly matching projected parents.
    # Metadata is included to avoid merging distinct scalar/setup operations
    # that happen to have no parents.
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for op in ops:
        parents = tuple(sorted((int(p), int(l)) for p, l in op["parents"].items()))
        key = (op["tag"], int(op["group"]), int(op["round"]), parents)
        buckets[key].append(int(op["index"]))
    classes: list[list[int]] = []
    for members in buckets.values():
        if len(members) >= 2:
            classes.append(sorted(members))
        else:
            classes.append(members)
    cls_of = {i: c for c, members in enumerate(classes) for i in members}
    durations = [max(1, (len(members) + capacity - 1) // capacity) for members in classes]
    demand = [capacity if len(members) > 1 else 1 for members in classes]

    # Weighted macro edges. For fully symmetric classes, the same-lane mapping
    # means starts only need to differ by the original lag. For an edge into a
    # singleton, conservatively use the last packed parent lane.
    edges: dict[tuple[int, int], int] = {}
    for child in ops:
        c = int(child["index"])
        cc = cls_of[c]
        for parent, lag in child["parents"].items():
            p = int(parent)
            pc = cls_of[p]
            if pc == cc:
                continue
            # A gather child commonly depends on every lane of its parent
            # class (not just the same-lane member), so its first issue must
            # wait for the last packed parent lane.  Using this conservative
            # adjustment also handles singleton consumers safely.
            extra = durations[pc] - 1 if len(classes[pc]) > 1 else 0
            edges[pc, cc] = max(edges.get((pc, cc), 0), int(lag) + extra)

    # Optional cross-resource order constraints.  A projected VALU solve can
    # provide a promising capacity order, but LOAD must choose a compatible
    # order rather than blindly importing its cycle labels.  The strict
    # partial-order format stores the explicit arcs in ``resource_edges``;
    # collapse those arcs onto the symmetric LOAD macros and retain the
    # strongest lag.  This is a research-only bridge and leaves the ordinary
    # macro model unchanged when EXTRA_EDGES is absent.
    extra_path = os.environ.get("EXTRA_EDGES", "")
    if extra_path:
        extra_payload = json.loads(Path(extra_path).read_text())
        extra_edges = extra_payload.get("resource_edges", ())
        added = 0
        for raw_parent, raw_child in extra_edges:
            parent = int(raw_parent)
            child = int(raw_child)
            if parent not in cls_of or child not in cls_of:
                continue
            pc, cc = cls_of[parent], cls_of[child]
            if pc == cc:
                # Operations in one symmetric macro are intentionally packed;
                # an intra-macro order is represented by the expansion below.
                continue
            old = edges.get((pc, cc), 0)
            edges[pc, cc] = max(old, 1)
            if edges[pc, cc] != old:
                added += 1
        print(f"extra_edges={extra_path} added_macro_edges={added}", flush=True)

    model = cp_model.CpModel()
    starts = []
    for c, members in enumerate(classes):
        # Macro domains use the original per-op earliest/latest bounds when
        # available.  Packing a class cannot begin before any member release.
        lo = max(int(op_by_index[i]["earliest"]) for i in members)
        hi = min(
            [horizon - durations[c]]
            + [int(op_by_index[i]["latest"]) - durations[c] + 1 for i in members]
        )
        if hi < lo:
            raise ValueError((c, members, lo, hi))
        starts.append(model.new_int_var(lo, hi, f"m{c}"))
    for (parent, child), lag in edges.items():
        model.add(starts[child] >= starts[parent] + lag)
    intervals = [
        model.new_fixed_size_interval_var(starts[c], durations[c], f"miv{c}")
        for c in range(len(classes))
    ]
    model.add_cumulative(intervals, demand, capacity)
    # Encourage a compact finish, with a tiny tie-break toward preserving the
    # natural class order (which tends to align corresponding lanes).
    end = model.new_int_var(0, horizon, "end")
    for c in range(len(classes)):
        model.add(end >= starts[c] + durations[c])
    model.minimize(end)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(os.environ.get("TIME_LIMIT", "120"))
    solver.parameters.num_workers = int(os.environ.get("WORKERS", "8"))
    solver.parameters.random_seed = int(os.environ.get("RANDOM_SEED", "1"))
    solver.parameters.randomize_search = bool(
        int(os.environ.get("RANDOMIZE", "0"))
    )
    solver.parameters.random_seed = int(os.environ.get("RANDOM_SEED", "1"))
    status = solver.solve(model)
    print(f"classes={len(classes)} edges={len(edges)} status={solver.status_name(status)}")
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return
    macro_starts = [solver.value(x) for x in starts]
    cycles: dict[int, int] = {}
    # Pack each class in stable member order, two operations per cycle.
    for c, members in enumerate(classes):
        for pos, i in enumerate(members):
            cycles[i] = macro_starts[c] + pos // capacity
    usage = defaultdict(int)
    for cycle in cycles.values():
        usage[cycle] += 1
    # Check all original projected constraints and capacity exactly.
    bad = []
    for child in ops:
        c = int(child["index"])
        for parent, lag in child["parents"].items():
            p = int(parent)
            if cycles[c] < cycles[p] + int(lag):
                bad.append((p, c, int(lag), cycles[p], cycles[c]))
    print(f"makespan={max(cycles.values()) + 1} max_usage={max(usage.values())} bad={len(bad)}")
    if bad:
        print("bad_sample", bad[:8])
        return
    out = Path(os.environ.get("OUT", ".search959/macro-load.json"))
    out.write_text(json.dumps({"engine": "load", "horizon": horizon, "cycles": {str(i): c for i, c in cycles.items()}}))
    print(f"output={out}")


if __name__ == "__main__":
    main()
