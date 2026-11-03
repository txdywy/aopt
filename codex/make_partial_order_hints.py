"""Convert native cycle calendars to auditable resource-order hint files.

Projected solvers emit cycles but intentionally do not harden those cycles
into the full DAG.  This utility adds only the capacity-k order arcs requested
by the caller, with optional tail/stride thinning, so the strict partial solver
can audit and solve a controlled subset of an engine teacher calendar.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from problem import SLOT_LIMITS


def main() -> None:
    paths = [Path(value) for value in os.environ.get("HINTS", "").split(",") if value]
    if not paths:
        raise ValueError("HINTS is required")
    stride = max(1, int(os.environ.get("EDGE_STRIDE", "1")))
    max_edges = int(os.environ.get("MAX_EDGES", "0"))
    tail = int(os.environ.get("TAIL_CYCLES", "-1"))
    prefix = Path(os.environ.get("OUT_PREFIX", "/tmp/aopt-partial"))
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for path in paths:
        payload = json.loads(path.read_text())
        engine = str(payload["engine"])
        if engine not in SLOT_LIMITS:
            raise ValueError(f"unknown engine {engine}")
        cycles = {int(index): int(cycle) for index, cycle in payload["cycles"].items()}
        order = sorted(cycles, key=lambda index: (cycles[index], index))
        capacity = SLOT_LIMITS[engine]
        edges = list(zip(order, order[capacity:]))
        if tail >= 0:
            cutoff = max(cycles.values(), default=0) - tail
            edges = [edge for edge in edges if cycles[edge[1]] >= cutoff]
        edges = [edge for rank, edge in enumerate(edges) if rank % stride == 0]
        if max_edges > 0:
            edges = edges[:max_edges]
        out = Path(f"{prefix}-{engine}.json")
        out.write_text(json.dumps({
            "engine": engine,
            "horizon": payload.get("horizon", max(cycles.values(), default=0) + 1),
            # Keep the source calendar alongside the explicit arcs.  The
            # projected solver uses it as a soft teacher even when the
            # partial-order file is consumed only for its retained edges.
            "cycles": {str(index): cycle for index, cycle in cycles.items()},
            "matched_indices": order,
            "resource_edges": [[a, b] for a, b in edges],
            "source": str(path),
        }))
        print(f"engine={engine} jobs={len(order)} edges={len(edges)} output={out}", flush=True)


if __name__ == "__main__":
    main()
