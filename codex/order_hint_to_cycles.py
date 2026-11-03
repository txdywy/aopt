"""Convert an explicit projected-engine order into a capacity-safe hint.

Several order optimizers intentionally keep the earliest DAG labels in their
``cycles`` field because those labels are useful for Hall diagnostics.  They
are not, however, a legal unary resource calendar: two independent operations
may share a label.  The strict full-DAG solver consumes cycle hints, so this
small bridge assigns consecutive capacity lanes to the explicit ``order``
while preserving the original payload metadata.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    source = Path(os.environ["SOURCE"])
    output = Path(os.environ.get("OUT", ".search959/order-hint-cycles.json"))
    payload = json.loads(source.read_text())
    raw_order = payload.get("order")
    raw_cycles = payload.get("cycles")
    if raw_order is None or not isinstance(raw_cycles, dict):
        raise ValueError("SOURCE must contain an explicit order and cycles map")
    order = [int(index) for index in raw_order]
    cycles = {int(index): int(cycle) for index, cycle in raw_cycles.items()}
    if len(order) != len(cycles) or set(order) != set(cycles):
        raise ValueError("order and cycles must cover exactly the same operations")
    capacity = int(os.environ.get("CAPACITY", "1"))
    if capacity < 1:
        raise ValueError("CAPACITY must be positive")
    mapped = {
        str(index): position // capacity
        for position, index in enumerate(order)
    }
    result = dict(payload)
    result["cycles"] = mapped
    result["source_cycles"] = {str(index): cycle for index, cycle in cycles.items()}
    result["capacity"] = capacity
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result))
    print(
        f"source={source} operations={len(order)} capacity={capacity} "
        f"span={max(mapped.values(), default=-1) + 1} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
