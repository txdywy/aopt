"""Enumerate VALU parity exchanges around the H959 mix4v8c0 graph."""

from __future__ import annotations

import gc
import os

from codex.search_h959_mix import evaluate


DEFAULT = {(31, 11), (31, 12), (31, 13), (31, 14), (16, 14), (21, 13), (21, 14)}


def encode(values):
    return ",".join(f"{g}:{r}" for g, r in sorted(values))


def main() -> None:
    best = None
    candidates = []
    # Single additions first: the exact A/L core says any useful change must
    # touch a previously scalar mirror cell.  Then test pairs around the best
    # one because the first cell can alter the V/F phase cut.
    for g in range(32):
        for r in range(4):
            if (g, r) not in DEFAULT:
                candidates.append(DEFAULT | {(g, r)})
    for index, values in enumerate(candidates):
        result = evaluate({"VECTOR_PARITY_SET": encode(values)})
        gc.collect()
        if result is None:
            continue
        key, payload = result
        record = (key, values, payload)
        if best is None or key < best[0]:
            best = record
            print(
                f"single={index} key={key} halls={payload['halls']} "
                f"counts={payload['counts']} parity={encode(values)}",
                flush=True,
            )
    print(f"best_single={best and (best[0], encode(best[1]))}", flush=True)


if __name__ == "__main__":
    main()
