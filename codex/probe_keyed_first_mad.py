"""Screen a first-hash rewrite that can reuse the mandatory node XOR.

This is an algebra experiment, not a kernel or a whole-hash lower bound.
An input XOR key S would be free at an existing node-XOR site if it could
remove C1 after the first xorshift. All unknown coefficients are enumerated
bit by bit; finite samples supply necessary conditions for equivalence.
"""

from __future__ import annotations

import json
import random


A0 = 4097
B0 = 0x7ED55D16
C1 = 0xC761C23C
MASK = (1 << 32) - 1


def search(output_key: int, width: int = 32, frontier_limit: int = 200_000):
    """Seek A*(x XOR S)+B == (A0*x+B0) XOR output_key modulo 2**width.

    At x=0, B is determined by A and S. Complementing S, negating A,
    and replacing B by B-A preserves the function, so S's low bit can
    be fixed to zero without losing any represented function.
    """
    rng = random.Random(900)
    xs = list(range(256)) + [rng.getrandbits(32) for _ in range(128)]
    xs.extend((MASK, 1 << 31))
    ys = [((A0 * x + B0) & MASK) ^ output_key for x in xs]
    frontier = [(0, 0)]
    counts = []
    for bit in range(width):
        flag = 1 << bit
        mask = (flag << 1) - 1
        following = []
        for old_a, old_s in frontier:
            for a_bit in (0, flag):
                for s_bit in ((0,) if bit == 0 else (0, flag)):
                    a, s = old_a | a_bit, old_s | s_bit
                    b = (ys[0] - a * s) & mask
                    if all(((a * (x ^ s) + b) ^ y) & mask == 0
                           for x, y in zip(xs, ys)):
                        following.append((a, s))
                        if len(following) > frontier_limit:
                            return {"status": "frontier_limit", "bits": bit + 1,
                                    "counts": counts, "candidates": []}
        frontier = following
        counts.append(len(frontier))
        if not frontier:
            return {"status": "excluded_family", "bits": bit + 1,
                    "counts": counts, "samples": len(xs), "candidates": []}
    modulus_mask = (1 << width) - 1
    return {
        "status": "sample_survivors_require_equivalence_check",
        "bits": width,
        "counts": counts,
        "samples": len(xs),
        "candidates": [
            {"a": a, "s": s, "b": (ys[0] - a * s) & modulus_mask}
            for a, s in frontier
        ],
    }


def main():
    # The inverse of x XOR (x >> 19) is itself at width 32.
    # Hence removing C1 from that stage asks the MAD to emit a XOR K.
    key = C1 ^ (C1 >> 19)
    target = search(key)
    assert target["status"] == "excluded_family" and target["bits"] == 6

    # Independent exhaustive check of the entire six-bit constant family.
    # B is fixed by x=0; both A and S are otherwise unrestricted.
    mask = 63
    survivors = []
    for a in range(64):
        for s in range(64):
            b = ((B0 ^ key) - a * s) & mask
            if all((a * (x ^ s) + b) & mask
                   == ((A0 * x + B0) ^ key) & mask for x in range(64)):
                survivors.append((a, s, b))
    assert not survivors

    # Positive controls survive full-width lifting. Their known candidates
    # are exact identities, independently of the finite sample screen.
    controls = {}
    for control_key in (0, 1 << 31):
        result = search(control_key)
        known = {"a": A0, "s": 0, "b": B0 ^ control_key}
        assert known in result["candidates"]
        controls[hex(control_key)] = {
            "status": "known_identity_retained", "candidates": len(result["candidates"])
        }
    print(json.dumps({
        "rewrite": "A*(x XOR S)+B == (4097*x+C0) XOR K",
        "K": hex(key), "target": target, "positive_controls": controls,
        "exhaustive_low6_pairs": 64 * 64,
        "scope": "Only this input-key/affine-MAD family is excluded; no global optimality claim.",
    }, indent=2))


if __name__ == "__main__":
    main()
