"""Screen two-instruction replacements for the three-op hash XOR/shift slice.

This is a finite-grammar search, not a proof that the hash is globally optimal.
Survivors need complete bit-vector equivalence verification before kernel use.
Only operations available on the challenge ISA are represented.
"""

import argparse
import json
import random

MASK = (1 << 32) - 1
C = 0xC761C23C
CONSTANTS = sorted({0, 1, MASK, C, C ^ (C >> 19),
                    0x7ED55D16, 0x165667B1, 0xD3A2646C,
                    0xFD7046C5, 0xB55A4F09,
                    *range(2, 34), *(1 << n for n in range(32))})
MULTIPLIERS = sorted({0, 1, 3, 9, 33, 4097, 16896, MASK,
                      *(1 << n for n in range(32)),
                      *((-(1 << n)) & MASK for n in range(32))})


def apply(op, a, b):
    if op == "+":
        return (a + b) & MASK
    if op == "-":
        return (a - b) & MASK
    if op == "^":
        return a ^ b
    if op == "&":
        return a & b
    if op == "|":
        return a | b
    if op == "*":
        return (a * b) & MASK
    if op == ">>":
        return a >> b if b < 32 else 0
    if op == "<<":
        return (a << b) & MASK if b < 32 else 0
    raise ValueError(op)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xor-constant", type=lambda value: int(value, 0), default=C)
    args = parser.parse_args()
    if not 0 <= args.xor_constant <= MASK:
        parser.error("xor constant must be uint32")
    rng = random.Random(0x900)
    xs = [0, 1, 16, MASK, 1 << 19, (1 << 19) - 1]
    xs += [rng.getrandbits(32) for _ in range(128)]
    ys = [(x ^ (x >> 19)) ^ args.xor_constant for x in xs]
    first = []
    for op in ("+", "-", "^", "&", "|", "*", ">>", "<<"):
        for k in (range(32) if op in (">>", "<<") else CONSTANTS):
            first.append((f"x {op} {k:#x}", lambda x, op=op, k=k: apply(op, x, k)))
    for m in MULTIPLIERS:
        for k in CONSTANTS:
            first.append((f"MAD(x,{m:#x},{k:#x})", lambda x, m=m, k=k: (x*m+k) & MASK))

    survivors = []
    tested = 0
    for name, f in first:
        ts = [f(x) for x in xs]
        # Infer arbitrary last-op constants from x=0 where invertible, rather
        # than limiting the second XOR/add/sub to the seed constant pool.
        for op, k in (("^", ts[0] ^ ys[0]),
                      ("+", (ys[0] - ts[0]) & MASK),
                      ("-", (ts[0] - ys[0]) & MASK)):
            tested += 1
            if all(apply(op, t, k) == y for t, y in zip(ts, ys)):
                survivors.append([name, f"t {op} {k:#x}"])
        # A final MAD can use every multiplier in the stated grammar and an
        # arbitrary inferred addend. Constants are uint32 throughout.
        for m in MULTIPLIERS:
            k = (ys[0] - ts[0]*m) & MASK
            tested += 1
            if all((t*m+k) & MASK == y for t, y in zip(ts, ys)):
                survivors.append([name, f"MAD(t,{m:#x},{k:#x})"])
        # Binary forms keep both the original input and first result live.
        for op in ("+", "-", "^", "&", "|", "*", ">>", "<<"):
            for reverse in (False, True):
                tested += 1
                if all(apply(op, x if reverse else t, t if reverse else x) == y
                       for t, x, y in zip(ts, xs, ys)):
                    survivors.append([name, f"{'x' if reverse else 't'} {op} {'t' if reverse else 'x'}"])
        for op in (">>", "<<", "&", "|"):
            for k in (range(32) if op in (">>", "<<") else CONSTANTS):
                tested += 1
                if all(apply(op, t, k) == y for t, y in zip(ts, ys)):
                    survivors.append([name, f"t {op} {k:#x}"])
    print(json.dumps({"slice": f"(x ^ (x >> 19)) ^ {args.xor_constant:#x}",
                      "samples": len(xs), "first_instructions": len(first),
                      "candidates": tested, "survivors": survivors,
                      "claim": "finite grammar screening only; not global optimality"}))


if __name__ == "__main__":
    main()
