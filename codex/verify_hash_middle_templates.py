"""Independent low-11-bit verification of the C++ template exclusions."""

import json

MASK = (1 << 11) - 1


def target(x):
    u = (33*x + 0xE9F8CC1D) & MASK
    v = (16896*x + 0xACCF6200) & MASK
    return (9*(u ^ v) + 0xFD7046C5) & MASK


def verify(mode):
    outputs = [target(x) for x in range(MASK + 1)]
    states = [(0, 0, 0)]
    counts = []
    for bit in range(11):
        flag = 1 << bit
        next_states = []
        for a, b, d in states:
            for choice in range(8):
                if bit == 0 and choice & 2:
                    continue
                aa = a | (flag if choice & 1 else 0)
                bb = b | (flag if choice & 2 else 0)
                dd = d | (flag if choice & 4 else 0)
                if mode == "xor":
                    f = outputs[0] ^ bb
                    evaluate = lambda x: (aa*x+bb) ^ (dd*x+f)
                elif mode == "nested":
                    f = outputs[0] - dd*bb
                    evaluate = lambda x: dd*((aa*x+bb) ^ x)+f
                else:
                    f = outputs[0] ^ (aa*bb+dd)
                    evaluate = lambda x: (aa*(x ^ bb)+dd) ^ f
                if all(((evaluate(x) ^ y) & flag) == 0 for x, y in enumerate(outputs)):
                    next_states.append((aa, bb, dd))
        states = next_states
        counts.append(len(states))
        if not states:
            break
    assert not states, (mode, counts)
    return {"mode": mode, "frontier_counts": counts, "inputs": len(outputs)}


def verify_quadratic():
    c = target(0)
    delta = target(1) - c
    rhs = (target(2) - c - 2*delta) & MASK
    rejected = []
    for a in range(MASK + 1):
        if (2*a) & MASK != rhs:
            continue
        b = (delta-a) & MASK
        witness = next((x for x in range(MASK + 1)
                        if (a*x*x+b*x+c) & MASK != target(x)), None)
        assert witness is not None, (a, b, c)
        rejected.append({"a": a, "b": b, "c": c, "counterexample": witness})
    return {"mode": "quadratic", "rejected_candidates": rejected}


if __name__ == "__main__":
    print(json.dumps([verify(mode) for mode in ("xor", "nested", "input-xor")] + [verify_quadratic()]))
