"""Bounded bit-vector CEGIS search with the challenge's integer MAD cost.

Install z3-solver in an isolated environment. Unknown/timeouts are never
reported as impossibility. Only an UNSAT counterexample query proves a
candidate equivalent for all uint32 inputs; no kernel is changed here.
"""

import argparse
import json
import random
import time
import z3

OPS = ("+", "-", "^", "&", "|", "<<", ">>", "*", "mad")


def choose(index, values):
    result = values[-1]
    for i in reversed(range(len(values)-1)):
        result = z3.If(index == i, values[i], result)
    return result


def apply(op, a, b, c):
    return choose(op, [a+b, a-b, a^b, a&b, a|b, a<<b, z3.LShR(a, b), a*b, a*b+c])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instructions", type=int, default=2)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--check-ms", type=int, default=15000)
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--canonical-operands", action="store_true",
                        help="Remove commutative operand permutations and unused third operands")
    parser.add_argument("--right-shift-amount", type=int,
                        help="Restrict logical right shifts to this literal (narrower grammar)")
    parser.add_argument("--extra-samples", type=int, default=0,
                        help="Deterministic uint32 samples added before synthesis")
    parser.add_argument("--high-bit-pairs", action="store_true",
                        help="Add paired inputs differing only in the highest bit")
    parser.add_argument("--slice", choices=("xor-shift", "middle", "first"), default="xor-shift")
    parser.add_argument("--width", type=int, default=32,
                        help="Low-bit screening for middle without shifts; UNSAT excludes full-width programs in this grammar")
    parser.add_argument("--op-sequence", help="Comma-separated fixed opcodes, e.g. mad,mad,^,mad")
    args = parser.parse_args()
    if not 1 <= args.instructions <= 4:
        parser.error("instruction count must be 1..4")
    if args.seconds <= 0 or args.check_ms <= 0:
        parser.error("time limits must be positive")
    if args.extra_samples < 0:
        parser.error("extra-samples must be nonnegative")
    if args.right_shift_amount is not None and not 0 <= args.right_shift_amount <= 32:
        parser.error("right-shift-amount must be in 0..32")
    if not 1 <= args.width <= 32:
        parser.error("width must be 1..32")
    if args.width != 32 and args.slice != "middle":
        parser.error("low-bit screening is supported only for middle")
    sequence = args.op_sequence.split(",") if args.op_sequence else None
    if sequence and (len(sequence) != args.instructions or any(op not in OPS for op in sequence)):
        parser.error("op-sequence must contain one valid opcode per instruction")
    if args.width != 32 and sequence and any(op in ("<<", ">>") for op in sequence):
        parser.error("low-bit screening excludes shifts: narrowing can change their semantics")
    solver = z3.SolverFor("QF_BV")
    spec = []
    for i in range(args.instructions):
        op, a, b, c = z3.BitVecs(f"op{i} a{i} b{i} c{i}", 4)
        k0, k1 = z3.BitVecs(f"k{i}_0 k{i}_1", args.width)
        solver.add(z3.ULT(op, len(OPS)))
        if args.width != 32:
            solver.add(op != OPS.index("<<"), op != OPS.index(">>"))
        if sequence:
            solver.add(op == OPS.index(sequence[i]))
        for operand in (a, b, c):
            solver.add(z3.ULT(operand, i+3))
        if args.canonical_operands:
            commutative = z3.Or(*(op == OPS.index(name) for name in ('+', '*', '^', '&', '|', 'mad')))
            solver.add(z3.Implies(commutative, z3.ULE(a, b)))
            solver.add(z3.Implies(op != OPS.index('mad'), c == 0))
        if args.right_shift_amount is not None:
            solver.add(z3.Implies(op == OPS.index('>>'),
                                 z3.And(b == i+2, k1 == args.right_shift_amount)))
        spec.append((op, a, b, c, k0, k1))

    def program(x, model=None):
        values = [x]
        for instruction_index, row in enumerate(spec):
            op, a, b, c, k0, k1 = [model.eval(v, model_completion=True) if model is not None else v for v in row]
            if sequence is not None:
                # Specialize before building sample formulas instead of
                # retaining all nine opcode branches behind a fixed selector.
                op = z3.BitVecVal(OPS.index(sequence[instruction_index]), 4)
            operands = values + [k0, k1]
            values.append(apply(op, choose(a, operands), choose(b, operands), choose(c, operands)))
        return z3.simplify(values[-1])

    def reference(x):
        if args.slice == "middle":
            return 9*((33*x+0xE9F8CC1D) ^ (16896*x+0xACCF6200))+0xFD7046C5
        if args.slice == "first":
            a = 4097*x + 0x7ED55D16
            return a ^ z3.LShR(a, 19) ^ (0 if args.control else 0xC761C23C)
        return x ^ z3.LShR(x, 19) ^ (0 if args.control else 0xC761C23C)

    samples = list(dict.fromkeys(value & ((1 << args.width)-1) for value in
                               [0, 1, 16, 0xFFFFFFFF, 1 << 19, (1 << 19)-1]))
    rng = random.Random(0)
    samples = list(dict.fromkeys(samples + [rng.getrandbits(args.width)
                                          for _ in range(args.extra_samples)]))
    if args.high_bit_pairs:
        samples = list(dict.fromkeys(samples + [value ^ (1 << (args.width-1))
                                               for value in samples]))
    for value in samples:
        x = z3.BitVecVal(value, args.width)
        solver.add(program(x) == reference(x))
    deadline = time.monotonic() + args.seconds
    iteration = 0
    while time.monotonic() < deadline:
        iteration += 1
        solver.set(timeout=min(args.check_ms, max(1, int((deadline-time.monotonic())*1000))))
        status = solver.check()
        print(json.dumps({"iteration": iteration, "samples": len(samples), "synthesis": str(status)}), flush=True)
        if status != z3.sat:
            print(json.dumps({"result": "grammar_unsatisfiable" if status == z3.unsat else "inconclusive",
                              "width": args.width, "op_sequence": sequence,
                              "right_shift_amount": args.right_shift_amount,
                              "reason": solver.reason_unknown() if status == z3.unknown else None}))
            return
        model = solver.model()
        x = z3.BitVec("counterexample_x", args.width)
        candidate = program(x, model)
        verify = z3.SolverFor("QF_BV")
        verify.set(timeout=min(args.check_ms, max(1, int((deadline-time.monotonic())*1000))))
        verify.add(candidate != reference(x))
        status = verify.check()
        if status == z3.unsat:
            decoded = []
            for row in spec:
                op, a, b, c, k0, k1 = [model.eval(v, model_completion=True).as_long() for v in row]
                decoded.append({"op": OPS[op], "operands": [a, b, c], "constants": [k0, k1]})
            print(json.dumps({"result": "proved_equivalent" if args.width == 32 else "proved_equivalent_modulo",
                              "width": args.width, "program": decoded,
                              "expression": str(candidate), "z3": z3.get_version_string()}))
            return
        if status == z3.unknown:
            print(json.dumps({"result": "inconclusive_verification", "reason": verify.reason_unknown()}))
            return
        value = verify.model().eval(x).as_long()
        samples.append(value)
        concrete = z3.BitVecVal(value, args.width)
        solver.add(program(concrete) == reference(concrete))
    print(json.dumps({"result": "deadline", "samples": len(samples)}))


if __name__ == "__main__":
    main()
