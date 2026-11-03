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


def middle_reference(x, input_key=0, output_key=0):
    """Full uint32 middle hash, optionally with XOR encodings at its ends.

    The same expression supports Python integers and Z3 bitvectors. Without
    shifts, truncating all operands to a smaller width is a sound projection.
    """
    if not 0 <= input_key <= 0xffffffff or not 0 <= output_key <= 0xffffffff:
        raise ValueError("middle XOR keys must be uint32")
    y = x ^ input_key
    return ((9*((33*y+0xE9F8CC1D) ^ (16896*y+0xACCF6200))
             +0xFD7046C5) ^ output_key) & 0xffffffff


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
    parser.add_argument("--consecutive-samples", type=int, default=0,
                        help="Include inputs 0..N-1, useful for shift-boundary screens")
    parser.add_argument("--high-bit-pairs", action="store_true",
                        help="Add paired inputs differing only in the highest bit")
    parser.add_argument("--first-boundaries", action="store_true",
                        help="Sample preimages around first-hash right-shift boundaries")
    parser.add_argument("--first-injective-cut", action="store_true",
                        help="For fixed positive SHR in MAD/SHR/MAD, retain an unshifted input")
    parser.add_argument("--slice", choices=("xor-shift", "middle", "first", "absorb-c1"), default="xor-shift")
    parser.add_argument("--middle-input-xor", type=lambda value: int(value, 0), default=0,
                        help="Input XOR key for --slice middle")
    parser.add_argument("--middle-output-xor", type=lambda value: int(value, 0), default=0,
                        help="Output XOR key for --slice middle")
    parser.add_argument("--width", type=int, default=32,
                        help="Low-bit screening for middle without shifts; UNSAT excludes full-width programs in this grammar")
    parser.add_argument("--op-sequence", help="Comma-separated fixed opcodes, e.g. mad,mad,^,mad")
    parser.add_argument("--first-mad-form", choices=("affine", "square-constant", "square-input"),
                        help="Fix first MAD structure; together these cover all one-input MAD functions")
    parser.add_argument("--first-multiplier", type=lambda value: int(value, 0),
                        help="Fix A in the affine first MAD; leaves its offset arbitrary")
    parser.add_argument("--last-mad-operands", help="Fix final MAD operand indices, e.g. 2,3,1")
    args = parser.parse_args()
    if any(not 0 <= key <= 0xffffffff for key in
           (args.middle_input_xor, args.middle_output_xor)):
        parser.error('middle XOR keys must be uint32')
    if args.slice != 'middle' and (args.middle_input_xor or args.middle_output_xor):
        parser.error('middle XOR keys require --slice middle')
    if args.first_boundaries and args.slice != 'first':
        parser.error('first-boundaries requires --slice first')
    if not 1 <= args.instructions <= 4:
        parser.error("instruction count must be 1..4")
    if args.seconds <= 0 or args.check_ms <= 0:
        parser.error("time limits must be positive")
    if args.extra_samples < 0:
        parser.error("extra-samples must be nonnegative")
    if not 0 <= args.consecutive_samples <= 4096:
        parser.error("consecutive-samples must be in 0..4096")
    if args.right_shift_amount is not None and not 0 <= args.right_shift_amount <= 32:
        parser.error("right-shift-amount must be in 0..32")
    if not 1 <= args.width <= 32:
        parser.error("width must be 1..32")
    if args.width != 32 and args.slice not in ("middle", "absorb-c1"):
        parser.error("low-bit screening is supported only for middle and absorb-c1")
    sequence = args.op_sequence.split(",") if args.op_sequence else None
    if args.first_injective_cut and not (args.slice == 'first' and
            sequence == ['mad','>>','mad'] and args.right_shift_amount is not None
            and 1 <= args.right_shift_amount <= 32):
        parser.error('first-injective-cut requires first slice and mad,>>,mad with positive literal shift')
    last_operands = None
    if args.last_mad_operands:
        try:
            last_operands = tuple(map(int,args.last_mad_operands.split(',')))
        except ValueError:
            parser.error('last-mad-operands must be three integer indices')
        if (not sequence or sequence[-1] != 'mad' or len(last_operands) != 3
                or any(i < 0 or i >= args.instructions+2 for i in last_operands)):
            parser.error('last-mad-operands requires final MAD and valid operand indices')
    if sequence and (len(sequence) != args.instructions or any(op not in OPS for op in sequence)):
        parser.error("op-sequence must contain one valid opcode per instruction")
    if args.first_mad_form and (not sequence or sequence[0] != "mad"):
        parser.error("first-mad-form requires an explicit sequence starting with mad")
    if args.first_multiplier is not None and args.first_mad_form != "affine":
        parser.error("first-multiplier requires first-mad-form affine")
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
        if i == 0 and args.first_mad_form:
            operands = {"affine": (0, 1, 2), "square-constant": (0, 0, 1),
                        "square-input": (0, 0, 0)}[args.first_mad_form]
            solver.add(a == operands[0], b == operands[1], c == operands[2])
            if args.first_multiplier is not None:
                solver.add(k0 == args.first_multiplier)
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
    if last_operands is not None:
        solver.add(*(operand == index for operand,index in zip(spec[-1][1:4],last_operands)))

    if args.first_injective_cut:
        # First hash is bijective: odd affine map, invertible XOR-shift, XOR.
        # If final MAD sees only the shifted intermediate and constants,
        # it has at most 2**(32-shift) distinct outputs, regardless of constants.
        # Its operand pool is [x, firstMAD, shifted, k0, k1].
        solver.add(z3.Or(*(operand == index for operand in spec[2][1:4]
                          for index in (0,1))))

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
        if args.slice == "absorb-c1":
            b = x ^ (0 if args.control else 0xC761C23C)
            return (33*b+0xE9F8CC1D) ^ (16896*b+0xACCF6200)
        if args.slice == "middle":
            return middle_reference(x, args.middle_input_xor, args.middle_output_xor)
        if args.slice == "first":
            a = 4097*x + 0x7ED55D16
            return a ^ z3.LShR(a, 19) ^ (0 if args.control else 0xC761C23C)
        return x ^ z3.LShR(x, 19) ^ (0 if args.control else 0xC761C23C)

    samples = list(dict.fromkeys(value & ((1 << args.width)-1) for value in
                               [0, 1, 16, 0xFFFFFFFF, 1 << 19, (1 << 19)-1]))
    rng = random.Random(0)
    samples = list(dict.fromkeys(samples + [rng.getrandbits(args.width)
                                          for _ in range(args.extra_samples)]))
    samples = list(dict.fromkeys(samples + [value & ((1 << args.width)-1)
                                           for value in range(args.consecutive_samples)]))
    if args.first_boundaries:
        inverse = pow(4097,-1,1<<32)
        boundaries = [((block*(1<<19)+delta-0x7ed55d16)*inverse)&0xffffffff
                      for block in (0,1,2,3,7,15,31,63,127,255,511,1023,2047,4095,8191)
                      for delta in (-1,0,1)]
        assert all(((4097*x+0x7ed55d16)&0x7ffff) in (0,1,0x7ffff) for x in boundaries)
        samples = list(dict.fromkeys(samples+boundaries))
    if args.high_bit_pairs:
        samples = list(dict.fromkeys(samples + [value ^ (1 << (args.width-1))
                                               for value in samples]))
    for value in samples:
        x = z3.BitVecVal(value, args.width)
        solver.add(program(x) == reference(x))
    deadline = time.monotonic() + args.seconds
    target_metadata = dict(slice=args.slice, width=args.width,
                           op_sequence=sequence,
                           middle_input_xor=args.middle_input_xor,
                           middle_output_xor=args.middle_output_xor)
    iteration = 0
    while time.monotonic() < deadline:
        iteration += 1
        solver.set(timeout=min(args.check_ms, max(1, int((deadline-time.monotonic())*1000))))
        status = solver.check()
        print(json.dumps({"iteration": iteration, "samples": len(samples), "synthesis": str(status)}), flush=True)
        if status != z3.sat:
            print(json.dumps({**target_metadata,
                              "result": "grammar_unsatisfiable" if status == z3.unsat else "inconclusive",
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
            print(json.dumps({**target_metadata,
                              "result": "proved_equivalent" if args.width == 32 else "proved_equivalent_modulo",
                              "program": decoded,
                              "expression": str(candidate), "z3": z3.get_version_string()}))
            return
        if status == z3.unknown:
            print(json.dumps({**target_metadata, "result": "inconclusive_verification",
                              "reason": verify.reason_unknown()}))
            return
        value = verify.model().eval(x).as_long()
        samples.append(value)
        concrete = z3.BitVecVal(value, args.width)
        solver.add(program(concrete) == reference(concrete))
    print(json.dumps({**target_metadata, "result": "deadline", "samples": len(samples)}))


if __name__ == "__main__":
    main()
