"""A narrow obstruction, not a lower bound for every address algorithm.

Can a single basic bit-vector operation turn arbitrary q into -5-(q&1)?
UNSAT on even a finite input subset excludes the tested expression family
on all uint32 inputs. SAT would only be a candidate, not an equivalence proof.
"""
import json
import z3


def main():
    samples = (0, 1, 2, 3, 4, 7, 0x80000000, 0xFFFFFFFF)
    operations = {
        "+": lambda a, b: a + b, "-": lambda a, b: a - b,
        "*": lambda a, b: a * b, "^": lambda a, b: a ^ b,
        "&": lambda a, b: a & b, "|": lambda a, b: a | b,
        "<<": lambda a, b: a << b, ">>": z3.LShR,
    }
    results = {}
    for name, operation in operations.items():
        for reverse in (False, True):
            constant = z3.BitVec("constant", 32)
            solver = z3.Solver()
            solver.set(timeout=5000)
            for value in samples:
                q = z3.BitVecVal(value, 32)
                expression = operation(constant, q) if reverse else operation(q, constant)
                solver.add(expression == z3.BitVecVal(-5 - (value & 1), 32))
            results[f"{'C op q' if reverse else 'q op C'}:{name}"] = str(solver.check())
    # An affine MAD directly on q cannot produce this alternating function.
    multiplier, bias = z3.BitVecs("multiplier bias", 32)
    solver = z3.Solver()
    for value in samples:
        solver.add(z3.BitVecVal(value, 32) * multiplier + bias ==
                   z3.BitVecVal(-5 - (value & 1), 32))
    results["q*M+B"] = str(solver.check())
    # Frozen cdiv evaluates the addition without uint32 truncation, then
    # truncates its result. Use33 bits for that intermediate, unlike add ISA.
    for name in ('//','%','cdiv'):
        constant=z3.BitVec('divisor',32)
        solver=z3.Solver()
        solver.set(timeout=5000)
        solver.add(constant!=0)
        for value in samples:
            q=z3.BitVecVal(value,32)
            if name=='//':expression=z3.UDiv(q,constant)
            elif name=='%':expression=z3.URem(q,constant)
            else:
                wide=z3.UDiv(z3.ZeroExt(1,q)+z3.ZeroExt(1,constant)-1,
                             z3.ZeroExt(1,constant))
                expression=z3.Extract(31,0,wide)
            solver.add(expression==z3.BitVecVal(-5-(value&1),32))
        results['q op C:'+name]=str(solver.check())
        # C op q is undefined on q=0 in the frozen ISA: no total uint32
        # function can use this family without an additional guard/operation.
        results['C op q:'+name]='undefined_at_zero'
    print(json.dumps({"scope": "listed one-operation constant families only; unsigned division/modulo and wide cdiv included",
                      "samples": samples, "results": results}, indent=2))
    assert all(result in ("unsat","undefined_at_zero") for result in results.values())


if __name__ == "__main__":
    main()
