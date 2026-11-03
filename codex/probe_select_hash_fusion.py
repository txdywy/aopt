"""Two-instruction MAD/XOR fusion probe with arbitrary node-pair constants.

Finite-sample UNSAT excludes a template. SAT is only a candidate and must
be checked over all 32-bit inputs before any implementation is proposed.
"""
import json
import random
import z3


def probe(order,n0,n1,correlated=False):
    constants = [z3.BitVec(f'k{i}',32) for i in range(4)]
    selectors = [[z3.Int(f's{i}_{j}') for j in range(3 if op == 'mad' else 2)]
                 for i,op in enumerate(order)]
    solver = z3.Solver()
    solver.set(timeout=8000)
    for i,ss in enumerate(selectors):
        for s in ss:
            solver.add(s >= 0,s < 6+i)

    def expression(x,p):
        pool = [x,p]+constants
        for op,ss in zip(order,selectors):
            args = []
            for s in ss:
                a = pool[-1]
                for i in range(len(pool)-2,-1,-1):
                    a = z3.If(s == i,pool[i],a)
                args.append(a)
            if op == 'mad':
                value = args[0]*args[1]+args[2]
            elif op == 'xor':
                value = args[0]^args[1]
            elif op == 'and':
                value = args[0]&args[1]
            elif op == 'or':
                value = args[0]|args[1]
            elif op == 'add':
                value = args[0]+args[1]
            elif op == 'sub':
                value = args[0]-args[1]
            elif op == 'shr':
                value = z3.LShR(args[0],args[1])
            elif op == 'shl':
                value = args[0]<<args[1]
            else:
                raise ValueError(op)
            pool.append(value)
        return pool[-1]

    def target(x,p):
        return (x ^ z3.If(p == 0,z3.BitVecVal(n0,32),z3.BitVecVal(n1,32)))*4097+0x7ed55d16

    rng = random.Random(900)
    samples = list(range(12))+[1<<i for i in range(32)]+[rng.getrandbits(32) for _ in range(12)]
    for value in samples:
        for bit in (0,1):
            if correlated and bit != value & 1:
                continue
            x,p = z3.BitVecVal(value,32),z3.BitVecVal(bit,32)
            solver.add(expression(x,p) == target(x,p))
    status = solver.check()
    result = dict(order=order,nodes=[hex(n0),hex(n1)],parity_correlated=correlated,sample_status=str(status))
    if status == z3.sat:
        model = solver.model()
        x,p = z3.BitVecs('x p',32)
        substitutions = [(v,model.eval(v,model_completion=True)) for v in constants+sum(selectors,[])]
        candidate = z3.simplify(z3.substitute(expression(x,p),substitutions))
        verify = z3.Solver()
        verify.set(timeout=8000)
        verify.add(z3.Or(p == 0,p == 1),candidate != target(x,p))
        if correlated:
            verify.add(p == x & 1)
        result.update(candidate=str(candidate),counterexample_status=str(verify.check()))
    return result


if __name__ == '__main__':
    # Positive control: identical nodes remove the selection requirement.
    for nodes in ((0x12345678,0x12345678),(0x12345678,0x9abcdef0)):
        for order in (('xor','mad'),('mad','xor')):
            print(json.dumps(probe(order,*nodes)),flush=True)
