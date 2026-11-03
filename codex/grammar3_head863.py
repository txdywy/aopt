"""Complete three-instruction grammar for the first hash segment.

Today the round input x (after the node XOR) reaches the middle hash F through
four instructions: MAD0, >>19, XOR, XOR C1. F's two leading MADs absorb any odd
affine image of their input, and the node pre-XOR can move an XOR key onto x.
The affine-compatible keys 0, 2^31, 2^32-1 and 2^31-1 are tried here.
A three-instruction replacement would save one VALU op per element per round.

Query: is there a program P of three instructions with

    P(u) == m*T(u XOR e) + c     for every uint32 u,  m odd,
    T(x) = g19(4097*x + C0) XOR C1,   g19(a) = a XOR (a >> 19)?

The grammar covers every VALU opcode except division, remainder and
comparisons: +, -, *, ^, &, |, <<, >> and MAD, with any SSA wiring. Every
operand is the input, an earlier result, or a free constant. Shift amounts may
be constants or runtime values, with full 32-bit semantics: amounts >= 32 give 0.
Single-data-operand affine forms (x+K, x-K, K-x, x*K, x<<K) are subsumed by
MAD(x,K1,K2). Two-operand linear forms (a+b, a-b, a*K+b) are subsumed by
MAD(a,K,b), and a*b by MAD(a,b,K).

Sound pruning only:
  * every instruction is used. Dead code is a shorter program, and with i1 as
    an identity MAD it embeds into this grammar.
  * the program contains an op that is not a T-function. T's bit 0 depends on
    input bit 19, and compositions of T-functions cannot do that.
  * the last instruction is not one whose result can never be a bijection
    (constant right shift, AND/OR with a constant, squares, shifted constants),
    and is not a single-operand affine map, which the output encoding already
    absorbs.
UNSAT on a finite sample set excludes the template for all constants.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
from pathlib import Path
import random
import sys
import time

MASK = (1 << 32) - 1
C0, C1 = 0x7ED55D16, 0xC761C23C
KEYS = (0, 1 << 31, MASK, (1 << 31) - 1)

UNARY = ('aff', 'sq', 'sqx', 'xk', 'andk', 'ork', 'shrk', 'kshl', 'kshr', 'sshl', 'sshr')
BINARY = ('lin1', 'lin2', 'prod', 'xor', 'and', 'or', 'shl_ab', 'shl_ba', 'shr_ab', 'shr_ba',
          'aab', 'bba', 'aba', 'abb')
TERNARY = ('mad_c', 'mad_b', 'mad_a')
# --full adds the remaining ALU opcodes: //, %, cdiv, <, ==
EXTRA_UNARY = ('divk', 'kdiv', 'modk', 'kmod', 'cdivk', 'kcdiv', 'ltk', 'klt', 'eqk')
EXTRA_BINARY = ('div_ab', 'div_ba', 'mod_ab', 'mod_ba', 'cdiv_ab', 'cdiv_ba', 'lt_ab', 'lt_ba', 'eq')
NCONST = {'aff': 2, 'sq': 1, 'xk': 1, 'andk': 1, 'ork': 1, 'shrk': 1, 'kshl': 1, 'kshr': 1,
          'lin1': 1, 'lin2': 1, 'prod': 1, **{f: 1 for f in EXTRA_UNARY}}
NONT = {'shrk', 'kshl', 'kshr', 'sshl', 'sshr', 'shl_ab', 'shl_ba', 'shr_ab', 'shr_ba',
        *EXTRA_UNARY, *EXTRA_BINARY}
NONINJ = {'sq', 'sqx', 'andk', 'ork', 'shrk', 'kshl', 'kshr', 'sshl', 'sshr', *EXTRA_UNARY}
COMMUTATIVE = {'prod', 'xor', 'and', 'or', 'eq'}
# which operand is the divisor: 'k' constant, 'a' or 'b' data; the simulator raises on zero
DIVISOR = {'divk': 'k', 'modk': 'k', 'cdivk': 'k', 'kdiv': 'a', 'kmod': 'a', 'kcdiv': 'a',
           'div_ab': 'b', 'mod_ab': 'b', 'cdiv_ab': 'b', 'div_ba': 'a', 'mod_ba': 'a', 'cdiv_ba': 'a'}


def g19(a):
    return a ^ (a >> 19)


def target(x):
    return g19((4097 * x + C0) & MASK) ^ C1


def ops(lib):
    """Opcode semantics over ints (lib=None) or z3 bit-vectors."""
    if lib is None:
        def shl(a, b):
            return (a << b) & MASK if b < 32 else 0

        def shr(a, b):
            return a >> b if b < 32 else 0
        M = lambda v: v & MASK
        div, mod = (lambda a, b: a // b), (lambda a, b: a % b)
        cdv = lambda a, b: (a + b - 1) // b
        lt, eq = (lambda a, b: int(a < b)), (lambda a, b: int(a == b))
    else:
        shl, shr = (lambda a, b: a << b), lib.LShR  # z3 gives 0 for amounts >= 32
        M = lambda v: v
        div, mod = lib.UDiv, lib.URem  # zero divisors are excluded by side constraints
        # Python evaluates a+b-1 without wrapping, so use 33 bits
        cdv = lambda a, b: lib.Extract(31, 0, lib.UDiv(lib.ZeroExt(1, a) + lib.ZeroExt(1, b) - 1,
                                                        lib.ZeroExt(1, b)))
        one, zero = lib.BitVecVal(1, 32), lib.BitVecVal(0, 32)
        lt = lambda a, b: lib.If(lib.ULT(a, b), one, zero)
        eq = lambda a, b: lib.If(a == b, one, zero)
    return {
        'divk': lambda a, k: div(a, k[0]), 'kdiv': lambda a, k: div(k[0], a),
        'modk': lambda a, k: mod(a, k[0]), 'kmod': lambda a, k: mod(k[0], a),
        'cdivk': lambda a, k: cdv(a, k[0]), 'kcdiv': lambda a, k: cdv(k[0], a),
        'ltk': lambda a, k: lt(a, k[0]), 'klt': lambda a, k: lt(k[0], a),
        'eqk': lambda a, k: eq(a, k[0]),
        'div_ab': lambda a, b, k: div(a, b), 'div_ba': lambda a, b, k: div(b, a),
        'mod_ab': lambda a, b, k: mod(a, b), 'mod_ba': lambda a, b, k: mod(b, a),
        'cdiv_ab': lambda a, b, k: cdv(a, b), 'cdiv_ba': lambda a, b, k: cdv(b, a),
        'lt_ab': lambda a, b, k: lt(a, b), 'lt_ba': lambda a, b, k: lt(b, a),
        'eq': lambda a, b, k: eq(a, b),
        'aff': lambda a, k: M(a * k[0] + k[1]),
        'sq': lambda a, k: M(a * a + k[0]),
        'sqx': lambda a, k: M(a * a + a),
        'xk': lambda a, k: a ^ k[0],
        'andk': lambda a, k: a & k[0],
        'ork': lambda a, k: a | k[0],
        'shrk': lambda a, k: shr(a, k[0]),
        'kshl': lambda a, k: shl(k[0], a),
        'kshr': lambda a, k: shr(k[0], a),
        'sshl': lambda a, k: shl(a, a),
        'sshr': lambda a, k: shr(a, a),
        'lin1': lambda a, b, k: M(k[0] * a + b),
        'lin2': lambda a, b, k: M(k[0] * b + a),
        'prod': lambda a, b, k: M(a * b + k[0]),
        'xor': lambda a, b, k: a ^ b,
        'and': lambda a, b, k: a & b,
        'or': lambda a, b, k: a | b,
        'shl_ab': lambda a, b, k: shl(a, b),
        'shl_ba': lambda a, b, k: shl(b, a),
        'shr_ab': lambda a, b, k: shr(a, b),
        'shr_ba': lambda a, b, k: shr(b, a),
        'aab': lambda a, b, k: M(a * a + b),
        'bba': lambda a, b, k: M(b * b + a),
        'aba': lambda a, b, k: M(a * b + a),
        'abb': lambda a, b, k: M(a * b + b),
        'mad_c': lambda a, b, c, k: M(a * b + c),
        'mad_b': lambda a, b, c, k: M(a * c + b),
        'mad_a': lambda a, b, c, k: M(b * c + a),
    }


def templates(full=False, domain30=False):
    """All canonical three-instruction templates after sound pruning.

    With domain30 the program only has to be injective on 2^30 inputs, so the
    injectivity-based rules are dropped; only the affine-output rule remains.
    """
    out, seen = [], set()
    un = UNARY + EXTRA_UNARY if full else UNARY
    bi = BINARY + EXTRA_BINARY if full else BINARY
    i1s = [(f, (0,)) for f in un]
    i2s = [(f, (0,)) for f in un] + [(f, (1,)) for f in un] + [(f, (0, 1)) for f in bi]
    i3s = ([(f, (2,)) for f in un] + [(f, (0, 2)) for f in bi] +
           [(f, (1, 2)) for f in bi] + [(f, (0, 1, 2)) for f in TERNARY])
    for p in itertools.product(i1s, i2s, i3s):
        if not any(f in NONT for f, _ in p):
            continue
        used = {a for _, args in p[1:] for a in args}
        if 1 not in used:
            continue
        f3, a3 = p[2]
        if f3 == 'aff':
            continue  # absorbed by the output encoding
        if not domain30:
            if len(a3) == 1 and f3 != 'xk':
                continue  # never a bijection
            # the output depends on u only through a non-injective unary value
            if len(a3) == 1 and p[1][0] in NONINJ and len(p[1][1]) == 1:
                continue
            if p[0][0] in NONINJ and 0 not in used:
                continue
            if f3 in ('mod_ab', 'mod_ba', 'lt_ab', 'lt_ba', 'eq'):
                continue  # a % b < b <= 2^32-1 and comparisons are 0/1: not surjective
            if len(a3) == 1 and p[1][0] in ('lt_ab', 'lt_ba', 'eq'):
                continue
        # a runtime divisor equal to the raw input is zero at u = 0
        if not domain30 and any(f in DIVISOR and DIVISOR[f] != 'k' and
                                args['ab'.index(DIVISOR[f])] == 0 for f, args in p):
            continue
        key = canonical(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def canonical(p):
    """Expression string with commutative operands sorted, for de-duplication."""
    names = ['u']  # constants are free, so the constant-free tree is the signature
    for f, args in p:
        a = [names[i] for i in args]
        if f in COMMUTATIVE:
            a = sorted(a)
        if f in ('lin1', 'lin2'):  # K*a+b: lin2(a,b) is lin1(b,a)
            a = a if f == 'lin1' else a[::-1]
            f = 'lin1'
        if f.endswith(('_ab', '_ba')):
            a = a if f.endswith('ab') else a[::-1]
            f = f[:-3]
        if f in ('aab', 'bba'):
            a = a if f == 'aab' else a[::-1]
            f = 'aab'
        if f in ('aba', 'abb'):
            a = a if f == 'aba' else a[::-1]
            f = 'aba'
        if f.startswith('mad_'):
            add = {'mad_c': a[2], 'mad_b': a[1], 'mad_a': a[0]}[f]
            mul = sorted(a[:{'mad_c': 2, 'mad_b': 1, 'mad_a': 0}[f]] + a[{'mad_c': 2, 'mad_b': 1, 'mad_a': 0}[f] + 1:])
            a, f = (mul + [add]), 'mad'
        names.append(f'{f}({",".join(a)})')
    return names[-1]


def describe(p):
    return ' ; '.join(f'i{n + 1}={f}({",".join("u" if i == 0 else f"i{i}" for i in args)})'
                      for n, (f, args) in enumerate(p))


def run_program(p, u, consts, sem, divisors=None):
    vals = [u]
    for (f, args), k in zip(p, consts):
        ins = [vals[i] for i in args]
        if divisors is not None and f in DIVISOR:
            d = DIVISOR[f]
            divisors.append(k[0] if d == 'k' else ins['ab'.index(d)])
        vals.append(sem[f](*ins, k))
    return vals[-1]


# last instruction a // b or cdiv(a, b): (numerator, denominator) operand positions
OUTDIV = {'div_ab': (0, 1), 'div_ba': (1, 0), 'cdiv_ab': (0, 1), 'cdiv_ba': (1, 0)}


def solve(p, seconds=60, free_key=False, seed=863, fix=None):
    import z3
    sem = ops(z3)
    consts = []
    for n, (f, _) in enumerate(p):
        cnt = NCONST.get(f, 0)
        if f == 'shrk':
            consts.append([z3.ZeroExt(27, z3.BitVec(f'k{n}_0', 5))])
        else:
            consts.append([z3.BitVec(f'k{n}_{j}', 32) for j in range(cnt)])
    m, c, esel = z3.BitVec('m', 32), z3.BitVec('c', 32), z3.BitVec('esel', 2)
    e = z3.BitVec('e', 32)  # used only with free_key
    rng = random.Random(seed)
    named = {'m': m, 'c': c, 'e': e, 'esel': esel,
             **{f'k{n}_{j}': k for n, ks in enumerate(consts) for j, k in enumerate(ks)}}
    pts = [0, 1, 2, 3, 1 << 12, 1 << 19, 1 << 20, 1 << 31, MASK, 0x80000001, 0x7FFFF] + \
          [rng.getrandbits(32) for _ in range(9)]
    solver = z3.SolverFor('QF_BV')
    solver.add(z3.Extract(0, 0, m) == 1)
    for name, v in (fix or {}).items():  # partial positive controls
        solver.add(named[name] == v)
    t0 = time.monotonic()

    def add(u):
        divs = []
        lhs = run_program(p, z3.BitVecVal(u, 32), consts, sem, divs)
        for d in divs:
            solver.add(d != 0)
        if free_key:  # any XOR key the node pre-XOR can carry
            a = 4097 * (z3.BitVecVal(u, 32) ^ e) + C0
            rhs = m * (a ^ z3.LShR(a, 19) ^ C1) + c
        else:
            # multiply m by concrete targets only; a symbolic product is far slower
            rv = [m * z3.BitVecVal(target(u ^ k), 32) + c for k in KEYS]
            rhs = z3.If(esel == 0, rv[0], z3.If(esel == 1, rv[1], z3.If(esel == 2, rv[2], rv[3])))
        solver.add(lhs == rhs)

    for u in pts:
        add(u)
    rounds = 0
    while True:
        left = seconds - (time.monotonic() - t0)
        if left <= 0:
            return dict(result='timeout', rounds=rounds, samples=len(pts))
        solver.set(timeout=int(left * 1000))
        rounds += 1
        st = solver.check()
        if st == z3.unsat:
            return dict(result='unsat', rounds=rounds, samples=len(pts),
                        seconds=round(time.monotonic() - t0, 2))
        if st == z3.unknown:
            return dict(result='timeout', rounds=rounds, samples=len(pts))
        mdl = solver.model()
        val = lambda v: mdl.eval(v, model_completion=True).as_long()
        kv = [[val(k) for k in ks] for ks in consts]
        mv, cv = val(m), val(c)
        ev = val(e) if free_key else KEYS[val(esel)]
        csem = ops(None)
        bad = None
        chk = random.Random(rounds)
        for _ in range(4096):
            u = chk.getrandbits(32)
            try:
                ok = run_program(p, u, kv, csem) == (mv * target(u ^ ev) + cv) & MASK
            except ZeroDivisionError:
                ok = False
            if not ok:
                bad = u
                break
        if bad is None:
            full = verify_full(p, kv, mv, cv, ev)
            if full is None:
                return dict(result='sat', consts=kv, m=mv, c=cv, e=ev, rounds=rounds,
                            seconds=round(time.monotonic() - t0, 2))
            bad = full
        pts.append(bad)
        add(bad)


D30 = 1 << 30  # round-0 hash inputs are value XOR root, both below 2^30


def solve_domain(p, seconds=60, key=0, seed=863, free_low=False, nsamples=20, sparse=False):
    """Round-0 variant: P(x XOR key) == m*T(x) + c only for x < 2^30.

    The root vector of round 0 can carry any key, so each key is its own query.
    With free_low the low 30 key bits are unknowns; key's top two bits stay fixed.
    Samples are then program inputs u whose top bits match the key, so
    x = u XOR key is always inside the domain.
    """
    import z3
    sem = ops(z3)
    consts = []
    for n, (f, _) in enumerate(p):
        if f == 'shrk':
            consts.append([z3.ZeroExt(27, z3.BitVec(f'k{n}_0', 5))])
        else:
            consts.append([z3.BitVec(f'k{n}_{j}', 32) for j in range(NCONST.get(f, 0))])
    m, c = z3.BitVec('m', 32), z3.BitVec('c', 32)
    top = key & ~(D30 - 1) & MASK
    ekey = z3.Concat(z3.BitVecVal(top >> 30, 2), z3.BitVec('elow', 30)) if free_low else None
    rng = random.Random(seed)
    pts = [0, 1, 2, 3, 1 << 12, 1 << 19, 1 << 20, 1 << 29, D30 - 1, 0x2AAAAAAA, 0x7FFFF] + \
          [rng.randrange(D30) for _ in range(9)]
    if sparse:  # few set bits keep products of unknowns with samples cheap
        pts = sorted({sum(1 << b for b in rng.sample(range(30), rng.randint(1, 3)))
                      for _ in range(4 * nsamples)})[:nsamples]
    else:
        pts = pts[:nsamples]
    if free_low:  # samples are u = top | low30
        pts = [top | x for x in pts]
    solver = z3.SolverFor('QF_BV')
    solver.add(z3.Extract(0, 0, m) == 1)
    t0 = time.monotonic()

    def add(x):
        divs = []
        if free_low:
            u = x
            lhs = run_program(p, z3.BitVecVal(u, 32), consts, sem, divs)
            a = 4097 * (z3.BitVecVal(u, 32) ^ ekey) + C0
            rhs = m * (a ^ z3.LShR(a, 19) ^ C1) + c
        else:
            lhs = run_program(p, z3.BitVecVal(x ^ key, 32), consts, sem, divs)
            rhs = m * z3.BitVecVal(target(x), 32) + c
        for d in divs:
            solver.add(d != 0)
        solver.add(lhs == rhs)

    for x in pts:
        add(x)
    rounds = 0
    while True:
        left = seconds - (time.monotonic() - t0)
        if left <= 0:
            return dict(result='timeout', rounds=rounds, samples=len(pts))
        solver.set(timeout=int(left * 1000))
        rounds += 1
        st = solver.check()
        if st == z3.unsat:
            return dict(result='unsat', rounds=rounds, samples=len(pts),
                        seconds=round(time.monotonic() - t0, 2))
        if st == z3.unknown:
            return dict(result='timeout', rounds=rounds, samples=len(pts))
        mdl = solver.model()
        val = lambda v: mdl.eval(v, model_completion=True).as_long()
        kv = [[val(k) for k in ks] for ks in consts]
        mv, cv = val(m), val(c)
        ev = val(ekey) if free_low else key
        csem = ops(None)
        bad = None
        chk = random.Random(rounds)
        for _ in range(4096):
            x = chk.randrange(D30)
            try:
                ok = run_program(p, x ^ ev, kv, csem) == (mv * target(x) + cv) & MASK
            except ZeroDivisionError:
                ok = False
            if not ok:
                bad = x
                break
        if bad is None:
            full = verify_full(p, kv, mv, cv, ev, domain=D30)
            if full is None:
                return dict(result='sat', consts=kv, m=mv, c=cv, e=ev, rounds=rounds,
                            seconds=round(time.monotonic() - t0, 2))
            bad = full
        if free_low:
            bad ^= ev  # store the program input; its top bits equal the key's
        pts.append(bad)
        add(bad)


def refute_low(p, seconds=60, k=16, domain_top=None, seed=863):
    """Necessary condition with a free 32-bit XOR key, compared on the low k bits.

    m is odd, so (m*T + c) mod 2^k only involves m mod 2^k and T mod 2^k; the
    program itself is still evaluated at full width. UNSAT here excludes the
    template for every key; SAT is inconclusive ('open'). domain_top selects
    the round-0 domain: inputs u with u >> 30 == domain_top and key top bits equal.
    """
    import z3
    sem = ops(z3)
    consts = []
    for n, (f, _) in enumerate(p):
        if f == 'shrk':
            consts.append([z3.ZeroExt(27, z3.BitVec(f'k{n}_0', 5))])
        else:
            consts.append([z3.BitVec(f'k{n}_{j}', 32) for j in range(NCONST.get(f, 0))])
    mk, ck = z3.BitVec('m', k), z3.BitVec('c', k)
    if domain_top is None:
        e = z3.BitVec('e', 32)
    else:
        e = z3.Concat(z3.BitVecVal(domain_top, 2), z3.BitVec('elow', 30))
    rng = random.Random(seed)
    base = [0, 1, 2, 3, 1 << 12, 1 << 19, 1 << 20, 1 << 29, D30 - 1, 0x2AAAAAAA, 0x7FFFF]
    if domain_top is None:
        pts = base + [1 << 31, MASK, 0x80000001] + [rng.getrandbits(32) for _ in range(18)]
    else:
        pts = [(domain_top << 30) | x for x in base + [rng.randrange(D30) for _ in range(21)]]
    solver = z3.SolverFor('QF_BV')
    solver.set(timeout=int(seconds * 1000))
    solver.add(z3.Extract(0, 0, mk) == 1)
    for u in pts:
        divs = []
        lhs = run_program(p, z3.BitVecVal(u, 32), consts, sem, divs)
        for d in divs:
            solver.add(d != 0)
        a = 4097 * (z3.BitVecVal(u, 32) ^ e) + C0
        t = z3.Extract(k - 1, 0, a ^ z3.LShR(a, 19) ^ C1)
        solver.add(z3.Extract(k - 1, 0, lhs) == mk * t + ck)
    t0 = time.monotonic()
    st = solver.check()
    res = 'unsat' if st == z3.unsat else 'open' if st == z3.sat else 'timeout'
    return dict(result=res, bits=k, samples=len(pts), seconds=round(time.monotonic() - t0, 2))


def _named(z3, name, width):
    """Bit-vector built from named Booleans, so CNF models decode by name."""
    bits = [z3.Bool(f'{name}_{i}') for i in range(width)]
    one, zero = z3.BitVecVal(1, 1), z3.BitVecVal(0, 1)
    bv = z3.Concat(*[z3.If(b, one, zero) for b in reversed(bits)]) if width > 1 else z3.If(bits[0], one, zero)
    return bv, bits


def solve_cnf(p, seconds=60, free_key=False, seed=863, engine='cadical195'):
    """Same query as solve(), bit-blasted per sample and solved incrementally by CaDiCaL."""
    import threading
    import z3
    from pysat.solvers import Solver
    sem = ops(z3)
    named = {}
    consts = []
    for n, (f, _) in enumerate(p):
        ks = []
        for j in range(NCONST.get(f, 0)):
            w = 5 if f == 'shrk' else 32
            bv, bits = _named(z3, f'k{n}_{j}', w)
            named[f'k{n}_{j}'] = bits
            ks.append(z3.ZeroExt(27, bv) if w == 5 else bv)
        consts.append(ks)
    m, named['m'] = _named(z3, 'm', 32)
    c, named['c'] = _named(z3, 'c', 32)
    e, named['e'] = _named(z3, 'e', 32)
    esel, named['esel'] = _named(z3, 'esel', 2)
    ids = {}  # SAT variable per named bit; other atoms get fresh variables per sample
    nxt = [0]

    def var():
        nxt[0] += 1
        return nxt[0]

    for bits in named.values():
        for b in bits:
            ids[b.get_id()] = var()
    tactic = z3.Then('simplify', 'bit-blast', 'tseitin-cnf')

    def clauses(cons):
        g = z3.Goal()
        g.add(*cons)
        res = tactic(g)
        local = {}
        out = []

        def lit(x):
            neg = z3.is_not(x)
            a = x.arg(0) if neg else x
            k = a.get_id()
            v = ids.get(k) or local.get(k)
            if v is None:
                v = local[k] = var()
            return -v if neg else v

        for cl in res[0]:
            if z3.is_false(cl):
                return None
            if z3.is_true(cl):
                continue
            out.append([lit(x) for x in cl.children()] if z3.is_or(cl) else [lit(cl)])
        return out

    rng = random.Random(seed)
    pts = [0, 1, 2, 3, 1 << 12, 1 << 19, 1 << 20, 1 << 31, MASK, 0x80000001, 0x7FFFF] + \
          [rng.getrandbits(32) for _ in range(9)]

    def sample(u):
        divs = []
        lhs = run_program(p, z3.BitVecVal(u, 32), consts, sem, divs)
        cons = [d != 0 for d in divs]
        if free_key:
            a = 4097 * (z3.BitVecVal(u, 32) ^ e) + C0
            rhs = m * (a ^ z3.LShR(a, 19) ^ C1) + c
        else:
            rv = [m * z3.BitVecVal(target(u ^ k), 32) + c for k in KEYS]
            rhs = z3.If(esel == 0, rv[0], z3.If(esel == 1, rv[1], z3.If(esel == 2, rv[2], rv[3])))
        return clauses(cons + [lhs == rhs])

    t0 = time.monotonic()
    sol = Solver(name=engine)
    sol.add_clause([ids[named['m'][0].get_id()]])  # m odd
    try:
        for u in pts:
            cl = sample(u)
            if cl is None:
                return dict(result='unsat', how='simplify', samples=len(pts))
            sol.append_formula(cl)
        rounds = 0
        while True:
            left = seconds - (time.monotonic() - t0)
            if left <= 0:
                return dict(result='timeout', rounds=rounds, samples=len(pts))
            rounds += 1
            if engine.startswith(('cadical', 'kissat')):
                st = sol.solve()  # pysat cannot interrupt these: no time limit, controls only
            else:
                timer = threading.Timer(left, sol.interrupt)
                timer.start()
                st = sol.solve_limited(expect_interrupt=True)
                timer.cancel()
                sol.clear_interrupt()
            if st is False:
                return dict(result='unsat', engine=engine, rounds=rounds, samples=len(pts),
                            seconds=round(time.monotonic() - t0, 2))
            if st is None:
                return dict(result='timeout', rounds=rounds, samples=len(pts))
            model = set(l for l in sol.get_model() if l > 0)
            val = lambda name: sum(1 << i for i, b in enumerate(named[name]) if ids[b.get_id()] in model)
            kv = []
            for n, (f, _) in enumerate(p):
                kv.append([val(f'k{n}_{j}') for j in range(NCONST.get(f, 0))])
            mv, cv = val('m'), val('c')
            ev = val('e') if free_key else KEYS[val('esel')]
            csem = ops(None)
            bad = None
            chk = random.Random(rounds)
            for _ in range(4096):
                u = chk.getrandbits(32)
                try:
                    ok = run_program(p, u, kv, csem) == (mv * target(u ^ ev) + cv) & MASK
                except ZeroDivisionError:
                    ok = False
                if not ok:
                    bad = u
                    break
            if bad is None:
                full = verify_full(p, kv, mv, cv, ev)
                if full is None:
                    return dict(result='sat', consts=kv, m=mv, c=cv, e=ev, rounds=rounds,
                                seconds=round(time.monotonic() - t0, 2))
                bad = full
            pts.append(bad)
            cl = sample(bad)
            if cl is None:
                return dict(result='unsat', how='simplify', samples=len(pts))
            sol.append_formula(cl)
    finally:
        sol.delete()


C_OPS = {
    'aff': 'A*K0+K1', 'sq': 'A*A+K0', 'sqx': 'A*A+A', 'xk': 'A^K0', 'andk': 'A&K0',
    'ork': 'A|K0', 'shrk': 'SHR(A,K0)', 'kshl': 'SHL(K0,A)', 'kshr': 'SHR(K0,A)',
    'sshl': 'SHL(A,A)', 'sshr': 'SHR(A,A)', 'lin1': 'K0*A+B', 'lin2': 'K0*B+A',
    'prod': 'A*B+K0', 'xor': 'A^B', 'and': 'A&B', 'or': 'A|B', 'shl_ab': 'SHL(A,B)',
    'shl_ba': 'SHL(B,A)', 'shr_ab': 'SHR(A,B)', 'shr_ba': 'SHR(B,A)', 'aab': 'A*A+B',
    'bba': 'B*B+A', 'aba': 'A*B+A', 'abb': 'A*B+B', 'mad_c': 'A*B+C', 'mad_b': 'A*C+B',
    'mad_a': 'B*C+A',
    'divk': 'DIV(A,K0)', 'kdiv': 'DIV(K0,A)', 'modk': 'MOD(A,K0)', 'kmod': 'MOD(K0,A)',
    'cdivk': 'CDIV(A,K0)', 'kcdiv': 'CDIV(K0,A)', 'ltk': '(A<K0)', 'klt': '(K0<A)',
    'eqk': '(A==K0)', 'div_ab': 'DIV(A,B)', 'div_ba': 'DIV(B,A)', 'mod_ab': 'MOD(A,B)',
    'mod_ba': 'MOD(B,A)', 'cdiv_ab': 'CDIV(A,B)', 'cdiv_ba': 'CDIV(B,A)', 'lt_ab': '(A<B)',
    'lt_ba': '(B<A)', 'eq': '(A==B)',
}


def c_program(p, kv, mv, cv, ev, domain=None):
    """C source that checks the candidate on all 2^32 inputs, or on x < domain."""
    lines = []
    for n, ((f, args), ks) in enumerate(zip(p, kv)):
        expr = C_OPS[f]
        for name, i in zip('ABC', args):
            expr = expr.replace(name, 'v[%d]' % i)
        for j, k in enumerate(ks):
            expr = expr.replace('K%d' % j, '%uu' % k)
        lines.append('    v[%d] = (uint32_t)(%s);' % (n + 1, expr))
    body = '\n'.join(lines)
    if domain is None:  # u runs over all words, x = u ^ e
        pick, step, xsrc, wit = 'v[0] = u;', '++u != 0', f'u ^ {ev}u', 'u'
    else:  # x runs over [0, domain), u = x ^ e; report x
        pick, step, xsrc, wit = f'u = it ^ {ev}u; v[0] = u;', f'++it < {domain}u', 'it', 'it'
    return f"""#include <stdint.h>
#include <stdio.h>
static inline uint32_t SHL(uint32_t a, uint32_t b) {{ return b < 32 ? a << b : 0; }}
static inline uint32_t SHR(uint32_t a, uint32_t b) {{ return b < 32 ? a >> b : 0; }}
static int zdiv;  /* the simulator raises on a zero divisor: such a program is invalid */
static inline uint32_t DIV(uint32_t a, uint32_t b) {{ if (!b) {{ zdiv = 1; return 0; }} return a / b; }}
static inline uint32_t MOD(uint32_t a, uint32_t b) {{ if (!b) {{ zdiv = 1; return 0; }} return a % b; }}
static inline uint32_t CDIV(uint32_t a, uint32_t b) {{
  if (!b) {{ zdiv = 1; return 0; }} return (uint32_t)(((uint64_t)a + b - 1) / b); }}
int main(void) {{
  uint32_t u = 0, it = 0;
  do {{
    uint32_t v[{len(p) + 1}]; {pick} zdiv = 0;
{body}
    if (zdiv) {{ printf("%u\\n", {wit}); return 0; }}
    uint32_t x = {xsrc};
    uint32_t a = 4097u * x + {C0}u;
    uint32_t t = a ^ (a >> 19) ^ {C1}u;
    if (v[{len(p)}] != (uint32_t)({mv}u * t + {cv}u)) {{ printf("%u\\n", {wit}); return 0; }}
  }} while ({step});
  printf("ok\\n");
  return 0;
}}
"""


def verify_full(p, kv, mv, cv, ev, domain=None):
    """Return None if the candidate is correct on all inputs, else a witness."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        src, exe = Path(d) / 'chk.c', Path(d) / 'chk'
        src.write_text(c_program(p, kv, mv, cv, ev, domain))
        subprocess.run(['clang', '-O2', '-o', str(exe), str(src)], check=True)
        out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout.strip()
    return None if out == 'ok' else int(out)


def job(args):
    idx, p, seconds, free_key, engine, key = args
    t0 = time.monotonic()
    try:
        if key is not None:
            r = solve_domain(p, seconds, key, free_low=free_key)
        elif engine == 'z3':
            r = solve(p, seconds, free_key)
        else:
            r = solve_cnf(p, seconds, free_key, engine=engine)
    except Exception as exc:  # report, never hide
        r = dict(result='error', error=f'{type(exc).__name__}: {exc}'[:200])
    r.update(idx=idx, template=describe(p), wall=round(time.monotonic() - t0, 2))
    if key is not None:
        r['key'] = key
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--count', action='store_true')
    ap.add_argument('--output', type=Path)
    ap.add_argument('--seconds', type=float, default=60)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--only', type=str, default='', help='comma-separated template indices')
    ap.add_argument('--skip-done', action='store_true', help='resume: skip indices already in --output')
    ap.add_argument('--full', action='store_true', help='add //, %%, cdiv, <, ==')
    ap.add_argument('--new-only', action='store_true', help='with --full: only templates using them')
    ap.add_argument('--free-key', action='store_true', help='any 32-bit XOR key on the input')
    ap.add_argument('--engine', default='z3', help='z3, or a pysat solver such as cadical195')
    ap.add_argument('--one', type=int, default=None, help='run one template in-process, print JSON')
    ap.add_argument('--domain30', type=str, default='',
                    help='round 0: comma-separated input keys, correctness only for x < 2^30')
    args = ap.parse_args()
    keys = [int(k, 0) for k in args.domain30.split(',')] if args.domain30 else [None]
    ts = templates(args.full, bool(args.domain30))
    if args.count:
        from collections import Counter
        print(len(ts))
        print(Counter(f for p in ts for f, _ in p).most_common())
        return
    if args.one is not None:
        print(json.dumps(job((args.one, ts[args.one], args.seconds, args.free_key, args.engine, keys[0]))))
        return
    idxs = [int(i) for i in args.only.split(',')] if args.only else list(range(len(ts)))
    if args.new_only:
        extra = set(EXTRA_UNARY) | set(EXTRA_BINARY)
        idxs = [i for i in idxs if any(f in extra for f, _ in ts[i])]
    done = set()
    if args.skip_done and args.output and args.output.exists():
        done = {(json.loads(l)['idx'], json.loads(l).get('key')) for l in args.output.open()}
    todo = [(i, ts[i], args.seconds, args.free_key, args.engine, k)
            for k in keys for i in idxs if (i, k) not in done]
    print(json.dumps(dict(templates=len(ts), todo=len(todo))), flush=True)
    mode = 'a' if args.skip_done else 'w'
    with args.output.open(mode) as out, ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(job, t) for t in todo]
        for f in as_completed(futs):
            r = f.result()
            out.write(json.dumps(r) + '\n')
            out.flush()
            if r['result'] not in ('unsat',):
                print(json.dumps(r), flush=True)


if __name__ == '__main__':
    main()
