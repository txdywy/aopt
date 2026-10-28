"""Merge the pinned-shift runs with the exact bit-0 rules; report what stays open.

For a single-shift template, s < 19 is impossible (grammar4_round0_pinned863).
Each shift amount s = 19..31 is a case, closed by one of:

  solver     the pinned z3 run is UNSAT (aff(xk(u)) cases with exact low-bit pinning)
  lemma      bit0_width (z3 on the bit-0 condition at width s+1) is UNSAT for the
             template's shift input; it is tried for cases the full run left open
  two-const  s = 19, shift input u + a*(u op b): the exhaustive check found nothing
  quad       shift input prod(u, aff(u)), s <= 29: reduces to s = 19 (see
             grammar4_round0_bit0_863), where the exhaustive check found nothing
  valuation  large s. Flipping input bit j in 20..29 moves the target by m * (T' - T),
             whose 2-adic valuation is exactly j - 19. Every value other than i3 = w >> s
             is a T-function and moves by a multiple of 2^j, while i3 has 32 - s bits, so
             0 < |di3| < 2^(32-s) whenever it moves. If the last instruction is A + B*i3
             (lin1, lin2, prod, the MADs; A, B T-functions or constants), the output moves
             by dA + B(u')*di3 + i3*dB, so v2(B * di3) must be j - 19 for every j: j = 20
             gives v2(B) <= 1, j = 29 gives v2(B) >= 10 - (31 - s); impossible for s >= 23.
             If it is X ^ i3, X | i3 or X & i3, the part of the move below bit 32 - s is
             nonzero and below 2^(32-s), so j - 19 <= 31 - s for j = 29: impossible for
             s >= 22.
  rigid      s = 19..24, rule 4 of grammar4_round0_bit0_863 (rigid differences):
             a witness pair of inputs whose output moves agree mod 2^(32-s) for
             any constants while the target's moves differ
  xor-aff    s = 21..29, shift input u ^ (mu*u + nu). Flipping u_s flips u_s and, for
             odd mu only, bit s of mu*u + nu, so mu is odd. Then with A = (mu-1)u + nu
             the bit is F = A_s ^ c_s, c_s the carry into bit s of u + A. Flipping
             u_(s-1) (s-1 >= 20, y19 unaffected) changes F by m0 ^ [X + Y in
             [2^(s-1), 2^s)], X = u mod 2^(s-1), Y = A mod 2^s, m = (mu-1)/2; so
             X -> mu*X + nu maps [0, 2^(s-1)) onto a half of Z/2^s. Exactly one X in
             [0, 2^(s-1)) has X + 1 outside it, so the half S has one z with z + mu
             outside S, which for an interval forces mu = +-1 mod 2^s. Then F is
             nu_s ^ (carry of u + nu, or borrow of nu - u, at bit s) ^ (const * u0),
             and that threshold in u mod 2^s must not depend on bits 20..s-1, so it is
             constant: F depends on u0 alone and cannot be y19 ^ g(u0)
  bit20      s = 20, shift input u + a*(u | b): flipping u20 moves w by +-2^20 (1 + a*[b20 = 0]),
             so b20 = 0 and a is odd; grammar4_round0_bit0_863.BFW_C then finds no a, b, g
             with bit20(w) == y19 ^ g(u0) on all u < 2^21
  mask       s = 19, shift input aff(u & K) or aff(u | K): a bit j < 20 missing
             from K (u & K) or present in it (u | K) hides input bit j from w, but a
             flip witness changes y19 with that bit alone (for j = 0, y19(u) ^ y19(u^1)
             also takes both values, so g cannot absorb it). So u & K = u and
             u | K = u mod 2^20, and bit 19 of w is affine: the affine lemma.

A template whose u feeds only an AND/OR with a constant is closed whole
(injectivity, see grammar4_round0_pinned863). A template is closed when all
its cases are.

Multi-shift templates: every assignment of shift amounts is a case, closed by
  dependency  the bit-0 dependency rule of grammar4_round0_audit863
  image       the output is a unary function of a value with fewer than 2^30 values
              (a shift output), so it cannot be injective on v < 2^30
  rigid       rule 5 of grammar4_round0_bit0_863 (rigid moves, any shift amounts)
  reduce      shifts by 0 are identities, and a shift of a shift output used nowhere
              else is one shift by the sum (0 if the sum is 32 or more, which leaves
              too few shifts); if one shift remains and it reads a bitwise or affine
              value, the bit-0 lemma of grammar4_round0_863 applies
  two-read    i1 = aff(u), the shifts read u (by s1) and i1 (by s2 <= 29), and only the
              last instruction uses them, so bit 0 of the output is Psi(u0, u_s1, L_s2).
              With L = 2^d * L' (L' odd-affine) and q = s2 - d, for every d:
              q < 0 or mu = 0: L_s2 is constant; y19 = Psi(u0, u_s1) contradicts the
                bitwise witnesses (inputs agreeing on bits 0 and s1, y19 differing);
              q in 20..29, q != s1: flipping u_q flips only L'_q, so Psi ignores it;
              q = s1 in 20..29: flipping u_q flips both, so y19 is a function of u0
                and bit q of u ^ L': the xor-aff rule (q >= 21) or the z3 lemma (q = 20);
              q < 19: flipping u19 flips y19, so s1 = 19 and y19 ^ u19 = Psi'(u0, L'_q),
                which witnesses (q < 18) or the exact P18 check (q = 18) exclude;
              q = 19: s1 >= 20 or s1 = 0 leaves the affine lemma; for s1 = 1..18 flipping
                u19 gives bit19(L') = y19 ^ G(u0, u_s1), the exact G1 check (no
                solutions); for s1 = 19, Psi is either constant on a class of (u0, u19),
                which two inputs of that class with different y19 exclude, or it gives
                the G1 check with t = 19.
              A Psi that ignores its last argument always leaves y19 a function of bits
              the witnesses above keep fixed.
  two-read    (second form) i1 = aff(u) and both shifts read i1, directly or one through
              the other (bit 0 then reads L at s1 and s1 + s2), only the last instruction
              uses them, and both read positions are <= 29. For every d, with p < q the
              positions in L': q < 19 is refuted by flipping u19; q in 20..29 by flipping
              u_q (only L'_q moves), which leaves one read; q = 19 needs
              bit19(L') = y19 ^ G(u0, bit_p(L')), which grammar4_round0_bit0_863.SUB_C
              refutes for every p = 0..18 (G2). One read is the affine bit-0 lemma.
  valuation-set  the possible 2-adic valuations of the output's move under a flip of
              input bit j (j = 20..29), tracked as sets through every instruction for all v
              and all constants, miss j - 19, the target's exact valuation
  solver      the pinned z3 run is UNSAT
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

from codex import grammar3_head863 as g
from codex import grammar4_round0_863 as h
from codex import grammar4_round0_pinned863 as q
from codex.grammar4_round0_audit863 import flip_witnesses
from codex import grammar4_round0_bit0_863 as b0

TWO_CONST_SUBS = {
    (('xk', (0,)), ('lin2', (0, 1))): 'xk_lin2',
    (('andk', (0,)), ('lin2', (0, 1))): 'andk_lin2',
    (('ork', (0,)), ('lin2', (0, 1))): 'ork_lin2',
}
TWO_CONST_SUBS_INV = {v: k for k, v in TWO_CONST_SUBS.items()}
QUAD_SUB = (('aff', (0,)), ('prod', (0, 1)))
MASK_SUBS = {(('andk', (0,)), ('aff', (1,))), (('ork', (0,)), ('aff', (1,)))}
XOR_AFF_SUB = (('aff', (0,)), ('xor', (0, 1)))


def mask_rule_holds(root):
    """The facts the mask rule uses, checked for this root."""
    y19 = h.y19_of(root)
    wit = flip_witnesses(root)
    both = {y19(u) ^ y19(u ^ 1) for u in range(1 << 12)}
    return len(wit) == 20 and both == {0, 1} and h.affine_refuted(root) == ()


def valuation_bound(p):
    """Smallest s that rule 'valuation' excludes for single-shift template p, or None."""
    sh = q.shifts(p)
    if len(sh) != 1 or sh[0] != len(p) - 2:
        return None
    f, args = p[-1]
    k = sh[0] + 1
    if args.count(k) != 1:
        return None
    if f in b0.RIGID_LAST:
        return 23
    if f in ('xor', 'or', 'and'):
        return 22
    return None


def exact_rule(p, s, bit0, mask_ok, witnesses):
    """Name of the exact rule that closes shift amount s of single-shift template p, or None."""
    sub = tuple(h.shift_input(p, q.shifts(p)[0]))
    vb = valuation_bound(p)
    if vb is not None and s >= vb:
        return 'valuation'
    if s in witnesses and b0.rigid_shape(p):
        return 'rigid'
    if s == 19 and sub in TWO_CONST_SUBS and bit0.get(TWO_CONST_SUBS[sub]) == []:
        return 'two-const'
    if s <= 29 and sub == QUAD_SUB and bit0.get('aff_prod') == []:
        return 'quad'
    if s == 19 and sub in MASK_SUBS and mask_ok:
        return 'mask'
    if s == 20 and sub == TWO_CONST_SUBS_INV['ork_lin2'] and bit0.get('ork_lin2_b20') == []:
        return 'bit20'
    if 21 <= s <= 29 and sub == XOR_AFF_SUB:
        return 'xor-aff'
    return None


def drop_zero_shifts(p, amounts):
    """Remove shifts by 0 (identities); returns the re-indexed program and amounts."""
    remap, out, am = {0: 0}, [], {}
    for k, (f, args) in enumerate(p):
        if f == 'shrk' and amounts[k] == 0:
            remap[k + 1] = remap[args[0]]
            continue
        out.append((f, tuple(remap[a] for a in args)))
        if f == 'shrk':
            am[len(out) - 1] = amounts[k]
        remap[k + 1] = len(out)
    return out, am


def merge_nested_shifts(p, amounts):
    """Fold x >> a >> b into x >> (a + b) when x >> a has no other use; None if a + b >= 32."""
    p, amounts = list(p), dict(amounts)
    while True:
        uses = {}
        for k, (_, args) in enumerate(p):
            for a in args:
                uses.setdefault(a, []).append(k)
        for k, (f, (a,)) in ((k, ins) for k, ins in enumerate(p) if ins[0] == 'shrk'):
            if a and p[a - 1][0] == 'shrk' and uses.get(a) == [k]:
                total = amounts[a - 1] + amounts[k]
                if total >= 32:
                    return None
                (src,) = p[a - 1][1]
                p[k] = ('shrk', (src,))
                amounts[k] = total
                # drop instruction a - 1 and re-index
                remap = {i: (i if i < a else i - 1) for i in range(len(p) + 1)}
                p = [(g_, tuple(remap[x] for x in args_)) for n, (g_, args_) in enumerate(p) if n != a - 1]
                amounts = {(n if n < a - 1 else n - 1): v for n, v in amounts.items() if n != a - 1}
                break
        else:
            return p, amounts


def image_bits(p, amounts):
    """Upper bound on log2 of the number of values each instruction takes (u: 30 bits)."""
    bits, width = [30], [30]
    for k, (f, args) in enumerate(p):
        if f == 'shrk':
            (a,) = args
            width.append(max(width[a] - amounts[k], 0))
            bits.append(min(bits[a], width[-1]))
        elif len(args) == 1:
            width.append(32)
            bits.append(bits[args[0]])
        else:
            width.append(32)
            bits.append(min(32, sum(bits[a] for a in args)))
    return bits


def two_read_shape(p):
    """(k_u, k_l): shift of u and shift of i1 = aff(u), used only by the last instruction."""
    if p[0] != ('aff', (0,)):
        return None
    sh = q.shifts(p)
    if len(sh) != 2 or len(p) != 4:
        return None
    ku = next((k for k in sh if p[k][1] == (0,)), None)
    kl = next((k for k in sh if p[k][1] == (1,)), None)
    if ku is None or kl is None:
        return None
    users = lambda v: [k for k, (_, args) in enumerate(p) if v in args]
    if users(ku + 1) != [3] or users(kl + 1) != [3]:
        return None
    return ku, kl


def two_read_l_positions(p, amounts):
    """Read positions in L = i1 = aff(u) of bit 0 of the output, if the template has that shape."""
    if p[0] != ('aff', (0,)) or len(p) != 4:
        return None
    sh = q.shifts(p)
    if len(sh) != 2:
        return None
    users = lambda v: [k for k, (_, args) in enumerate(p) if v in args]
    pos = {}
    for k in sh:
        (a,) = p[k][1]
        if a == 1:
            pos[k] = amounts[k]
        elif a - 1 in sh and p[a - 1][1] == (1,):
            pos[k] = amounts[a - 1] + amounts[k]       # x >> s1 >> s2 reads bit s1 + s2
        else:
            return None
    for k in sh:
        if any(u not in sh and u != 3 for u in users(k + 1)):
            return None
    return tuple(pos[k] for k in sh)


def psi_witnesses(root):
    """For q = 0..17: inputs equal mod 2^(q+1) whose y19 ^ u19 differ."""
    import random
    y19 = h.y19_of(root)
    psi = lambda u: y19(u) ^ (u >> 19 & 1)
    rng, out = random.Random(17), {}
    for qq in range(18):
        m = 1 << (qq + 1)
        while qq not in out:
            u = rng.randrange(1 << 20)
            v = (u & (m - 1)) | (rng.randrange(1 << 20) & ~(m - 1))
            if psi(u) != psi(v):
                out[qq] = (u, v)
    return out


def class_witnesses(root):
    """For each value of (u0, u19), two inputs with those bits and different y19."""
    y19 = h.y19_of(root)
    out = {}
    for u in range(1 << 20):
        c = (u & 1, u >> 19 & 1)
        out.setdefault(c, {}).setdefault(y19(u), u)
        if len(out) == 4 and all(len(v) == 2 for v in out.values()):
            break
    return {c: (v[0], v[1]) for c, v in out.items() if len(v) == 2}


def two_read_facts(root, lemma_cache):
    """Every exact fact rule two-read relies on, checked for this root."""
    return (h.bitwise_refuted(root) and h.affine_refuted(root) == () and len(class_witnesses(root)) == 4 and
            b0.pair_read_solutions(root) == () and len(psi_witnesses(root)) == 18 and
            b0.g2_refuted(root) and
            lemma_cache.get(f"{g.canonical([('aff', (0,)), ('xor', (0, 1))])}@20", {}).get('verdict') == 'unsat')


# Rule valuation-set (sets of possible valuations 0..32 of each value's move; _INF = no move).
_INF = 64
_FULL = frozenset(range(0, 33)) | {_INF}

def _vge(a):
    return frozenset(range(a, 33)) | {_INF} if a <= 32 else frozenset({_INF})

def _vsum(A, B):
    out = set()
    for a in A:
        for b in B:
            if a == _INF: out.add(b)
            elif b == _INF: out.add(a)
            elif a != b: out.add(min(a, b))
            else: out |= _vge(a)
    return frozenset(out)

def _vscale(A):            # multiply the move by an unknown value (constant or T-function)
    out = set()
    for a in A:
        out |= _vge(a) if a != _INF else {_INF}
    return frozenset(out)

def move_valuations(p, amounts, j):
    """state per value: (kind, V) with kind 'T' (T-function of v: move valuations >= j, value
    low-determined) or 'S' (shift-derived, bounded by width w)."""
    st = [('T', frozenset({j}), 30)]          # u moves by exactly +-2^j; 30-bit value
    for k, (f, args) in enumerate(p):
        ops = [st[a] for a in args]
        if f == 'shrk':
            s = amounts[k]
            kind, V, w = ops[0]
            if s >= w:
                st.append(('T', frozenset({_INF}), 0)); continue
            if s == 0:
                st.append(ops[0]); continue
            nw = w - s
            # move of A >> s: if every valuation of A's move is >= s, the shifted move is exact:
            # valuations shift down by s, but the value has only nw bits (|move| < 2^nw)
            out = set()
            for a in V:
                if a == _INF: out.add(_INF)
                elif a >= s:
                    if a - s < nw: out.add(a - s)
                    else: out.add(_INF)        # move is a multiple of 2^nw within nw bits: zero
                else:
                    out |= frozenset(range(0, nw)) | {_INF}
            st.append(('S', frozenset(out), nw)); continue
        if all(o[0] == 'T' for o in ops):
            V = _vge(j)
            if f in ('xk', 'andk', 'ork') and ops[0][1] == frozenset({j}):
                V = frozenset({j, _INF})
            if f == 'aff':
                V = _vscale(ops[0][1])
            st.append(('T', V, 32)); continue
        # mixed: only forms affine in the shift-derived operands with T coefficients
        if f in ('xor', 'or', 'and', 'xk', 'andk', 'ork'):
            # lowest changed bit is at least the lowest operand change
            m = min(min(o[1]) for o in ops)
            st.append(('S', frozenset(range(m, 33)) | {_INF}, 32)); continue
        if f == 'aff':
            st.append(('S', _vscale(ops[0][1]), 32)); continue
        if f == 'lin1':     # K*a + b
            st.append(('S', _vsum(_vscale(ops[0][1]), ops[1][1]), 32)); continue
        if f == 'lin2':     # a + K*b
            st.append(('S', _vsum(ops[0][1], _vscale(ops[1][1])), 32)); continue
        if f in ('prod', 'mad_c', 'mad_b', 'mad_a'):
            if f == 'prod':
                (x, y), add = ops, None
            else:
                a, b, c = ops
                (x, y), add = {'mad_c': ((a, b), c), 'mad_b': ((a, c), b), 'mad_a': ((b, c), a)}[f]
            # d(xy) = x' dy + y dx (coefficients unknown) ; products of two shift-derived values allowed
            V = _vsum(_vscale(y[1]), _vscale(x[1]))
            if add is not None:
                V = _vsum(V, add[1])
            st.append(('S', V, 32)); continue
        st.append(('S', _FULL, 32))
    return st[-1][1]

def valuation_set_refuted(p, amounts):
    for j in range(20, 30):
        if (j - 19) not in move_valuations(p, amounts, j):
            return j
    return None


def multi_rule(p, amounts, root, two_read_ok=False):
    """Exact rule closing one shift assignment of a multi-shift template, or None."""
    from codex.grammar4_round0_audit863 import bit0_support
    sup = bit0_support(p, amounts)
    if sup is None or not frozenset(range(20)) <= sup:
        return 'dependency'
    if image_bits(p, amounts)[-1] < 30:
        return 'image'
    if b0.rigid_any(p, amounts, root):
        return 'rigid'
    tr = two_read_shape(p)
    if two_read_ok and tr and amounts[tr[1]] <= 29:
        return 'two-read'
    pos = two_read_l_positions(p, amounts)
    if two_read_ok and pos and max(pos) <= 29:
        return 'two-read'
    if valuation_set_refuted(p, amounts):
        return 'valuation-set'
    r, am = drop_zero_shifts(p, amounts)
    merged = merge_nested_shifts(r, am)
    if merged:
        r, am = merged
        if len(am) == 1 and h.sub_class(h.shift_input(r, next(iter(am)))) in ('affine', 'bitwise'):
            return 'reduce'
    return None


def multi_cases(root, two_read_ok=False):
    """{(idx, amounts tuple): rule or None} for every open multi-shift template."""
    import itertools
    ts = h.templates4()
    out = {}
    for idx in q.open_templates():
        p = ts[idx]
        sh = q.shifts(p)
        if len(sh) < 2:
            continue
        for combo in itertools.product(range(32), repeat=len(sh)):
            out[(idx, combo)] = multi_rule(p, dict(zip(sh, combo)), root, two_read_ok)
    return out


def lemma_job(args):
    key, sub, s, root, seconds = args
    t0 = time.monotonic()
    return key, s, h.bit0_width(list(sub), root, s, seconds), round(time.monotonic() - t0, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('inputs', nargs='+', help='pinned-run JSONL files')
    ap.add_argument('--bit0', type=Path, required=True, help='grammar4_round0_bit0_863 output')
    ap.add_argument('--lemma-cache', type=Path, required=True)
    ap.add_argument('--lemma-seconds', type=float, default=120)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--todo', type=Path, help='write the idx:s cases still needing the solver, then stop')
    ap.add_argument('--multi-todo', type=Path, help='write the multi-shift combos needing the solver, then stop')
    args = ap.parse_args()
    root = h.ROOTS[0]
    ts = h.templates4()
    cache = json.loads(args.lemma_cache.read_text()) if args.lemma_cache.exists() else {}
    two_read_ok = two_read_facts(root, cache)
    if args.multi_todo:
        mc = multi_cases(root, two_read_ok)
        todo = {}
        for (idx, combo), rule in sorted(mc.items()):
            if rule is None:
                todo.setdefault(str(idx), []).append(list(combo))
        args.multi_todo.write_text(json.dumps(todo) + '\n')
        tally = {}
        for rule in mc.values():
            tally[rule or 'solver'] = tally.get(rule or 'solver', 0) + 1
        print(json.dumps(dict(multi_cases=len(mc), by_rule=tally)))
        return
    bit0 = json.loads(args.bit0.read_text())
    runs = {}
    for path in args.inputs:
        if not Path(path).exists():
            continue
        for line in open(path):
            r = json.loads(line)
            if r['result'] == 'sat':
                raise SystemExit(f'SAT found: {r}')
            amounts = tuple(s for _, s in sorted((int(k), s) for k, s in r['amounts'].items()))
            key = (r['idx'], amounts)
            if key not in runs or r['result'] == 'unsat':
                runs[key] = r
    solved = lambda idx, amounts: runs.get((idx, amounts), {}).get('result') == 'unsat'
    mask_ok = mask_rule_holds(root)
    witnesses = b0.rigid_witnesses(root)
    cases, whole = {}, {}
    for idx in q.open_templates():
        p = ts[idx]
        if q.single_use_mask(p):
            whole[idx] = 'mask-single-use'
            continue
        sh = q.shifts(p)
        if len(sh) != 1:
            continue
        sub = tuple(h.shift_input(p, sh[0]))
        for s in range(19, 32):
            how = exact_rule(p, s, bit0, mask_ok, witnesses)
            if how is None and solved(idx, (s,)):
                how = 'solver'
            cases[(idx, s)] = (how, sub)
    lemma_unsat = lambda sub, s: cache.get(f'{g.canonical(list(sub))}@{s}', {}).get('verdict') == 'unsat'
    if args.todo:
        todo = [f'{i}:{s}' for (i, s), (how, sub) in sorted(cases.items())
                if how is None and not lemma_unsat(sub, s)]
        args.todo.write_text(','.join(todo) + '\n')
        print(json.dumps(dict(solver_todo=len(todo))))
        return
    need = {}
    for (idx, s), (how, sub) in cases.items():
        key = f'{g.canonical(list(sub))}@{s}'
        if how is None and key not in cache:
            need[key] = (key, sub, s, root, args.lemma_seconds)
    print(json.dumps(dict(cases=len(cases), lemma_todo=len(need))), flush=True)
    if need:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for f in as_completed([pool.submit(lemma_job, j) for j in need.values()]):
                key, s, verdict, secs = f.result()
                cache[key] = dict(verdict=verdict, seconds=secs)
                args.lemma_cache.write_text(json.dumps(cache, indent=0, sort_keys=True) + '\n')
                print(json.dumps(dict(key=key, verdict=verdict, seconds=secs)), flush=True)
    open_cases, tally = [], {}
    for (idx, s), (how, sub) in sorted(cases.items()):
        if how is None:
            key = f'{g.canonical(list(sub))}@{s}'
            how = 'lemma' if cache.get(key, {}).get('verdict') == 'unsat' else None
        tally[how or 'open'] = tally.get(how or 'open', 0) + 1
        if how is None:
            open_cases.append([idx, [s]])
    multi_tally = {}
    for (idx, combo), rule in sorted(multi_cases(root, two_read_ok).items()):
        if rule is None and solved(idx, combo):
            rule = 'solver'
        multi_tally[rule or 'open'] = multi_tally.get(rule or 'open', 0) + 1
        if rule is None:
            open_cases.append([idx, list(combo)])
    open_templates = sorted({i for i, _ in open_cases})
    summary = dict(templates=len(q.open_templates()), mask_single_use=len(whole),
                   single_shift_cases=len(cases), single_cases_by_rule=tally,
                   multi_shift_cases=sum(multi_tally.values()), multi_cases_by_rule=multi_tally,
                   templates_open=len(open_templates), cases_open=len(open_cases),
                   mask_rule_facts=mask_ok, two_read_facts=two_read_ok)
    print(json.dumps(summary), flush=True)
    out = dict(summary=summary, open_templates={i: g.describe(ts[i]) for i in open_templates},
               open_cases=open_cases)
    args.output.write_text(json.dumps(out, separators=(',', ':')) + '\n')


if __name__ == '__main__':
    main()
