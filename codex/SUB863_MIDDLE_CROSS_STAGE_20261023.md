# Cross-stage middle-hash search

The verified executable remains **864 cycles**. This research produces no
faster kernel. It closes a previously unresolved family: with one uint32
input and arbitrary preloaded constants, the following middle hash cannot
be computed by exactly two MAD instructions and one XOR in any order:

```
F(x) = 9*((33*x + 0xe9f8cc1d) XOR (16896*x + 0xaccf6200)) + 0xfd7046c5
```

Arithmetic wraps modulo 2^32. Current lowering uses MAD, MAD, XOR, MAD.
This result does not cover other opcodes, memory, additional live inputs,
or joint computation with another hash stage or address. It is not a global
864-cycle lower bound or a lower bound for every three-instruction grammar.

## MAD, MAD, XOR

Every single-input first MAD is affine, x*x+K, or x*x+x. For affine first
MAD, the second result is quadratic; the other final XOR operand is affine.
Relax the program to `(a*x*x+b*x+c) XOR (d*x+e)`. For fixed d,e, let
G(x)=F(x) XOR (d*x+e). Inputs 0,1,2 determine:

```
c = G(0)
delta = G(1)-G(0)
2*a = G(2)-G(0)-2*delta
b = delta-a
```

An odd right side is impossible. Otherwise the two possible a values differ
by 2^(w-1), and their b values by its negative. The corresponding functions
are identical because 2^(w-1)*x*(x-1) vanishes modulo 2^w.
`screen_quadratic_xor_affine863.py` therefore needs only one quadratic per d,e.
All **4,194,304** low-11-bit pairs are rejected. Every retained candidate
would be checked on the entire 2,048-input domain, not just the initial witnesses.

For nonlinear first MAD, relax its second result to an arbitrary degree-four
polynomial. The other XOR operand is x, a constant, x*x+K, or x*x+x.
`screen_square_mad_xor863.py` checks all **4,098** cases using modular linear
consistency on 73 necessary inputs. Every case is inconsistent. Independent
polynomial coefficients make this a superset of realizable second MADs.
Cases where the second result is unused are also contained in these relaxations.

## MAD, XOR, MAD

With affine first result t=A*x+B, a relevant second XOR is t XOR x or t XOR K.
The final operand pool is x,t,u,k0,k1. There are 35 commutative canonical
operand triples that use u, hence **70** cases in `search_middle_outer_roles863.py`.
At 11 bits and 103 necessary inputs, 64 are UNSAT and six initially time out.
Constant renaming reduces the six to three cases:

```
t*k + (t XOR K)
(t XOR K)*k + t
(t XOR K)*k + c
```

F is a permutation modulo every power of two: 33*x+B has an invertible bit
diagonal, while 16896*x+D has its variable part shifted left by nine bits;
XOR preserves that diagonal, and the final multiplier 9 is odd. Since all
three remaining forms depend only on t, A must be odd. Adding that necessary
condition proves the first two cases UNSAT; the third still times out.

For the third case, write F=D*((A*x+B) XOR K)+C. Both A and D must be odd.
The low nine bits of F are exactly `297*x+458`. Inverting both affine maps
implies that XOR K must itself be affine modulo 512. Evaluating an affine
XOR map at 0,1,2,4,... shows that K's low eight bits must all be identical:
its slope at 1 is +1 or -1, and every power of two below the top bit must
have the same sign. Complementing K, negating D, and changing C normalizes
these eight bits to zero. Flipping K's top bit can also be absorbed in C.
At width 11 only K in **{0,256,512,768}** remains.

For each odd A, arbitrary B, and normalized K, outputs at 0 and 1 uniquely
determine D and C: the difference of the two XOR values is odd and invertible.
`screen_affine_xor_affine863.py` checks all **8,388,608** normalized cases;
none survives. This replaces the inconclusive SMT result with enumeration.

The square-constant and square-input first-MAD forms were independently
rechecked with the generic operand-selector solver at width 12: both UNSAT.
If the XOR does not consume the first MAD, the program can be reordered to
XOR, MAD, MAD. If it cancels itself or its result is unused, the result is a
degree-four polynomial, also contained in the following relaxation at K=0.

## XOR, MAD, MAD

The existing `screen_xor_mad_polynomial.py --slice middle` relaxation was
reproduced: all 2,048 low-11-bit XOR keys are inconsistent on 125 inputs.
It permits all 15 monomials of total degree at most four in x and x XOR K,
which includes arbitrary SSA operand choices of the two subsequent MADs.

All these operations commute with low-bit projection, so exclusions at
11 or 12 bits also exclude the corresponding uint32 programs.

## Reproduction and validation

Install optional research dependencies from `requirements-synthesis.txt`.
The submission itself still requires only the Python standard library.

```bash
python3.12 -m codex.screen_quadratic_xor_affine863
python3.12 -m codex.screen_square_mad_xor863
python3.12 -m codex.search_middle_outer_roles863 --output /tmp/outer-roles.json
python3.12 -m codex.screen_affine_xor_affine863
python3.12 -m codex.screen_xor_mad_polynomial --width 11 --slice middle
python3.12 -m unittest codex.test_middle_cross_stage863 codex.test_modular_polynomial
```

Detailed results, timeouts, solver versions, and square-first commands are in
[`middle_cross_stage863_results.json`](middle_cross_stage863_results.json).
The refinement calls use `check('constant', role, timeout_ms=15000,
injective_cut=True)` for roles (1,3,2), (2,3,1), and (2,3,4).

**13 tests pass**, including exhaustive coefficient oracles at small widths,
normalization compared against unnormalized enumeration, positive controls
at the proof width, all XOR-affine keys through eight bits, modular-elimination
oracles, and the two negative exhaustive searches. Positive controls ensure
that the search can recover valid programs. No submission or frozen simulator
files changed; previous 864-cycle correctness results remain applicable.

The next useful search scope is other arithmetic opcodes or multiple live
inputs across stage boundaries. Increasing the timeout for these closed
two-MAD/one-XOR templates cannot yield a correct three-instruction replacement.
