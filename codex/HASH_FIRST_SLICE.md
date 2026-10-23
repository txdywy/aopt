# First hash slice synthesis, 2026-09-07

Target (uint32 arithmetic):

```
a = 4097*x + 0x7ed55d16
b = a ^ (a >> 19) ^ 0xc761c23c
```

The current implementation costs four VALU instructions (MAD, shift, two
XORs). Saving one instruction here would affect all512 SIMD hash invocations;
this is a substantially larger possible work reduction than table setup tweaks.

`synthesize_hash_cegis.py --slice first` now supports this target. Three fixed
three-instruction grammars returned **UNSAT** at full32-bit width:

- `mad,>>,^`
- `mad,^,>>`
- `>>,mad,^`

Each instruction may choose operands from the original input, prior results,
and two arbitrary per-instruction uint32 constants. The exclusions apply to
those exact grammars, not every three-instruction program. UNSAT on the six
concrete initial input constraints already excludes equivalence for all inputs.

Positive control: removing the final XOR constant (`--control`) with grammar
`mad,>>,^` synthesizes the expected affine MAD, shift19 and XOR. Its independent
counterexample query is UNSAT, proving equivalence for every uint32 input.
Five research tests pass using Z3 5.1.0, including both positive and negative
first-slice checks.

Unrestricted three-instruction grammar (+,-,^,&,|,<<,>>,* and MAD) with a
40-second synthesis-check timeout returned **unknown / timeout**. This is
inconclusive, not an impossibility result. No kernel rewrite or cycle improvement
was obtained in this experiment.

Reproduce a fixed grammar:

```sh
/tmp/aopt-smt/bin/python codex/synthesize_hash_cegis.py --slice first \
  --instructions 3 --op-sequence 'mad,>>,^' --seconds 40 --check-ms 35000
```

Append `--control` for the independently verified positive control. Full grammar:

```sh
/tmp/aopt-smt/bin/python codex/synthesize_hash_cegis.py --slice first \
  --instructions 3 --seconds 45 --check-ms 40000
```

## Fixed-sequence partition

`search_first_hash_sequences.py` enumerates all48 distinct permutations of
one MAD, one logical right shift, and one operation from the nine-op grammar.
At two seconds per synthesis query, 42 were UNSAT, four timed out, and two
leading-subtraction sequences initially failed argument parsing. The runner
now uses `--op-sequence=...` so leading minus signs are passed as values.
Rerunning these two with a14-second query limit gave one more UNSAT
(`-,mad,>>`) and one timeout (`-,>>,mad`). Thus the inspected partition has
**43 excluded sequences and five unresolved**, not48 exclusions.

Unresolved sequences: `*,>>,mad`, `+,>>,mad`, `-,>>,mad`, `>>,>>,mad`,
`mad,>>,mad`. No candidate has been promoted to an implementation.

The synthesizer now accepts `--extra-samples N` with deterministic seed0.
Additional samples constrain candidate generation only: positive equivalence
still requires an UNSAT full-width counterexample query. This option does not
alter the default search or kernel. Five research tests continue to pass.
All five unresolved sequences were retried with24 extra samples (30 total)
and11-second query limits; all still timed out. These retries add no exclusion.

## Symmetry reduction

`--canonical-operands` orders the first two operand selectors for commutative
operations (+,*,^,&,|,MAD) and fixes the unused third operand of all non-MAD
operations. This preserves represented functions: swapping the multiplicands
does not change MAD, and the third selector has no semantics for binary ops.
New tests prove these identities over uint32; six research tests pass, including
positive-control synthesis with canonicalization enabled.

Rechecking the five unresolved sequences with canonicalization, six samples
and8-second query limits produced two additional UNSAT results:
`-,>>,mad` and `>>,>>,mad`. Cumulative partition result is now **45 excluded,
three unresolved** (`*,>>,mad`, `+,>>,mad`, `mad,>>,mad`). No shorter equivalent
implementation has been found.

An optional `--right-shift-amount N` restricts all right shifts to a literal
in0..32; this deliberately narrows the grammar and is reported in the result.
N=19 with3-second query limits did not exclude any additional unresolved class.
Some already excluded sequences timed out under that different encoding and
shorter limit; this does not invalidate their earlier UNSAT results.

## Stopping the current solver tuning branch

Fixed-opcode formulas now substitute the opcode literal before constructing
the sample program. This preserves the grammar, avoiding symbolic branches
whose opcode is already fixed. The three unresolved cases still timed out
with15-second query limits.

`--high-bit-pairs` doubles initial samples with their top-bit-flipped partners.
For this first hash slice the exact identity is
`f(x ^ 0x80000000) == f(x) ^ 0x80001000`: multiplication by odd4097 preserves
the top-bit toggle, addition preserves it modulo2^32, and shift19 moves a copy
to bit12. An independent uint32 counterexample query proves the identity.
Seven research tests pass, including canonicalized synthesis with paired
samples and both positive/negative controls.

All three unresolved sequences still timed out with12 paired samples and
10-second query limits. No new exclusions and no equivalent implementation
resulted. Leave these three unknown, and deprioritize this particular solver
tuning branch rather than repeatedly extending identical timeouts. The overall
sub900 optimization objective remains open; these tests do not establish hash
optimality or any universal kernel lower bound.

## Consecutive-input and piecewise-polynomial screen

The three unresolved orders were retested at fixed shift19 using64 consecutive
inputs plus8 random samples,canonical operands and20s solver limits. All
three still timed out. Seven existing synthesis regression tests passed.

An independent algebraic check now closes `+,>>19,mad` without SMT:
`python -m codex.prove_first_add_shift`. On x=0..63 the first addition and
original x are affine, and their logical shift19 changes at most once,
including uint32 wrap. The final MAD must be quadratic on each of at most
two intervals. Checking third finite differences modulo2**32 rejects every
possible split of the64 target outputs. All65 positive two-piece quadratic
controls pass, including modular wrap and even coefficients.

This excludes arbitrary operand/constant selections for that fixed-shift
order only. The unrestricted-shift version remains unresolved, as do the
multiplication-first and MAD-first orders. No faster executable results.

## Addition-first order fully closed

The piecewise-polynomial proof now covers arbitrary right-shift operands,
not only literal19. For constant shifts1..6 use64 inputs spaced by2**shift:
all available quantities are affine in the sample index except at most one
wrap. For shifts7..31 the original consecutive-input proof applies. Shift0
is identity; shifts>=32 produce zero. All33 representative cases fail the
necessary two-piece quadratic condition.

For variable shift amounts, the only nonconstant possibilities after the
first addition are x,2*x,and x+K. Over x=64..255 the first two always shift
to zero. The third can yield a nonzero shift result only inside one interval
of at most32 inputs, including wrapping x+K. Outside that interval the final
MAD must be quadratic. All225 placements of a relaxed32-input exceptional
window fail. Constant shift amounts are covered separately above.

`python -m codex.prove_first_add_shift` passes290 positive controls and finds
no surviving case. Consequently `+,>>,mad` is excluded with all legal SSA
operand selections and arbitrary constants, without SMT timeouts. The48-order
first-slice partition is now46 excluded and two unresolved: `*,>>,mad` and
`mad,>>,mad`. No whole-hash lower bound or improved executable score follows.

## First-MAD operand partition

New `--first-mad-form` fixes the first instruction to one of three forms:
`A*x+B`, `x*x+C`, or `x*x+x`. These cover every one-input MAD with constant
operands; the added unit test checks all27 selections from x,k0,k1 with
full32 equivalence queries. Eight synthesis tests pass.

For `mad,>>,mad`,full32,canonical operands,32 consecutive inputs plus their
high-bit partners,and15s solver limit: square-constant and square-input are
UNSAT; affine is UNKNOWN/timeout. Unlike the earlier fixed-shift screens,
the shift operands here are unrestricted. A multiplication-only first op is
either affine (B=0),square-constant(C=0),or constant, so its unresolved cases
are contained in the affine branch as well. Both opcode orders remain
globally unresolved, but further searches need only consider first A*x+B.
No shorter equivalent candidate or faster kernel was produced.

## Fixed multiplier and bounded-affine screen

New optional `--first-multiplier` constrains A in `--first-mad-form affine`
while keeping B arbitrary. Full32 `mad,>>,mad`,32 consecutive samples plus
high-bit pairs (70 total),canonical operands,15s solver limit:

- A=4097,unrestricted shift operands: UNKNOWN/timeout.
- A=-4097,unrestricted shift operands: UNSAT.
- A=4097,literal shift19: UNSAT.

Positive control first-slice-without-C1 (`mad,>>,^`,A=4097) synthesized the
known correct expression and passed full32 counterexample verification.
Eight synthesis regression tests pass.

The polynomial argument also excludes a continuous family independently of
SMT: for literal shift s>=6 and signed |A|<=floor((2**s-1)/63), first A*x+B
traverses less than one shift-bin width on x=0..63. Arbitrary B and uint32
wrap still yield at most one boundary. Shifted x is constant; the last MAD
must be piecewise quadratic, already refuted by the64 target outputs.
At s=18 this covers|A|<=4161; at s=19 it covers|A|<=8322, both including
the original multiplier4097. The proof script reports bounds for s=6..31.
Other multipliers/shift configurations remain unresolved. No kernel changed.

## Literal-19 affine branch closed

New bounded run: first slice, three instructions `mad,>>,mad`, first MAD
affine with arbitrary multiplier and offset, literal right shift19, canonical
operands, high-bit pairs,64 extra samples,20 seconds overall /15 seconds per
check. The first synthesis query with140 full32 samples returned UNSAT.
Unlike the earlier fixed-A4097 check, this excludes every first multiplier
in this configured two-constant grammar. It does not exclude extra constants,
other shift amounts, or richer available inputs. Unrestricted-shift branches
remain open. No executable candidate and no change in full-kernel cycles.

Adjacent/aligned literal-shift screen with the same140 samples and arbitrary
affine first MAD (12 seconds total /10 seconds per check): literal16 UNSAT,
literal18 UNSAT, literal20 UNKNOWN due to timeout. The closed literal set
for this specific grammar now includes16,18,19. Do not interpret the20 result
as an exclusion or spend another run on unchanged constraints.

Boundary-driven check: `--first-boundaries` adds modular preimages of
first-affine outputs one below/at/one above 15 chosen multiples of2^19.
The inverse4097 exists modulo2^32; an assertion checks those low19 bits.
With high-bit pairing this produces224 samples. Literal20 remains UNKNOWN
at12 seconds/check. Eight existing synthesis regression tests still pass.
No candidate; stop this branch until a new structural restriction is derived.

Scope clarification: the synthesizer allocates two arbitrary constants PER
instruction, not two shared constants for the entire program. Earlier wording
"two-constant grammar" refers to that per-instruction operand pool. The new
boundary flag defaults off and leaves previous searches unchanged.

Injectivity pruning (`--first-injective-cut`, default off): for first-slice
MAD/positive-literal-SHR/MAD, require final MAD to consume original x or the
unshifted first result. Otherwise all its variable inputs come from a word
with at most2^(32-s) distinct values, contradicting the bijective target.
The flag rejects incompatible configurations rather than applying the lemma
to other slices. Literal20 with224 boundary/high-bit samples still times out
at12 seconds. Eight existing regressions pass; no new kernel candidate.

Literal20 structural splits now tested separately via default-off
`--last-mad-operands`. Let x be original input, a first affine MAD, and
b the result of the literal20 shift (its source remains selected by grammar).
Final forms `b*K+a`, `x*K+b`, and `a*K+b` each returned UNSAT with224
samples and6 seconds/check. This narrows the UNKNOWN aggregate case without
claiming all other final operand combinations closed. Eight regressions pass.
The flag validates final opcode/operand range and does not alter default search.

Complete final-operand census (`screen_first_mad_shapes.py`):75 commutative
MAD shapes, of which40 bypass SHR and are at most quadratic,12 see only the
shifted variable and constants and violate cardinality,15 return UNSAT,8 time
out at2 seconds/check. The exact target third difference at x=0 is0xffff7fe8,
checked in the script, certifying the40 polynomial exclusions.

Timeout shapes: (1,2,1),(1,2,2),(1,2,3),(1,2,4),(2,3,0),(2,3,1),
(2,4,0),(2,4,1), with pool[x,a,b,k0,k1]. Prior6-second evidence already
closes(2,3,1); the shorter run does not invalidate that. This is a finite
structural inventory, not an all-shapes impossibility result or cycle gain.

## Four nonlinear timeout shapes closed by parity counting

For literal20, remaining forms a*b+a, a*b+b, a*b+k0, a*b+k1 have no direct
x operand. If b shifts x, then x=0..3 gives b=0 and a quadratic/affine result,
contradicted by the third-difference witness. If b shifts a constant, the same
polynomial obstruction applies. Thus only b=a>>20 needs consideration.

If a=A*x+B has even A, all later computation depends on a and is noninjective.
If A is odd, a ranges over every32-bit word. Its bits0 and20 are independent.
Parity of a*b+a is u*v XOR u; parity of a*b+b is u*v XOR v; parity of a*b+C
is u*v XOR (C&1). Each has one or three odd outcomes among four, rather than
the two required for a bijection. Therefore all four forms are impossible.

`prove_first_product_parity.py` verifies the count exhaustively for66 smaller
word/shift combinations; the bit-independence argument covers full32 shift20
without exhaustive2^32 execution. This closes shapes(1,2,1..4), not every
remaining linear combination. In particular b*K+x remains outside this proof.

## Literal20 final remaining forms: image cardinality

`prove_first_shift_cardinality.py` evaluates8192 deterministic uint32 inputs
and observes8192 distinct `(H(x)-x) mod2^32`. For any uint32 function y(x),
logical y>>20 takes at most4096 values; multiplication by any constant K
cannot increase that count. Therefore H(x)=x+K*(y(x)>>20) is impossible.
This exact finite witness closes(2,3,0) and(2,4,0), and also applies to larger
literal shifts. Random generation chooses the witness only; the proof is the
explicitly computed distinct-value count, not a statistical inference.

Combined with the earlier UNSAT for(2,3,1), constant-name symmetry closes
(2,4,1): both mean b*K+a with a single arbitrary final-step constant. Together
with polynomial/cardinality pruning,15 SMT exclusions, and four parity-count
exclusions, all75 final MAD shapes for affine-MAD / literal-SHR20 / MAD are
now excluded in this grammar. Other literal or variable shifts remain outside
this combined conclusion. No faster hash or full-kernel score resulted.

## General a+K*(a>>s) obstruction

`prove_affine_shift_permutation.py` proves the affine-source variant for all
1<=s<32, rather than just literal20. With even first multiplier A the map
already loses information. With odd A, a ranges over every word. Each block
of2^s consecutive a values maps to an interval of the same length starting at
b*(2^s+K). A bijection must tile the cyclic word space with these equal-length
intervals, forcing starts congruent modulo2^s; the first starts at0, so K is
divisible by2^s. The output then equals a modulo2^s and is affine in x there.

Target H violates this for every s: for s<=19, x=0 and x=2^19 are congruent
but H differs in its low bit; for s>=20 its second difference on0,1,2 is12,
not zero modulo2^s. The script asserts both witnesses and exhaustively checks
the interval permutation criterion on3076 smaller word/coefficient cases.
This proof requires the shifted source to be a itself; it does not silently
cover b=x>>s or arbitrary nonlinear shifted sources. No full-kernel gain.

## 2026-10-23: remaining affine branch fully closed

`prove_first_affine_shift863.py` closes affine MAD / arbitrary logical right
shift / MAD. It partitions all 75 final operand shapes, excludes even first
multipliers even when the final MAD directly uses x, and handles every literal
shift using parity, interval, sum and finite-difference certificates. For a
variable affine shift count, an arbitrary-quadratic necessary relaxation is
UNSAT on 128 full32 inputs; constructive unpinned controls are SAT.

Both non-affine first-MAD branches were independently reconfirmed UNSAT at
full32 with unrestricted shift operands, in `first_square_shift863_results.json`.
Thus MAD/SHR/MAD and its multiplication-first subfamily are closed. The earlier
48-order partition now has no unresolved order, within its stated one-input
grammar. This does not prove optimality over all three-instruction ISA programs.
See [the complete argument](SUB863_LIVE_AND_SHIFT_20261023.md) and
`first_affine_shift863_results.json`. The verified kernel remains 864 cycles.
