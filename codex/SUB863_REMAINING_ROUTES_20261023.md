# Remaining routes below 864

**Later continuation:** the live-input experiment below has now been completed
by structural partitions. All 64 unshifted three-op orders are excluded even
with the specified extra first-stage intermediates; the two remaining first-slice
MAD/SHR/MAD-related orders are also closed. See
[new proofs, evidence and remaining scope](SUB863_LIVE_AND_SHIFT_20261023.md).
The original timeout results below remain a record of that earlier experiment,
not the current closure status. The executable best is still 864.

The verified submission is still **864 cycles**. This continuation found no
replacement program and makes no global optimality claim. It adds a complete
bounded-instruction result for an isolated hash slice and a distinct experiment
with additional live inputs.

## Middle hash: four instructions are optimal in the unshifted grammar

For one input x, define the uint32 function

```
F(x) = 9*((33*x+0xe9f8cc1d) XOR (16896*x+0xaccf6200))+0xfd7046c5
```

The current lowering is MAD,MAD,XOR,MAD. Every length-three order over
`{MAD,XOR,AND,OR}` is now excluded with arbitrary SSA operand choices and
preloaded constants. The 64 orders split as follows:

| Orders | Count | Evidence |
|---|---:|---|
| Three MADs | 1 | Integer-polynomial mixed-difference certificate |
| Two MADs and one XOR | 3 | Previous cross-stage audit and exhaustive searches |
| Two MADs and one AND/OR | 6 | New operand domination, polynomial relaxation, split SMT |
| One MAD and two bitwise instructions | 27 | New low-12-bit fixed-order UNSAT checks |
| Pure bitwise | 27 | Nonlocal output-bit witness |

Addition, subtraction and multiplication are expressible by one MAD, so
adding those opcodes does not invalidate the three-instruction exclusion.
Programs with fewer instructions can be padded with XOR zero. Consequently
four is optimal **for this function, one input, and this instruction set**.
This does not prove that the entire hash or the 864-cycle kernel is optimal.
Shifts, division, comparisons, FLOW, memory and other live inputs remain
outside the result.

Detailed evidence is in [`middle_bitwise_closure863_results.json`](middle_bitwise_closure863_results.json),
with the earlier XOR cases in `middle_cross_stage863_results.json`.

### Final AND/OR: necessary operand domination

If `Q AND P=F`, P must contain every set bit of F. If `Q OR P=F`, P cannot
contain a bit absent from F. Apart from the second MAD result, the other
operand is an original input, a constant, or the first MAD. The complete
first-MAD partition is affine, x*x+K, and x*x+x.

At 11 bits, `screen_middle_lattice863.py` exhausts 4,194,304 affine pairs
per opcode, all 2,048 square constants, and the square-input form. Only
the neutral constant survives: all ones for AND, zero for OR. That leaves
F equal to the second MAD's degree-at-most-four polynomial, which is impossible.
Repeated final operands are also polynomial and excluded below.

### Initial AND/OR: polynomial relaxation

`screen_xor_mad_polynomial.py` now also accepts `--bitwise '&'` and `'|'`.
For every one of the 2,048 keys, its necessary degree-four polynomial
system in x and x OP K is inconsistent on 125 inputs at 11 bits. Arbitrary
coefficients make this a superset of all two-MAD continuations. Constants
and identity first operations are contained in the polynomial cases too.

### Middle AND/OR: complete operand splits

For first affine t=A*x+B, the relevant bitwise result is t OP x or t OP K.
The final MAD has 35 canonical operand roles that use that result.
All 70 AND cases are UNSAT. Of 70 OR cases, 69 are initially UNSAT and one
times out: `(t OR K)*D+t`. F is bijective; since that form depends only on t,
A must be odd. Adding this necessary condition proves the final case UNSAT.
Both square-first forms are independently UNSAT at 12 bits for AND and OR.
Bypassed bitwise results are polynomial; bitwise operations that do not use
the first MAD can be reordered to the initial-bitwise case above.

### Four-input certificate for every integer polynomial

For every integer polynomial P, of any degree, the mixed difference
`P(x+h+k)-P(x+h)-P(x+k)+P(x)` is divisible by h*k. For F, use x=0,h=256,k=8:

```
inputs modulo 2048:   0,    8,  256,  264
F outputs:        1482, 1810, 1738, 1042
mixed difference modulo 2048: 1024
```

Here h*k=2048, so F cannot be an integer polynomial. This excludes not
only three MADs but every expression from x using only additions,
subtractions, multiplications and constants. It does not exclude bitwise
or other non-polynomial operations. `prove_middle_polynomial863.py` emits
the four-input certificate.

Pure bitwise compositions are bit-local. Inputs 0 and 1 differ only at bit
zero, while their F outputs 1482 and 243 have XOR bits 1336 outside bit zero.
Thus all 27 pure bitwise orders are excluded independently of SMT.

## Additional live intermediates: still open

The single-input proof does not apply when a rewrite uses values already
computed by another stage. `search_live_middle863.py` adds the previous
affine result a and its incoming state prior to the middle input b.
It uses a sound restricted domain of the full32 first hash:

```
a = 4097*prior + C0,  with 0 <= a < 2**w and w <= 19
a >> 19 = 0
b = a XOR C1
available low words: b, a, prior, zero
```

Every low word b has a valid full32 preimage because 4097 is odd. Therefore
UNSAT on this restricted domain would exclude the corresponding full32
rewrite; a low-bit SAT result would still require additional verification.
The tool rejects right/left shifts and widths above 19 rather than applying
an invalid narrowing argument.

Nine width-12 searches tested all orders of two MADs with one XOR, AND or OR,
with 64 samples and five seconds per query. **All nine are UNKNOWN/timeouts**;
none produced a candidate. An unrestricted known positive control also times
out. A supplied positive witness is SAT and its independent all-4096-input
counterexample query is UNSAT, validating the encoding but not demonstrating
that the broad search can discover the witness. These results are in
[`middle_live863_results.json`](middle_live863_results.json).

This is an open search scope, not evidence that a shortcut exists. A useful
next step requires structural partitions or a new identity, not longer runs
of the identical operand-selector model.

## Remaining whole-kernel opportunities

- Cross-stage rewrites can retain earlier state, affine values, shifted
  values or selected nodes as extra inputs. The nine queries above leave
  this scope undecided. Any candidate must be verified under full32 semantics.
- Three-instruction rewrites containing shifts, division or comparisons
  are not covered by the isolated unshifted-grammar result. Previous two-op
  exclusions do not settle those three-op programs.
- A new recurrence or record layout can change the graph. The current
  graph's 864 bound still requires at least 53 arithmetic lanes saved if
  its startup/drain losses remain 52/116. Layout estimates must include
  record initialization, pointer generation, live-value transposes, LOAD
  pressure, FLOW pressure and downstream address dependencies. No new
  layout passing those necessary budgets was found in this continuation.

Deleting one generic vector hash operation would remove 512*8=4,096 lane
operations, enough to cross the arithmetic-work gap. That calculation is a
search motivation, not a predicted achievable cycle score.

## Reproduce and validate

Optional research dependencies are in `requirements-synthesis.txt`.

```bash
python3.12 -m codex.screen_middle_lattice863 --bitwise '&'
python3.12 -m codex.screen_middle_lattice863 --bitwise '|'
python3.12 -m codex.screen_xor_mad_polynomial --width 11 --slice middle --bitwise '&'
python3.12 -m codex.screen_xor_mad_polynomial --width 11 --slice middle --bitwise '|'
python3.12 -m codex.search_middle_outer_roles863 --bitwise '&' --output /tmp/and.json
python3.12 -m codex.search_middle_outer_roles863 --bitwise '|' --output /tmp/or.json
python3.12 -m codex.screen_middle_single_mad863 --output /tmp/single-mad.json
python3.12 -m codex.prove_middle_polynomial863
python3.12 -m codex.search_live_middle863 --sequence 'mad,mad,^'
python3.12 -m unittest codex.test_middle_bitwise863 codex.test_middle_cross_stage863 codex.test_modular_polynomial
```

**23 tests pass**: independent small-width enumeration, valid positive
programs at the proof width, AND/OR synthesis controls, nonunit modular
elimination, polynomial controls through degree 64, valid full32 preimages
of the restricted live-input model, and the supplied live-input witness.
The extra OR refinement uses `check('constant',(2,3,1),timeout_ms=15000,
injective_cut=True,bitwise='|')`; square-first commands are saved in the JSON.

The active submission and separately preserved 864/865 files retain their
previous SHA256 values. `tests/` and `problem.py` are unchanged. There is no
new executable requiring a repeat of the already-passed 270 kernel cases.
