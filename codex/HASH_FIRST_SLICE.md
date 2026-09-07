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
