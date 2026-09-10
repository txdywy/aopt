# Middle hash plus outer MAD synthesis

Target (all arithmetic uint32):

```
9*((33*x+0xE9F8CC1D) XOR (16896*x+0xACCF6200))+0xFD7046C5
```

Current lowering is two MADs,one XOR,one MAD. A three-op replacement would
save one instruction per SIMD hash invocation without additional lookup data.

At width12,32 consecutive plus16 random samples (49 total),canonical
operands and15s solver timeout, each of `mad,^,mad`, `mad,mad,^`, and
`^,mad,mad` returned UNKNOWN/timeout. These bounded attempts found no candidate.

Independent necessary-condition test:
`python -m codex.screen_xor_mad_polynomial --width 11 --slice middle`.
All2048 XOR keys are inconsistent with the degree<=4 polynomial relaxation
on125 deterministic inputs (4.638s). Hence `^,mad,mad` is excluded for
arbitrary SSA operands/constants, under the same proof used for C1 absorption:
after the initial XOR, two MADs lie inside the15-monomial polynomial family.
This test uses no shifts, so low11 exclusion lifts to uint32.

Five independent modular-elimination/affine regression tests pass. The other
two opcode orders above remain unresolved at this broader operand-selection
scope. No faster kernel or full-width candidate resulted. This is not a
global lower bound on the middle hash, and does not cover extra inputs or
shift/division/select instructions.

## First-MAD structure partition

Each of `mad,^,mad` and `mad,mad,^` was split using `--first-mad-form`
into affine,square-constant,and square-input. At width12,32 consecutive
plus16 random samples,canonical operands and10s solver limit,all four square
branches were UNSAT; both affine branches timed out. This uses the proven
complete one-input MAD partition from test_hash_synthesis.py, not a heuristic
restriction claiming full coverage without a check.

For `mad,^,mad`,fixing the first affine multiplier to33 (offset still arbitrary)
returned UNSAT with the same samples and12s solver limit. Thus reusing that
original multiplier and moving constants alone cannot produce a three-op
solution of this order. Arbitrary affine multipliers remain unresolved.
Eight synthesis tests pass. No full-width candidate or faster kernel resulted.

## AND/OR extension

Width12 with32 random plus32 consecutive inputs (65 unique), canonical
operands,3 seconds/check: `mad,&,mad`, `mad,|,mad`, `mad,mad,&`, and
`mad,mad,|` all UNKNOWN. Splitting the first two to affine first MAD with
multiplier33 and arbitrary offset: AND order remains UNKNOWN; OR order
returns UNSAT. Because no shifts occur, this low12 UNSAT excludes the fixed33
OR template at full32. No positive candidate or kernel change. Other AND/OR
orders and arbitrary first multipliers are not excluded by these runs.
