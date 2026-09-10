# Cross-stage C1 absorption screen

Target: with b=x XOR 0xC761C23C, compute
`(33*b+0xE9F8CC1D) XOR (16896*b+0xACCF6200)`.
Proposed replacement: `(A*x+B) XOR (D*x+F)` for arbitrary uint32 constants.
If valid, this would absorb the preceding fixed XOR while retaining two MADs
and one XOR, eliminating one hash operation per iteration.

`synthesize_hash_middle.cpp absorb-c1` bit-lifting gives frontier sizes2,8,8,0
at widths1..4, excluding the template on the supplied samples. Independently,
`test_absorb_hash_constant.py` enumerates all4096 low-four-bit(A,B,D) tuples
and infers F from x=0, without symmetry assumptions. None works for all16
inputs. The original middle expression passes its positive control.

This excludes every full-width replacement of this exact form because low
bits of multiplication/addition/XOR depend only on low input and constant
bits. It does not exclude other three-instruction forms involving shifts,
selection, additional available intermediates, or a changed representation.
No kernel rewrite or improved cycle score resulted.

## Alternative three-operation structures

Two additional forms were tested against the same target:

- nested: `D*((A*x+B) XOR x)+F`;
- input XOR: `(A*(x XOR B)+D) XOR F`.

The C++ modes`absorb-c1-nested` and`absorb-c1-input-xor` both exhaust their
bit-lifting frontier at width11, not at a frontier limit. Independent Python
bit lifting without symmetry reduction, checking all2048 low-eleven-bit
inputs, reproduces both exclusions. Three Python tests pass, including the
existing positive control for the original parallel-affine expression.

These results concern exactly the stated templates and their uint32 constants;
they do not exclude arbitrary three-instruction programs, extra hash inputs,
or transformations involving right shifts, whose low bits depend on high bits.
No performance improvement resulted. Avoid repeating these same three
constant-absorption template searches.

## Operand-selecting synthesis (2026-09-09)

`synthesize_hash_cegis.py --slice absorb-c1` now searches the same target
with SSA operand selectors: each instruction may use x, earlier results,
or either of two arbitrary constants. MAD operands need not be affine in x;
e.g. x*x and reuse of earlier results are included. Low-width searches forbid
shifts, whose narrowing semantics would invalidate the exclusion argument.

Results (Z3 5.1.0, canonical commutative operand ordering):

- width4, three unrestricted instructions,16 extra samples: equivalent
  modulo16 candidate `(x XOR12)+13`; independently checked all16 inputs.
- width12, three unrestricted instructions,24 extra samples,40s solver
  limit: UNKNOWN/timeout. This does not prove absence of a replacement.
- width12, fixed `mad,^,mad`,32 extra samples,30s solver limit: UNSAT.
  This excludes this opcode order with arbitrary legal SSA operands and
  constants, a broader family than the earlier affine nested template.
- width8, same fixed order: SAT/equivalent modulo256. This is only a
  low-width result; width12 UNSAT prevents lifting this opcode order.
- positive control (C1 omitted),width12,fixed `mad,mad,^`,24 extra samples:
  proved equivalent. An independent exhaustive4096-input check confirms
  `(1053+2081*x) XOR (2560+2560*x)` modulo4096 equals the control target.

No full32 candidate and no faster kernel resulted. Remaining opcode orders
and unrestricted search are unresolved, not globally excluded.

### Complete MAD/XOR opcode-order screen

All eight length-three opcode sequences over MAD and XOR have now been
tested, retaining arbitrary SSA operand selection and two constants per op.
At width12 with32 extra samples and30s per solver call:

| Sequence | Result |
|---|---|
| MAD,MAD,MAD | UNSAT |
| MAD,MAD,XOR | UNSAT |
| MAD,XOR,MAD | UNSAT (previous run) |
| MAD,XOR,XOR | UNSAT |
| XOR,MAD,MAD | UNKNOWN/timeout |
| XOR,MAD,XOR | UNSAT |
| XOR,XOR,MAD | UNSAT |
| XOR,XOR,XOR | UNSAT |

The unresolved XOR,MAD,MAD case also timed out at width11 with12 extra
samples and35s per solver call. Do not repeat these identical configurations
or label the entire family impossible. Existing three independent affine
template tests still pass. No full32 candidate was produced and no submitted
kernel was changed. Further work must either use a structurally different
proof/search for the unresolved case or introduce other instructions/inputs;
the seven eliminated orders are not promising repeat searches.

### Closing XOR,MAD,MAD by polynomial relaxation

`screen_xor_mad_polynomial.py --width 11` tests all2048 low-bit XOR keys
against125 deterministic inputs. No key survives (3.784s on this host).
The first XOR yields either x XOR K or a constant; the latter is covered by
K=0 in the relaxation. Two MADs then yield a polynomial in x and y=x XOR K
of total degree at most4. The screen allows all15 degree<=4 monomials with
independent coefficients, a strict superset of legal two-MAD continuations.
Each key is rejected by inconsistent linear congruences modulo2048.
Since XOR/MAD commute with low-bit projection, this excludes uint32 versions
of the same structure without needing a full32 solver search.

The modular elimination uses least-2-adic-valuation pivots, including nonunit
pivots, rather than incorrectly treating this ring as a field. Its two unit
tests include400 random systems checked against exhaustive assignments over
widths1..4 and explicit nonunit consistency cases. A positive polynomial
control `(x*(x XOR37)+19)**2+x` survives at its known key37 at widths8 and11;
the latter tests the same125 inputs and2048 keys (3.936s).

Combined with the seven previous fixed-order SMT exclusions, this closes
the three-instruction MAD/XOR-only search for this target from x alone.
It does NOT exclude shifts,AND/OR,extra available inputs,changed state
representations,or a different hash partition. No faster kernel follows.
Stop retrying C1 absorption in this closed grammar.

### AND/OR extension: all64 no-shift opcode orders closed

The30 additional length-three orders over `{mad,^,&,|}` that contain MAD
and at least one AND/OR all returned UNSAT at width12,32 extra samples,
canonical operands,1000ms solver limit (3s search budget). Each was run via
`synthesize_hash_cegis.py --slice absorb-c1 --width 12 --instructions 3
--op-sequence ORDER --canonical-operands --extra-samples 32 --seconds 3
--check-ms 1000`. There were no timeouts or surviving candidates.

The27 pure bitwise orders require no solver: any composition of unshifted
AND/OR/XOR and constants is bit-local. Inputs0 and1 differ only at bit0,
but target outputs modulo4096 are4057 and2554; their XOR has nonzero bits
outside bit0 (mask1570). Hence no pure bitwise composition can implement it.
The remaining7 orders containing only MAD/XOR and at least one MAD were
closed above. Together these cover64/64 length-three opcode orders.

Binary add/subtract/multiply can each be expressed by one MAD with constant
1,-1,or0; therefore adding those opcodes cannot rescue this three-operation
grammar. This remains a result only for the C1-plus-middle target from x
alone, with no shifts, division, selection, extra inputs, or lookup tables.
It is not a lower bound for the entire hash or tree-walk algorithm.

### Full32 right-shift9 screen

All six permutations of one MAD,one XOR,and one logical right-shift9
returned UNSAT for the same C1 absorption target under full32 semantics.
Five were rejected using24 random samples plus high-bit pairs (60 total),
canonical operands and15s solver timeout. `>>,mad,^` timed out with those
inputs but returned UNSAT with `--consecutive-samples 32` (35 total including
the default boundary inputs), no random extras/high-bit pairs. All right
shifts were explicitly constrained to literal9; this says nothing about
other shifts or variable shifts. No low-bit right-shift argument is used.

The consecutive-input option is additive and defaults to zero, preserving
existing experiments. Five independent polynomial/affine regression tests
pass. No new kernel candidate or improved cycle score resulted.
