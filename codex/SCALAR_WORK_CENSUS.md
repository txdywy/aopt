# Executed 983 scalar-work census

Run `python3.12 -m codex.profile_scalar_hash`. Ten full-width random inputs
match independent reference final values and execute 983 cycles. Frozen VM
unchanged. Producer labels observe pre-packet register definitions, then update
after execution; they do not infer static constants from coincidental values.

Scalar multiplication is only 9 slots, not a meaningful source of the missing
3689 arithmetic lane-operations at 899 cycles. All 2635 vector MADs are already
fused. Runtime coefficient classifications vary across inputs and are reported
as ranges, not assumed invariant.

Seed-zero dominant immediate producer patterns:

- 3528 scalar ANDs read a load-constant and a vector XOR result.
- 256 scalar ANDs read a load-constant and a vector MAD result.
- 1792 scalar subtractions read load-constant minus vector MAD result.
- 1776 scalar XORs read vector-select and vector-XOR results.
- 1384 scalar XORs read load-constant and vector-MAD results.
- 1024 scalar XORs read gathered tree data and vector-XOR results.

These are opcode ancestry counts, NOT complete semantic stage attribution.
The next probe must identify the actual constant masks and MAD coefficients
at the 3528/1792 sites, then inspect the reaching instruction chain. In
particular, do not conflate a select output with a costly ALU mux: vselect is
on FLOW, and a subsequent XOR may be the indispensable node/hash mixing.

Candidate scope: parity extraction + affine address update, or constant XOR
absorption at MAD boundaries. Existing narrow single-op impossibility results
still apply; a new experiment must differ in representation or available inputs.

## Constant and recurrence follow-up

The extended ten-input census still validates final values at 983 cycles.
Seed zero: 3656 scalar ANDs use mask 1, 128 use mask 2. The 1792 subtractions
are exactly 256 occurrences each of `37-m`, `69-m`, `133-m`, `261-m`,
`517-m`, `1029-m`, `2053-m`. Their reaching MAD has multiplier 2 and
runtime addend 0 or 1 (893 and 899 observations respectively). Thus these
are the seven level-dependent conversions from binary-path state to tree
addresses, not hash arithmetic. Scalar constant XORs use C1 1344 times and
C5 208 times.

Even deleting ALL 1792 conversions at zero replacement cost would reduce
the measured arithmetic lane-work from 57629 to 55837: an optimistic floor
of ceil(55837/60)=931, still above 899. Real reversed-tree preprocessing also
costs loads/stores and was already investigated; do not repeat it as a new
standalone sub-900 design. Deleting 1344 scalar C1 XORs as well only reaches
ceil(54493/60)=909. This does not preclude a combination that additionally
removes vector C1 XORs, but the constant-absorption algebra remains unsolved.

These ceilings use the optimistic interchangeable ALU/VALU lane-work model,
not a predicted schedule. They omit dependencies, engine imbalance, and all
replacement work. Full output requirements and browser size still apply.

## MAD census: an alternative to changing the hash

Seed-zero executed vector MAD coefficients split into exactly 2048 hash
operations (512 each of 4097,33,16896,9), 338 operations with coefficient 2,
and 249 with other observed uniform coefficients. The latter values vary with
runtime data and must not be hard-coded. Their semantic role is not yet proven
by this coefficient-only observation; inspect reaching definitions and consumers.

This exposes a more promising combined budget than scalar multiplication:
249*8 + 1792 = 3784 lane units, slightly exceeding the 3689 deficit. If these
249 operations are replaceable selection overhead, eliminating them together
with address conversions would yield an optimistic 898 arithmetic floor
(ceil((57629-3784)/60)). Only 95 lane units of headroom remain, and LOAD/FLOW
gates still fail without further changes. This is a candidate budget, NOT a
claim that either deletion is implementable or that a 898 schedule exists.

## The 249 non-hash MADs are Boolean interpolation

Follow-up ancestry census: all 249 have a Boolean first operand, a vector
subtraction as coefficient producer, and a broadcast as addend producer.
248 Boolean vectors came from scalar ANDs; one came from vector AND.
Generator inspection corroborates cached node-pair differences and bottom-pair
MAD selection (`level4_diff`, `HYBRID_MADD_PAIRS`). This is interpolation
`base + bit * (other-base)`, not redundant arithmetic.

Replacing all 249 with FLOW vselect would increase 936 FLOW slots to 1185,
requiring 286 FLOW slots removed elsewhere to fit 899 (even if source nodes
remain live and materialization is free). Scalar multiply+add would consume
16 ALU slots per vector instead of one VALU, increasing lane-work. An ALU
bitwise mux requires mask construction and multiple operations, also not a
zero-cost deletion. Therefore the preceding 898 free-deletion budget does
not constitute an implementable route. A useful replacement must fold node
selection into another operation or remove an existing traversal/dispatch,
not merely switch the execution engine.

## Direct-table XOR fusion audit

Generator inspection confirms that `FUSED_DIRECT_BRANCH_XOR` already suppresses
the later node XOR for covered direct-table lanes in both cached depth-four
and ordinary gather paths. The triple-table experiment previously fixed the
cached-path double-XOR bug and has end-to-end regressions. Therefore proposing
"fuse table lookup and node XOR" again is not a new optimization for that path.
The remaining Boolean-interpolation MADs belong to other lookup lowerings;
removing them requires changing coverage/control cost, not enabling this flag.

No performance improvement resulted from this audit. Avoid conflating the
existing fusion with a missing copy-elimination opportunity, and do not rerun
old coverage sweeps without a new cost-reducing mechanism.
