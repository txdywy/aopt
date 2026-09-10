# Complete-window test on compact983

The candidate window is PCs938..968 inclusive, exiting to970. Packet968 is
not a bare jump: it contains6 VALU,2 LOAD,12 ALU and1 STORE operations.
It must be included when attempting to save one cycle. A preliminary smaller
window excluding968 was SAT but would skip live operations; it is rejected
and was never executed or exported.

Removing only the old exit FLOW and scheduling all live operations with an
inserted exit over30 cycles is UNSAT by capacity:184 VALU slots required,
180 available. Complete DAG598 operations/1766 edges, SHA256
81c79207b3118c370a2679ce52fc98ca7aa530d4f23b71eb2af853da0d6aa4e8.
Thus this fixed-lowering window cannot save a cycle by permutation alone.
At least4 vector operations must be lowered differently or removed, or the
window boundary expanded. This is not a global optimality claim for983.

## Four-vector scalar-lowering screen

`/tmp/aopt-smt/bin/python -m codex.experiment_small_983_lowering`
tests eight deterministic choices of four basic vector operations, expanding
each into eight lane-equivalent ALU instructions. All live operations at968
are retained, and the new exit still targets970. Each candidate has626
operations; the dependency floor is29 cycles. The first/last candidate sets
and two seeded random sets were UNSAT; four other seeded sets timed out at
3000ms (UNKNOWN, not infeasibility proofs). No candidate was SAT and no new
cycle result or submission was produced. This excludes only the four tested
fixed lowerings, not the scalar-lowering approach generally.

The capacity is extremely tight:359/360 ALU and180/180 VALU slots. A useful
next experiment should jointly select lowering and schedule, or change the
window boundary, rather than repeatedly sampling the same fixed choices.

## Expanded window896..968

This complete straight-line window has1413 operations and5175 dependencies,
with769 ALU,430 VALU,134 LOAD and71 non-exit FLOW operations. Its72-cycle
resource capacities are sufficient after reserving the new jump to970.
The unrestricted10-second solve returned UNKNOWN/timeout. With the additional
restriction that instructions only advance by at most4 or16 cycles, both
solves returned UNSAT. These are restricted-schedule exclusions, not proof
that the unrestricted72-cycle schedule is impossible. The dependency-only
floor is64. Graph digest:
`b528ce4492a218d29eabde3247777d1ba4632dc4abf335c42c3855ed9dffac1e`.

`schedule_tail` now accepts optional `advance_only`; default behavior is
unchanged. Certificate validation checks movement bounds independently.
Two unit tests in`test_schedule_movement` cover a successful compression,
a no-movement exclusion and rejection of a certificate violating the bounds.
No improved kernel or cycle score was produced by this screen.

## Memory-order correction after liveness cleanup

STORE-to-STORE lag is now zero, with original store slot order retained by
schedule emission. This matches frozen_problem's ordered mem_write updates,
even when two writes overlap. STORE-to-LOAD remains lag one. New tests cover
both cases; the movement/liveness suites total7 passing tests.

After removing774 conservatively dead static operations, window973..983
still cannot fit10 cycles:148 operations/475 edges, dependency floor11,
digest`195d58b1228270feb9e772e6dbdfca704c1a231398e80738cafb3c45c1e5fae3`.
The prior eleven-store chain is no longer imposed, but other dependencies
still prevent this fixed graph from shortening. Expanded896..968 with
advance_only16 also remains UNSAT under corrected store ordering, digest
`010d94eb2acb503a8652f8bbe3efd8e3920f76b30d0e65746e54eec665e773c0`.
No new performance result was produced. Earlier graph digests describe the
older, stricter memory model and must not be treated as current certificates.
