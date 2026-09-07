# Isolated lookup substitution costs

Measured with `audit_lookup_marginals.py` on 2026-09-07, across all 32 groups
and rounds 3, 4, 14, 15 (128 candidates plus four baselines). These are **DAG
measurements, not execution scores or additive lower bounds**. Each candidate
starts from the same baseline with 88 scalar hash operations restored to VALU,
four-lane compact tables, final special table retained, and dead fused-node
transforms omitted. Vector index packing and Horner packing are off.

Arithmetic work below means `ALU + 8*VALU` slot units, not scalar instruction
equivalents for MAD. Setup instructions are included; their costs can be
shared when multiple substitutions are enabled.

| Round / group class | Number | Arithmetic work delta | FLOW delta |
|---|---:|---:|---:|
| 3, every group | 32 | +24 | -3 |
| 4, cached groups | 16 | -56 | -3 |
| 4, gather groups | 16 | +16 | +4 |
| 14, saved-path groups | 22 | +40 | -3 |
| 14, ordinary mirror groups | 9 | 0 | -3 |
| 15, cached groups | 15 | -16 | -3 |
| 15, gather groups | 16 | +8 | +4 |
| 15, special group31 | 1 | +24 | +4 |

Round14 group31 fails DAG generation with `ValueError: max() iterable argument
is empty`; it is not a valid candidate and was not assigned a numeric cost.

The nine zero-additional-work round14 groups are:
`2,8,13,15,19,22,24,26,30`. This difference was hidden by the earlier all-group
averages: saved paths need reconstruction for a table index; ordinary mirrors
do not. These nine substitutions are candidates for relieving FLOW pressure
in a future mixed-depth design. They do not reduce arithmetic work themselves.

Current mixed-depth assembly remains unimplemented. No combination is claimed
valid solely because isolated deltas look attractive. Actual combined DAGs,
assembled dependencies, frozen-machine correctness and cycles must be checked.

## Executed nine-group check

The nine round14 groups together pass 20 full32 seeds at **994 cycles**, 75047
instruction packets and scratch1536. Additional execution options were
`--relax-jump --no-packet-repair --prioritize-copies --dispatch-lead 32
--teacher-weight 1024 --last-group-lead 8 --policy 3`. DAG totals:
ALU10927, VALU5834, LOAD1878, FLOW912, STORE62. Compared with the no-table DAG,
the combined arithmetic delta is exactly zero and FLOW falls by27. This
confirms the local resource trade, but not a speed improvement over best981.

## Radix-eight vector index packing

Default-off `VECTOR_DEPTH3_DIRECT_INDEX`, driver `--vector-depth3-index`,
replaces the index vector copy with an overlapping-source MAD `p*8 + p_next`.
Only even lanes are consumed. Four-lane records then merge adjacent pairs
using radix64, computed once in unused temporary lane7 as `8*8`. This removes
seven scalar ALU operations per group without allocating another scratch word.
Two-lane records require no radix64 materialization. Saved-path round14 groups
are excluded because their complete mirror is unavailable; the existing path
is retained. Relocated layouts, SSA round temporaries, mixed individual direct
lanes and nonadjacent lane maps are rejected when packing applies.

Nine ordinary-mirror round14 groups: ALU10927 ->10864, VALU5834 unchanged;
20 full32 seeds correct at **994**, unchanged. Relative to the no-table DAG,
these nine substitutions now save63 arithmetic lane-work units and27 FLOW.

All32 round3 groups: ALU11295 ->11071, VALU5884 unchanged; 20 full32 seeds
correct at **1069**, versus1018 before packing. Work reduction is not a cycle
improvement: overlapping vector reads and the scratch-lane radix introduce
additional dependencies, and this experiment also reuses teacher priorities
from a different DAG. The specific source of the schedule regression has not
been isolated. Keep the flag disabled by default; do not replace best981.
