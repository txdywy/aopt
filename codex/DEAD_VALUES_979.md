# Lower-work979 DAG dead-definition audit

The scalar-cache graph has18 definitions with no DAG consumer:

-16 scalar level4 cached-node copies. They are consumed implicitly by emitted
 table variants and are NOT removable based on this report.
-`vbroadcast47,3`: vector constant8, requiring assembled-program read auditing
 before removing it (8 lane-work at most).
-Last `paired_direct_table_base_advance`, group17 round15, `+6,6,9`:
 requires checking assembler consumers and later scalar reuse (1 lane-work).

Thus the simple unconsumed-definition scan exposes no large dead-computation
region capable of closing the below900 gap. At most9 arithmetic lane-work is
immediately worth investigating here; that is not yet proven removable.
Do not remove the16 cache definitions or extrapolate this scan to a global
proof of liveness. The report now includes exact slots and group/round context
to make the remaining two candidates auditable against emitted instructions.

## Assembled vector8 constant audit

`python3.12 -m codex.audit_feedback_constant` scanned all4064537 emitted
packets of the scalar-cache feedback program, including all table entries.
Scratch47..54 has zero reads and exactly one write: packet3
`valu: vbroadcast47,3`. Immediate jump targets are not treated as scratch
reads. Unsupported instructions fail the audit rather than being ignored.
Therefore that broadcast can be removed from this program without changing
outputs. Packet3 remains nonempty, so removal does not create a zero-cost
empty-packet scoring artifact. `--drop-unused-constant` implements the removal
in the captured candidate only; no standalone file is changed.

Removal validation:100 full32 seeds,3 boundary patterns and all9 unchanged
official tests passed at979. It saves one VALU broadcast (8 lane-work), not
an executed cycle. Reproduce with `inspect_feedback_tail --combine
--scalar-cache --drop-unused-constant` under the Z3-enabled Python runtime.
