# Alias scalar level4 cache directly to retained staging values

The source words166..181 are finalized at packet8 and never written again
in the979 prefix or any large-table recipe. Their16 copies at303+8*i are
the only writes to the destination words. Large-table patches change only
ALU source operands; all other engines are checked not to consume the old
cache locations. Thus replace ALU sources and recipe patch values with the
retained staging words and remove the16 scalar copies.

`python3.12 -m codex.experiment_cache_alias_979` passed20 full32 seeds plus3
boundary patterns at979. It removes16 ALU operations, not an executed cycle.
It does not yet reclaim destination scratch or modify the standalone979 file.
This demonstrates an actual extra work reduction after the initial DAG-only
audit correctly refused to delete live cached values without source remapping.

## Scratch headroom after aliasing

Complete prefix/recipe operand coverage, including every varying table source,
finds182 wholly unused scratch words: half-open ranges[47,55),[303,429),
[1488,1536). This is not a liveness estimate: none of these words is read or
written anywhere in the encoded program. Twenty full32 seeds and3 boundaries
still pass979 after adding this audit.

The cache region releases126 words, not128: speculative unaligned tail
instructions still write429/430 as dead leading lanes. It offers15 disjoint
eight-word vectors plus6 words; the other two ranges add7 vectors. Allocation
can use these audited regions but must explicitly update new read/write and
table-patch dependencies. Existing declared scratch_ptr1536 is a reservation,
not evidence that all words are used. No new workspace placement is yet tested.

## Generator integration

Opt-in `ALIAS_LEVEL4_TABLE_CACHE`/`--alias-level4-cache` now removes the copies
before scheduling, redirecting level4 node locations to retained staging words.
The old allocated cache region is marked omitted so accidental graph reads
fail early; allocation layout itself is preserved for now. The switch requires
scalar-cache mode and inherits its layout guards. Default remainsFalse.

The recorded981 driver configuration plus scalar-cache, alias-cache and
10-percent/four-round feedback passed20 full32 cases at980 before the separate
tail rewrite. DAG ALU count10895 (down16), VALU5641. Seven dead-value tests and
the explicit invalid-mode guard passed. This enables subsequent workspace
allocation changes but does not itself beat979 or export a new submission.

## Private first-traversal workspaces

Added opt-in PRIVATE_CACHE_WORKSPACE_GROUPS / --private-cache-groups. Up to3
unique groups receive disjoint32-word blocks starting at old cached_node_15:
three condition vectors plus one spill each, used only before round11. This
requires alias-cache mode and leaves the existing allocator and second-pass
workspace assignments unchanged. Default empty tuple preserves the baseline.

Group31 alone passed20 full32 tests at980 pre-tail, with unchanged DAG work.
Seven dead-value regression tests also passed. No speedup for this assignment
is claimed; it validates a concrete use of the freed scratch and makes other
group assignments testable without expanding physical scratch beyond1536.

Additional4-round feedback screens for private groups0/1/2,8/16/24,
29/30/31 and0/15/31 all produce raw985->984 and compacted980. These are
screens only, not four additional correctness claims. An explicit DAG audit
of0/1/2 finds87 first-traversal writes to the private range, tagged mirror_bit
and tree_select, confirming the option is not a no-op. The tested private
workspace choices simply do not improve makespan under this priority scheme.
Do not repeat these same assignments expecting a different deterministic result.
