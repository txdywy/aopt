# Omitted level-4 setup experiment

The default-off `OMIT_LEVEL4_DIFF_SETUP` flag removes eight vector subtraction
instructions. `_Graph.emit` rejects a read of an omitted definition until a
new write establishes that scratch word. In-place reads are checked before
writes. The special paired table assembler is separately restricted to its
node-reading modes (`PAIRED_EARLY_XOR` or `PAIRED_FLOW_SELECT`). Allocation is
unchanged: some addresses are subsequently repurposed.

For the 31 cached lookups plus 12 gather replacements, with Horner indexing,
DAG counts change from VALU 5628 to 5620. Other counts remain ALU 11287,
LOAD 1781, FLOW 894, STORE 62. This is not a measured cycle improvement.
The new dead-definition audit exposes 16 cached-node broadcasts with no DAG
consumers, but those are needed by table assembly and MUST NOT be deleted
based solely on this audit.

Full 31-cache arity-2 assembly with compact returns and teacher weight 1024
fails its adjacency check: group 0, lanes (0,1), dispatch/copy PCs 310/312.
Turning off compact returns permits assembly but execution encounters an
out-of-range load. Thus this full configuration is not validated and must not
replace the 983-cycle submission. Next work should isolate the multi-round
table scheduling/return issue before further whole-graph arithmetic tuning.

Follow-up: the elided-target assembler used unit-stride table indexing even
when it emitted two packets per entry. This is a definite layout invariant
violation, not evidence that omitted differences are wrong. A regression test
first reproduced the execution IndexError; the assembler now rejects such a
layout before execution, just as compact mode already did. This is a safety
check, not yet support for nonadjacent copies. Disabling relaxed jumps and
enabling packet repair still leaves a gap (group 9, PCs 685/687), so that
configuration change alone does not solve the scheduling issue.

The default-off `PRIORITIZE_PAIRED_COPIES` scheduler option now selects ready
paired-copy ALU instructions ahead of unrelated arithmetic. With compact
returns, omitted differences, Horner indexing, relaxed jumps, no packet repair,
vector indexes and teacher weight 1024, all 31 cached depth-4 lookups in both
traversals execute correctly for 20 full32 seeds: **1060 cycles**, 33129 code
packets, scratch 1536. This is not a speed record, but replaces the invalid
whole-configuration experiment with an executable baseline. Counts: ALU 11599,
VALU 5569, LOAD 1877, FLOW 970, STORE 62. The regression test now checks both
rejection without prioritization and actual correct execution with it.

## Full four-lane measurements

Using the same flags as the successful two-lane experiment but `--arity 4`:
20 full32 seeds all pass at **1016 cycles**, 4064573 instruction packets,
scratch 1536. DAG counts ALU 11599, VALU 5569, LOAD 1877, FLOW 846, STORE 62.
Thus wider entries improve this experimental family by 44 cycles, not the
983-cycle record.

Adding 12 gather replacements (first-round groups
0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,29,31;
same final-round groups) and `--restore-hash-count 39 --profile-slots`:
20 full32 seeds all pass at **1054 cycles**, 5637475 instruction packets.
DAG: ALU 11287, VALU 5620, LOAD 1781, FLOW 894, STORE 62.
Actual executed slots are ALU 11283, VALU 5620, LOAD 1791, FLOW 895,
STORE 62; 964 main-stream cycles plus 90 table cycles. Only one table cycle
has zero VALU slots, so empty table arithmetic slots are not the principal
explanation. The lower DAG load count does not automatically produce a faster
schedule; this combination is slower than cached-only four-lane tables.
Do not confuse the driver's `main_cycles` (linear layout length, 1054) with
the profiler's actually executed main-stream cycles (964).

The unchanged standalone submission again passes all 9 official tests at 983.

## Reversed relocation combined with full tables

The 43-lookup four-lane configuration with `--reverse --restore-hash-count 22`,
Horner indexing, omitted differences, copy priority, compact returns, recycled
stride, relaxed jumps and no packet repair fails to place special-final table
base setup when `--keep-final` is enabled (`base_prep_pc` search StopIteration).
This is an assembly/setup scheduling failure, not a cycle result.

Removing `--keep-final` makes the configuration executable: all 10 full32 seeds
pass at **1201 cycles**, 5637297 packets, scratch 1535. Actual slots:
ALU 10463, VALU 5665, LOAD 1830, FLOW 889, STORE 94. Its relaxed arithmetic
bound is 930, but actual schedule is much longer. Target 899 still needs at
least 1843 fewer arithmetic lane-work units and 32 fewer load slots, even
under relaxed arithmetic reassignment. This does not support treating reversed
relocation as an immediate performance improvement; it reduces work while
introducing serious scheduling/setup costs in this combined configuration.

## Parallel index preparation

The full 31-cache, four-lane non-reversed configuration with `--parallel`
(otherwise identical to 1016: Horner, omitted differences, copy priority,
teacher 1024, no packet repair, relaxed jumps, compact return, vector indexes)
passes 20 full32 seeds at **1005 cycles**, 4064562 code packets. Actual slots
remain ALU 11595, VALU 5569, LOAD 1887, FLOW 847, STORE 62; the improvement
comes from dependency scheduling, not work reduction. Actual execution splits
into 939 main-stream and 66 table cycles. It remains slower than 983.

Adding `--restore-hash-count 22` to that parallel configuration reduces the
score further to **1002 cycles**, all 20 full32 seeds correct, 4064559 packets,
scratch 1536. DAG totals ALU 11423, VALU 5591, LOAD 1877, FLOW 846, STORE 62.
This is a 14-cycle improvement within the full-table family, not a global
record and not a sub-900 solution.

Parallel preparation on the 43-lookup, restore-39 configuration reduces 1054
to **1011 cycles**, all 20 full32 seeds correct (5637432 packets). Work counts
are unchanged. This confirms a substantial serialization penalty, but it is
still slower than cached-only parallel preparation.

`--schedule-only` now stops explicitly before table assembly with
`verified_execution: false`. It reports linear preflight packet counts, never
claims correctness or a measured score. A nine-case cached31 screen gave:

| Restored hashes | lead 0 | lead 8 | lead 16 |
|---|---:|---:|---:|
| 0 | 1005 | 1001 | 998 |
| 22 | 1002 | 998 | 998 |
| 44 | 1000 | 996 | 995 |

These are preflight sizes only; the best candidate requires full execution.

Full verification of restore44/lead16 subsequently passed all 20 full32 seeds
at **995 cycles**, 4064552 packets, scratch 1536. DAG ALU 11247, VALU 5613,
LOAD 1877, FLOW 846, STORE 62. This is the best measured cached31 family result
so far, but still slower than the standalone 983 record.

## Extended readiness/vector-balance screen

Cached31 preflight sizes for restore44/66/88 at lead16/32/64:
`995/992/987`, `989/986/986`, `988/985/985` respectively.
At leads96/128/192, restore88 and restore110 all report985; restore132
reports987. This is a local plateau, not a proof of schedule optimality.

The restore88/lead32 candidate was fully assembled and verified on **100
full32 seeds**, all **985 cycles**, 4064542 packets, scratch1536. DAG ALU10895,
VALU5657, LOAD1877, FLOW846, STORE62. It is the new best in this full-table
family, not the global 983 submission. Reproduce with the cached31 group lists
above, `--arity 4 --round 4 --keep-final --compact-return --recycle-stride
--relax-jump --no-packet-repair --vector-index --horner-index --omit-level4-diffs
--prioritize-copies --parallel --restore-hash-count 88 --dispatch-lead 32
--seeds 100 --pattern full32 --teacher-weight 1024`.

Gather12 parallel preflight sizes, columns lead32/64/128:
restore66=`994/991/995`, restore88=`985/989/987`, restore110=`993/986/986`.
These gather candidates have not received full execution verification at these
new settings. Their best preflight does not improve the validated cached31
result, so no additional multi-million-packet assembly was performed here.
Fifteen dynamic-lowering, omitted-definition, multi-round, and baseline tests
pass on this worktree.

## Leave-one-lookup-out screen (2026-09-07)

At cached31 restore88/lead32, parallel preparation, Horner and copy priority,
each of the 16 first-round and 15 final-round cached lookups was independently
removed from table coverage. Difference omission was disabled because the
original mux can consume those coefficients. All 16 first-round removals
report985 preflight packets. Final-round removals also report985 except groups
5 and12, which report986. No new execution score was claimed or table program
assembled. The 31-case experiment provides no evidence for improving this
configuration by dropping a single table lookup. The next diagnostic should
inspect the terminal dependency/resource chain rather than repeat coverage
tuning. Equal lengths alone do not prove which chain is critical.

## Terminal schedule inspection

The restore88/lead32 preflight leaves group31's last value vector679 active
after other outputs finish. The final chain contains the special paired select
at974, node XOR975, hash start976, hash middle979-980, stage4 at981, final
shift/constant982, join983, and output984. Scratch-slot-to-tag lookup can alias
reused instructions across rounds, so labels for earlier identical slots must
not be treated as exact provenance; the final vector/output sequence is clear.

Replacing special-final lookup with a full four-lane direct lookup (disable
keep-final, add31 to final-round group list) worsens preflight985 to1007.
Separately enabling existing PAIRED_EARLY_XOR before constructing both teacher
and candidate reports987. That latter check changes teacher too and is not a
controlled single-variable candidate comparison. Neither configuration was
fully assembled or claimed as a score. Straight replacement of the late
lookup therefore did not expose a better candidate; retain the 983 artifact.

## Last-group readiness priority: experimental record 982

The driver now supports `--last-group-lead`, applied only to candidate external
scores after constructing the unchanged teacher. On cached31 restore88/lead32,
last-group advances2/4/8/16/32/64 give preflight988/992/982/994/999/1003.
Advance8 was fully assembled: **100 full32 seeds all pass at982 cycles**,
4064539 instruction packets, scratch1536. This improves the local measured
record by one cycle, but is a multi-million-instruction research program, not
a portable exported submission. Exact command and outstanding validation are
in `table_982_result.json`; the compact983 submission remains untouched.

Boundary follow-up: the same982 configuration was built once and run nine
times, cycling zero/ones/alternating three times. Every output comparison
passes and every run takes982 cycles. These are three distinct deterministic
boundary patterns, not nine independent random test cases. The driver accepts
`--pattern boundaries` and requires at least three runs to ensure coverage.

## Engine ordering: experimental record981

Neighboring last-group advances5/6/7/9/10/11/12 give preflight
987/985/985/985/983/985/983, no improvement over advance8.
At unchanged restore88/lead32/last-group8, policy3 or7 with teacher1024
gives981 preflight packets. Policy3 was fully assembled and tested on100
full32 seeds: every run correct at **981 cycles**, 4064538 packets, scratch1536.
Exact reproducer is `table_981_result.json`. Policy7 was not fully executed.
This is a new experimental record, still not a portable submission or sub900.

The981 candidate subsequently passes zero/ones/alternating, each at981. All
three execution profiles agree: ALU10891, VALU5657, LOAD1887, STORE62, FLOW847;
915 main-stream cycles and66 table cycles. Unchanged LOAD lower bound944 and
relaxed arithmetic bound936 preclude reaching899 by scheduling alone. At least
89 load slots and2207 arithmetic lane-work units must disappear (or the
algorithm/representation must otherwise change). These are work-specific
bounds, not impossibility results for alternative algorithms.

## Moving gather replacement to the final traversal

The first-traversal gather12 configuration with policy3/dispatch32, tested at
restore66/88/110 and last-group4/8/12, gives preflight
1007/997/998, 1002/991/993, 1003/991/995. No candidate beats981.

Moving12 replacements to final traversal exposes an unsupported representation:
some groups save only three path bits, but direct depth4 reconstruction indexes
four bits. Both early12 and late12 final-gather sets raise IndexError during
construction. A trial routing three-bit groups to the packed mirror also fails
assembly for isolated group14 (`first_jump_pc` has no match); that trial was
reverted. Do not assume a two-line length check solves this: final non-cache
lookup routing and table jump-register discovery must be audited together.
No new execution score was obtained or claimed for these final-gather variants.

Follow-up fixed the three-bit path handling consistently: reconstruction uses
saved bits only when four are present; both individual and paired assembler
index-base maps apply the same condition. The prior trial had not updated all
matching map sites. Group14 final direct table now executes correctly on30
full32 seeds at994 cycles. This is a regression case, not a speed record.
Nine multi-round/baseline tests pass. The expensive generic Python line-tracing
diagnostic was interrupted once the untraced executable regression succeeded;
no trace hooks remain in source. Early12 final-gather placement now reaches
scheduling preflight (994 packets) instead of failing index reconstruction.

Late12 final-gather replacement has now been fully assembled and executed:
20 full32 seeds pass at994 cycles, 5637415 packets, scratch1536. Actual LOAD
1791 and FLOW895 fit within an899-cycle resource budget, but arithmetic
ALU10987/VALU5657 has relaxed bound938 and2303 excess lane-work units at899.
This is not a faster kernel, but is an executable candidate with the required
load/control work budgets. Counts and coverage are in `final_gather12_result.json`.
The portable983 submission again passes all9 unchanged official tests.

## Arithmetic decomposition of load-feasible994

DAG lane-work (VALU weighted8, ALU weighted1) identifies major non-hash costs:
mirror-bit extraction3584, mirror updates2064, gather addresses1696, dynamic
node transformation768. Direct-table index high/low packing totals388, vector
pair packing128, saved-path Horner reconstruction360, base preparation/advance
172. Thus eliminating *all* of those last four direct-index categories would
save only1048 lane units, less than the measured2303-unit excess at899. This is
an optimistic diagnostic, not a proposed legal deletion: all have consumers.
Index tuning alone cannot remove enough work in this fixed decomposition.

The omitted level4 differences are already gone. Cached node broadcasts flagged
as dead by DAG-only analysis are still read by generated table entries and are
not removable. More promising work-reduction targets must include path/address
maintenance or hash algebra, with explicit end-to-end accounting for setup.

## Original-layout address fusion check

With physical address a=idx+7 and deferred-C5 hash q, the true parity is
1-(q&1), hence child address=2*a-5-(q&1). A hoped-for lowering computes the
addend -5-(q&1) in one operation and then uses a MAD. The SMT check in
`prove_address_bit_form.py` excludes q op C and C op q for +,-,*,^,&,|,<<,>>,
and affine q*M+B: all17 families are UNSAT even on an eight-value input subset.
This is a sound rejection of those families for arbitrary uint32 inputs,
not a general instruction lower bound. Division/cdiv, alternative state
representations, and multi-instruction expressions are not covered.

For example `(q&1) ^ 0xfffffffb` computes the required addend, but takes two
operations before MAD, so it does not eliminate an operation versus the
existing parity/affine/subtraction formulation. No kernel change was made.

Eleven dead-definition, multi-round, and baseline-fingerprint tests pass.
The default kernel and the standalone 983-cycle artifact are unchanged.
