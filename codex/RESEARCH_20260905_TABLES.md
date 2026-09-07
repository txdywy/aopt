# 2026-09-05: wider direct tables

The default submission remains **987 cycles** in
`codex/perf_takehome_under1000.py`. The unchanged submission suite passed all
nine tests after the experiments. No sub-987 or sub-900 result was obtained
in this phase.

The new experiment driver is `codex/experiment_paired_base.py`. It constructs
the candidate, executes it on `tests/frozen_problem.py`, and compares all 256
output values with `reference_kernel2`. Printed `cycles` are executed machine
cycles, not static program length. `--graph-only` reports operation counts
without claiming an executed score.

## Results

| Candidate | Executed cycles | Verification |
| --- | ---: | --- |
| Existing default | 987 | Unchanged 9-test submission suite |
| 15 groups, serial pair tables | 1141 | 3 random inputs |
| Same, parallel address preparation | 1191 | 3 random inputs |
| Same, plus reversed tree layout | 1231 | 3 random inputs |
| One first-pass four-lane table group, ordinary scheduler | 1069 | 10 random inputs |
| Same, source schedule priorities, historical final table disabled | 1006 | 3 random inputs |
| Two first-pass four-lane table groups, compact return, final table retained | **988** | 100 ordinary random inputs, 30 full-width inputs, 3 boundary inputs |
| Two groups, vector pair-index packing | 989 | 10 random inputs |
| Two groups, vector packing and early tail-store lowering | 990 | 10 random inputs |

Different scheduler controls and group placements were also evaluated. Some
calendars fail the required adjacent copy/return-packet check; those are
rejected, never counted as performance results.

Reproduce the verified 988-cycle candidate from the repository root:

```sh
/opt/homebrew/bin/python3.12 codex/experiment_paired_base.py \
  --arity 4 --group-ids 3,4 --round 4 --seeds 100 \
  --teacher-weight 1024 --keep-final --compact-return \
  --recycle-stride --relax-jump --no-packet-repair
```

Repeat with `--pattern full32 --seeds 30`, or with each of
`--pattern zero`, `--pattern ones`, and `--pattern alternating` at one seed.
All these checks produced 988 cycles. This is a research alternative, not the
best submission. Its generated program contains 263,457 bundles, versus
1,312 for the incumbent; browser construction memory has not been validated.

## Implemented experiments

- Four-lane table indices use base-16 packing, with 65,536 entries per record.
  Two records cover an eight-lane group.
- Compact returns replay the complete adjacent target packet and return to
  the following main-stream packet. This removes the extra executed packet
  charged by the older elided-return scheme.
- The table stride can reuse the dead scalar M23 register after its vector
  broadcast. An optional scalar shift derives the stride instead of loading
  it. Scalar M23 consumers are rejected in this mode.
- Jump dependencies account for every possible cached source, rather than
  only the representative node-zero copy. An optional zero-lag readiness edge
  permits a producer in the jump packet, since the table reads it next cycle.
- Packet repair excludes the return marker and copied-value parents to avoid
  introducing a jump/copy dependency cycle. The list scheduler now diagnoses
  a dependency cycle instead of spinning indefinitely.
- One overlapping-source vector MADD can pack four adjacent index pairs into
  the even lanes. Scalar operations combine pairs into four-lane indices.
- Independent tail output pointers can optionally be lowered before scheduling
  for the ordinary rolling-output configuration.

Every new optimization switch defaults off; the incumbent embedded schedule
is preserved.

## Resource evidence and next constraint

For first-pass four-lane tables with vector index packing, compact returns,
the historical final table retained, and an ALU-derived recycled stride:

| Covered groups | ALU | VALU | LOAD | FLOW | Largest individual engine bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 incumbent first-cache groups | 11,360 | 5,652 | 1,876 | 891 | 947 |
| All 32 groups | 11,360 | 5,668 | 1,748 | 955 | 955 |

These are bounds for these particular operation graphs, **not measured
cycle counts and not global optimality bounds**. Setup and branch assembly
can add operations. The 16-group graph reduces the largest arithmetic engine
bound from the incumbent's 961 to 947, but cannot reach 899 by scheduling alone.
The 32-group graph reduces load pressure while increasing FLOW pressure.
Further progress toward 900 requires additional operation elimination or a
different lookup/address representation, alongside scheduling improvements.

## Follow-up: reproducible priority sweep

`sweep_paired_schedule.py` runs 72 configurations, including zero-table controls,
group sets `3,4` and `31`, teacher weights 64/256/1024/4096, height weights
1/2/4, and both stride-materialization choices. Each successful construction
is executed against the frozen reference on one input; these are screening
tests, not replacements for multi-input validation.

```sh
/opt/homebrew/bin/python3.12 codex/sweep_paired_schedule.py \
  --output .search959/paired_teacher_20260905.jsonl
```

All 72 completed correctly: 16 at 988, 16 at 989, 34 at 990, and 6 at 991
cycles. A separate 30-case sweep over dispatch lead (-12/-4/0/4/12), height
weight (0/8), and group sets (`3,4`, `31`, `29,31`) also stayed at 988–990.
There is no new incumbent. The zero-table controls illustrate that rebuilding
with source priorities does not reproduce the exact embedded calendar.

Combining reversed layout and vector pair packing was investigated but not
retained: tested calendars failed either final-table base placement or the
adjacent return-packet requirement before execution. The existing restriction
is preserved. Without vector packing, the reversed 16-group graph has 10,228
ALU, 5,709 VALU, 1,908 LOAD, 894 FLOW, and 94 STORE operations; its LOAD bound
is already 954 cycles, despite reduced ALU work.

## Follow-up: hash slice search and shallow tables

`search_hash_slice.py` screens two-instruction replacements for
`(x ^ (x >> 19)) ^ 0xC761C23C`. Its finite grammar contains scalar/VALU
arithmetic and bitwise operations, shifts, and MAD, with the constant and
multiplier pools explicitly recorded in the source. The final XOR/add/sub
constant and final MAD addend are inferred without restricting them to the
seed pool. Both the original input and first result can feed the last binary
operation. This does not enumerate every possible two-instruction program.

The run screened 1,487,232 candidates using 134 boundary/random uint32 inputs
and found no survivors. The positive control `--xor-constant 0` correctly
found `t = x >> 19; result = t ^ x`. Failure on one input disproves a candidate;
passing the samples would only nominate it for full bit-vector verification.
Neither result establishes global optimality of the ten-instruction hash.

Depth-three four-lane direct tables were also counted with compact returns
and the final special table retained. At round 3, covering 8/16/32 groups
changes ALU counts to 11,615/11,743/11,999 and VALU counts to
5,780/5,796/5,828. FLOW falls to 915/891/843. These variants exchange the wrong
resource for the current bottleneck. Round-14 coverage of the first 8/16
groups similarly increases arithmetic; coverage including group 31 conflicts
with the special final-table preparation and was rejected before execution.

Two additional public-source leads were checked. The currently visible
[Alone-King repository](https://github.com/Alone-King/verso-optimized-takehome-solution)
contains a compression-project README and web assets, not an inspectable fast
challenge kernel. The Hugging Face lead could not be fetched. Neither is
evidence of a usable new implementation or a new score.

## Depth-five scalar node cache prototype

The paired-direct lowering now experimentally supports round 5. It reserves
32 scalar node words plus a pointer and radix word, loads the preprocessed
level-five nodes in four vector loads, and makes table copies read scalar
sources (no lane offset). Explicit dependencies keep those loads after tree
preprocessing stores. The normal depth-3/4 vector-source path is unchanged.

The experiment driver retains nine path-bit workspaces and reduces the mux
spill pool to four, using its existing modulo mapping. Scratch use is 1530.
An initially tested eight-workspace alternative corrupted live path state
and was discarded; it is not a valid performance candidate.

Reproduce the valid two-lane experiment:

```sh
/opt/homebrew/bin/python3.12 codex/experiment_paired_base.py \
  --round 5 --arity 2 --group-ids 3 --compact-return --recycle-stride \
  --relax-jump --no-packet-repair --keep-final --teacher-weight 1024 \
  --pattern full32 --seeds 30
```

| Configuration | Executed cycles | Checks | Program bundles |
| --- | ---: | --- | ---: |
| One group, two-lane records | 1002 | 10 ordinary + 30 full-width inputs | 5423 |
| One group, four-lane records | 998 | 10 ordinary inputs | 2098475 |

Both are slower than the incumbent, and the four-lane code size is already
large. A 16-group four-lane graph has 1755 LOAD operations but 1003 FLOW
operations, so eliminating random loads this way alone cannot reach 899.
That is a graph count, not an executed score. Any wider use needs an explicit
FLOW budget and a practical program-size bound, not just more cached groups.

After the lowering changes, all nine unchanged submission tests still passed
at 987 cycles. These experimental depth-five paths are disabled by default.

## Return-to-dispatch fusion

`fuse_table_returns.py` implements an opt-in post-assembly experiment: when
an out-of-line table returns to an indirect-jump packet, merge that packet
into every table variant and dispatch directly. Program addresses do not
change. The pass checks engine capacity, register RAW/WAW hazards, and
conservatively rejects possible store/load ordering changes. A return site is
changed only if every variant is safe, avoiding an input-dependent mixture
of fused and unfused entries at that site.

Four unit tests pass, including frozen-machine execution that saves exactly
one cycle on a constructed valid case, and rejection tests for capacity,
register dependencies, and a single unsafe variant among otherwise valid
entries. Run `python3.12 -m unittest codex.test_fuse_table_returns`.

The incumbent has three candidate sites but all three exceed packet capacity.
The following experimental build has six sites, also all capacity-rejected;
it passed ten normal inputs at 990 cycles (no fusion applied):

```sh
/opt/homebrew/bin/python3.12 codex/experiment_paired_base.py \
  --round 4 --arity 2 --group-ids 3,4 --compact-return --recycle-stride \
  --relax-jump --no-packet-repair --keep-final --teacher-weight 1024 \
  --parallel --fuse-returns --seeds 10
```

This establishes that eliminating the return edge is semantically possible,
but not a free cycle reduction in the currently packed calendars. To exploit
it, scheduling must reserve capacity for the merged packet while preserving
the scalar index and selected-node dependencies. No new best score resulted.

## 2026-09-06: mixed-round table placement

`DIRECT_BRANCH_ROUND_OVERRIDES` allows disjoint groups to use the same-depth
tables in different traversals. Selection, fused node XOR, optional prepacking,
and table address reconstruction all use the per-group round. The driver
exposes `--also-final-group-ids`; overlapping group sets are rejected because
the current table-record identity is still group plus lanes, not group plus
round plus lanes. Defaults remain unchanged.

Round-4 group 3 plus round-15 group 4 produced:

| Mode | Executed cycles | Validation |
| --- | ---: | --- |
| Two-lane records | 990 | 10 ordinary inputs |
| Four-lane records | 989 | 10 ordinary inputs |
| Four-lane, first-pass vector index packing | 988 | 30 full-width inputs |

Reproduce the final row with:

```sh
/opt/homebrew/bin/python3.12 codex/experiment_paired_base.py \
  --round 4 --arity 4 --group-ids 3 --also-final-group-ids 4 \
  --compact-return --recycle-stride --relax-jump --no-packet-repair \
  --keep-final --teacher-weight 1024 --vector-index --seeds 30 --pattern full32
```

Covering all 16 first-cache groups and the seven non-overlapping final-cache
groups (0,1,6,10,17,23,28) yields graph counts ALU 11471, VALU 5624, LOAD
1877, FLOW 870, STORE 62 with vector index packing. The ALU capacity bound is
956 cycles; the LOAD bound is 939. These are not executed scores. This mixed
placement meaningfully reduces FLOW but still needs arithmetic elimination
or rebalancing, and cannot be scheduled below 900 in its current form.

The unchanged submission suite again passed all nine tests at 987 after
these compiler changes. No new incumbent was installed.

## Mixed-table arithmetic rebalance

The driver now exposes `--restore-hash-count`: it takes currently scalarized
hash constant-XOR operations, ordered by late round and launch phase, and
restores them to VALU. One restored operation removes eight ALU instructions
and adds one VALU instruction. The source teacher remains the unchanged
incumbent; the override is applied only to the candidate.

For the 23-group mixed graph above, restoring 22 operations gives ALU 11295,
VALU 5646, LOAD 1877, FLOW 870, STORE 62. Its largest individual capacity
bound decreases from 956 to 942 cycles. This is still a graph bound, not a
measured 942-cycle implementation, and it still excludes a sub-900 schedule.

On the small mixed group-3/group-4 candidate, restoration counts
1/2/3/4/6/8 all passed three inputs and all executed at 988 cycles.

The larger graph exposed a real assembler lookup ambiguity: the special
final table searched for the first indirect jump using a recycled register,
which could belong to an earlier mixed table. It now locates the unique
special return marker and checks the immediately preceding jump. The nine
submission tests passed at 987 after this fix.

With that ambiguity removed, the large candidate still fails genuine compact
packet adjacency: without packet repair, group 19's first record occupies
PCs 173 and 175; with repair, group 13 occupies PCs 225 and 227. These builds
are rejected before machine execution and are not performance results. The
next scheduling problem is adjacency-aware packing, in addition to the
remaining operation-count gap to 899.

## Preprocessing-depth cross-check

`--preprocess-depth` now exposes the existing preprocessing alternatives in
the mixed-table driver. For the 23-group, restore-22 graph, the measured
operation counts and corresponding largest engine capacity bounds are:

| Depth | ALU | VALU | LOAD | STORE | Capacity bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 11523 | 5736 | 1847 | 32 | 961 |
| 4 | 11435 | 5691 | 1849 | 34 | 953 |
| 5 | 11383 | 5672 | 1853 | 38 | 949 |
| 6 | 11335 | 5655 | 1861 | 46 | 945 |
| 7 | 11295 | 5646 | 1877 | 62 | 942 |
| 8 | 11271 | 5653 | 1909 | 94 | 955 |

These are fixed-graph necessary bounds, not executed scores. Reduced
preprocessing saves loads but adds enough arithmetic to lose the benefit;
depth 8 adds too many loads. Depth 7 remains the best of these alternatives.

The small mixed group-3/group-4, restore-2 configuration was executed with
ten inputs at each depth: depth 4 = 1003, depth 5 = 1001, depth 6 = 996,
depth 8 = 1005 cycles. All outputs matched the frozen reference. The same
configuration at depth 7 had previously passed at 988. No incumbent changed.

## Exhaustive coefficient search for three-instruction hash middles

`synthesize_hash_middle.cpp` searches two templates for the exact middle
function `9*((33*x+0xE9F8CC1D) XOR (16896*x+0xACCF6200))+0xFD7046C5`,
with all arithmetic modulo 2^32:

1. `(A*x+B) XOR (D*x+F)` — two MADs and one XOR.
2. `D*((A*x+B) XOR x)+F` — MAD, XOR, MAD.

Unlike the earlier finite constant-pool search, this covers arbitrary uint32
coefficients within each stated template. At x=0, F is inferred from B (and
D for the nested template). Complement symmetry normalizes B's low bit to
zero without loss of solutions. The search lifts all remaining coefficient
bits one at a time. Lower output bits depend only on lower coefficient bits,
so rejecting a partial coefficient assignment cannot discard a valid extension.

Both searches have an empty frontier at **11 bits**. Inputs 0 through 2047
alone suffice; a separate Python implementation independently reproduced both
empty frontiers. Consequently neither template can match even the low 11
bits on those inputs, and neither can implement the full uint32 function.
This is not a proof against all three-instruction programs or all nine-op
hash implementations.

Positive controls retain their known solutions through all 32 bits and end
with eight equivalent/sample-equivalent coefficient tuples each. No frontier
limit was reached in any run.

```sh
clang++ -O3 -std=c++17 codex/synthesize_hash_middle.cpp -o /tmp/aopt-synthesize-hash-middle
/tmp/aopt-synthesize-hash-middle xor
/tmp/aopt-synthesize-hash-middle nested
/tmp/aopt-synthesize-hash-middle control
/tmp/aopt-synthesize-hash-middle nested-control
```

The result removes two complete algebraic template families from the next
search, rather than just failing to find convenient constants. No submission
kernel change or improved cycle score resulted from this proof experiment.

### Third template and independent proof checker

The additional `input-xor` mode searches `(A*(x XOR B)+D) XOR F`, again
with unrestricted coefficients. Here F follows from x=0 and complementing
B can be compensated by negating A and replacing D with D-A. Its frontier
also empties at 11 bits. `input-xor-control` preserves a known valid instance
through 32 bits; no search limit is reached.

`verify_hash_middle_templates.py` independently repeats all three exclusions
using Python integers reduced by output-bit masks, and the complete input
set 0..2047. It reproduces the three empty 11-bit frontiers. The same checker
excludes every quadratic polynomial modulo 2048: x=0,1,2 leave only
coefficient triples (0,809,1482) and (1024,1833,1482), and both fail at x=16.
Thus merely replacing the middle with products of affine expressions cannot
work either. This statement concerns polynomial forms, not all possible
three-instruction ISA programs.

```sh
/tmp/aopt-synthesize-hash-middle input-xor
/tmp/aopt-synthesize-hash-middle input-xor-control
/opt/homebrew/bin/python3.12 codex/verify_hash_middle_templates.py
```

## Executed table-packet utilization, not inferred idle time

An independent architecture review raised the possibility that indirect
table copies create VALU-empty cycles. This is true of some shared/elided
table designs but must not be assumed for the current compact-return
implementation, which clones the complete scheduled packet.

The new `--profile-slots` driver flag counts instructions at execution time,
separating main-stream PCs from out-of-line PCs. Profiling does not alter
the frozen machine's execution semantics; the two phase cycle counts must
sum to the machine's cycle count.

The verified mixed group-3/group-4 compact four-lane candidate at 988 cycles
executed 8 table cycles carrying 45 VALU instructions (48 available slots),
96 ALU instructions (96 slots), 8 loads and 3 stores. There were no
VALU-empty table cycles. All three test inputs gave identical counts and
correct outputs. Thus these table cycles are already 93.75% VALU-utilized
and fully ALU-utilized, not 8 empty arithmetic cycles waiting to be reclaimed.

For comparison, the older elided-return two-lane candidate for first-pass
groups 3,4 executed at 998 cycles: 12 table cycles, 8 of them VALU-empty,
and only 16 table VALU instructions. It also passed three inputs. These
are different program structures, so the comparison establishes the idle
packet mechanism, not a controlled arity-only speedup.

The public-source refresh found an organizer's
[Optimization Arena retrospective](https://www.benedict.dev/optimization-arena-learnings)
discussing local-optimum traps and independent research diversity, but no
new fast kernel implementation. Following that methodological lead prompted
the independent review above; the actual packet hypothesis was then checked
against the local frozen machine instead of being accepted from analogy.

## Symbolic instruction synthesis with full-width counterexamples

Z3 5.1.0.0 was installed in `/tmp/aopt-smt`, using a separate temporary uv
cache, following the project's [official Python installation instructions](https://github.com/Z3Prover/z3#python).
The submission has no new dependency. `requirements-synthesis.txt` pins this
optional research dependency.

`synthesize_hash_cegis.py` searches SSA instruction sequences with arbitrary
32-bit constants and operations add/sub/XOR/AND/OR/left shift/logical right
shift/multiply/MAD. Operand selectors can use the original input, previous
results, or constants. Every candidate is checked by asking for a full-width
counterexample; only an UNSAT counterexample query establishes equivalence.
Synthesis timeouts explicitly return inconclusive, never impossibility.

Results:

- The exact stage-1 function `x XOR (x >> 19) XOR 0xC761C23C` is UNSAT at
  two instructions in this grammar, already on the six initial inputs.
- The control without the XOR constant synthesizes `x >> 19; XOR x`, and
  a separate full-uint32 counterexample query proves it equivalent.
- At three instructions the original target synthesizes the expected shift
  plus two XORs, also with full-width equivalence proved.
- The four-operation hash middle searched at three instructions hit the
  15-second solver-check timeout. This is **unresolved**, not an exclusion.

```sh
/tmp/aopt-smt/bin/python codex/synthesize_hash_cegis.py --control --seconds 30 --check-ms 10000
/tmp/aopt-smt/bin/python codex/synthesize_hash_cegis.py --seconds 30 --check-ms 10000
/tmp/aopt-smt/bin/python codex/synthesize_hash_cegis.py --instructions 3 --seconds 30 --check-ms 10000
/tmp/aopt-smt/bin/python codex/synthesize_hash_cegis.py --slice middle --instructions 3 --seconds 40 --check-ms 15000
```

This extends the earlier fixed-template work to operand and opcode synthesis,
including shifts. It does not cover every ISA instruction or prove the whole
hash globally optimal. The best executable submission remains 987 cycles.

## Low-bit screening and fixed opcode structures (2026-09-06)

The symbolic driver now accepts `--width` and `--op-sequence`. Narrow-width
screening is restricted to the middle slice and excludes both shifts. For
add/subtract/multiply/MAD/XOR/AND/OR, projection onto the low bits commutes
with every operation. Therefore an UNSAT reduced-width search excludes a
full-width implementation in that same grammar. A SAT reduced-width result
does **not** establish a full-width implementation and is explicitly labeled
`proved_equivalent_modulo`.

New bounded experiments:

| Search | Outcome |
| --- | --- |
| Middle, 3 ops, 8 bits, unrestricted nonshift grammar | Proved low-byte expression `41*x + 202` |
| Middle, 3 ops, 11 bits, unrestricted nonshift grammar, 45 s check | Timeout; unresolved |
| Middle, 3 ops, 11 bits, MAD/XOR/MAD, 30 s total | Timeout after 5 SAT candidates and counterexamples |
| Middle, 4 ops, 11 bits, MAD/MAD/XOR/MAD, 50 s total | Timeout after 5 SAT candidates and counterexamples |

The low-byte expression is not a kernel optimization: even input zero is a
counterexample at 32 bits. The fixed four-op positive topology also timing
out shows that these solver timeouts say little about existence. No new
impossibility conclusion follows from the 11-bit runs.

`test_hash_synthesis.py` adds four passing research checks: low-byte
equivalence, failure of its full-width extension, low-bit projection for all
seven permitted operations, and a right-shift counterexample explaining the
restriction. The original two-op XOR-shift control still synthesizes and
passes full-width verification. The submission suite was also rerun: all
nine tests pass at **987 cycles**. No submission algorithm was changed.

```sh
/tmp/aopt-smt/bin/python -m unittest codex.test_hash_synthesis -v
/tmp/aopt-smt/bin/python codex/synthesize_hash_cegis.py --slice middle --instructions 3 --width 8 --seconds 30 --check-ms 25000
/tmp/aopt-smt/bin/python codex/synthesize_hash_cegis.py --slice middle --instructions 3 --width 11 --seconds 50 --check-ms 45000
/tmp/aopt-smt/bin/python codex/synthesize_hash_cegis.py --slice middle --instructions 3 --width 11 --op-sequence 'mad,^,mad' --seconds 30 --check-ms 25000
/tmp/aopt-smt/bin/python codex/synthesize_hash_cegis.py --slice middle --instructions 4 --width 11 --op-sequence 'mad,mad,^,mad' --seconds 50 --check-ms 45000
```

## Verified executable improvement: 986 cycles (2026-09-06)

**New best submission: `codex/perf_takehome_986.py`.** The old
`perf_takehome_under1000.py` remains the unmodified-by-this-experiment 987
baseline, including its previously accumulated experimental switches.
`run_submission_tests.py` now selects the new standalone submission; the
official tests and frozen simulator remain unchanged.

The breakthrough is a tail-specific speculative XOR lowering, not a new
hash identity or a scheduler seed:

1. The last group has two candidate node vectors assembled in four branch
   tables. Its final selection originally precedes a separate node XOR.
2. At packet 974, the first six candidate lanes and final hash value are
   ready. XOR both candidate vectors with that value in two spare VALU
   slots. The last two candidate lanes are deliberately overwritten later.
3. Change all 64 variants of the last table so its four scalar candidate
   copies XOR the appropriate value lane instead of copying with OR.
4. The selection at packet 976 now writes the hash input directly. Remove
   the original node XOR at packet 977, starting the terminal hash one
   packet earlier.
5. Revectorize other split terminal operations and retime output stores,
   preserving the old-value semantics of streaming pointers. Both remaining
   outputs and halt now fit packet 985, yielding 986 executed cycles.

This requires **both** selection speculation and terminal schedule repair.
The speculative XOR adds arithmetic work; this is a latency improvement,
not a reduction in the global arithmetic resource lower bound. It is not
evidence that this transformation alone can reach 899 cycles.

Validation completed:

- 100 ordinary random inputs: all 256 output values correct, 986 cycles.
- 100 independent full-uint32 random inputs: correct, 986 cycles.
- Three runs each of all-zero, all-one, and alternating-bit trees/inputs:
  correct, 986 cycles.
- The unchanged official suite: all nine tests pass at 986 cycles using
  the exported standalone submission.
- Four regression tests: original program unchanged, incompatible layouts
  rejected, export instruction-for-instruction equality, reproducible source
  and successful compilation/execution without a `__file__` global.

The source transform `experiment_tail_speculation.py` checks a SHA-256 of
the entire incumbent instruction layout before applying positional edits.
The exporter embeds only the resulting legal ISA instruction stream, never
runtime tree values or expected outputs. It uses standard-library decoding
and `problem.DebugInfo`, with no project-local imports or filesystem access.

```sh
/opt/homebrew/bin/python3.12 codex/experiment_tail_speculation.py --seeds 100
/opt/homebrew/bin/python3.12 codex/experiment_tail_speculation.py --seeds 100 --full32
/opt/homebrew/bin/python3.12 -m unittest codex.test_tail_speculation -v
/opt/homebrew/bin/python3.12 -m codex.run_submission_tests
# Print the reproducible standalone source (no filesystem writes):
/opt/homebrew/bin/python3.12 -m codex.export_tail_submission
```

The public-source refresh found no new sub-900 implementation to reproduce.
The local instruction trace, rather than an unverified external score,
identified the selection/XOR dependency above. The under-900 objective
remains open; the actionable next question is where a similar speculative
lowering can shorten other bottleneck paths without increasing saturated
engine demand.

## Verified executable improvement: 985 cycles (2026-09-06)

**New best standalone submission: `codex/perf_takehome_985.py`.** Both the
986 export and the original 987 builder remain available. The local runner
now tests the 985 export against the unchanged official test suite.

`experiment_tail_quad.py` combines the final four speculative pair tables
into two quad tables. Each new table has 4096 variants, selected by a radix-64
combination of two original pair indexes. The table still returns both
possible final-node candidates for each lane, so it does not wait for the
last parity bit. It contains eight scalar candidate copies per entry.
Independent shifts were hoisted out of the first table and a split join was
revectorized in the second table to respect the twelve-ALU-slot limit.

All eight candidates are now available before packet 974. The speculative
XORs and parity extraction execute there; selection advances from packet
976 to packet 975. Table coarsening alone still executes 986 cycles because
other outputs remain on the terminal path.

`schedule_tail_window.py` builds the exact register-dependency graph for
the straight-line terminal window: 109 instructions, 184 edges, RAW/WAW
lag one, WAR lag zero. The window has no memory loads and its output stores
are to distinct output vectors; their pointer/value register dependencies
are retained. The solver found a legal nine-cycle schedule, and the frozen
machine executes it at **985 cycles**. An eight-cycle schedule is UNSAT for
this fixed instruction graph, not for the full optimization problem.

The nine-cycle schedule is saved in `tail_quad_schedule.json` with a graph
fingerprint. Export uses this certificate, independently checks every
dependency and engine capacity, and does not import Z3. The standalone file
also has no solver or project-local dependency. It contains 9504 static
packets (including retained unreachable older tables), compared with 1312
in the 986 artifact; its encoded source is 133998 bytes.

Validation:

- 100 full-uint32 random cases and 100 normal random cases: all 256 output
  values correct in every case, all at 985 cycles.
- Original official suite against the exported file: nine tests pass, 985.
- All-zero, all-one, and alternating-bit boundary cases, three offsets
  each, executed directly through the standalone export: correct, 985.
- New regression checks validate exact certificate/export agreement,
  reject a bad graph fingerprint and invalid schedule, and reproduce and
  compile the standalone source without `__file__`.

```sh
/tmp/aopt-smt/bin/python -m codex.experiment_tail_quad --schedule --seeds 100
/tmp/aopt-smt/bin/python -m codex.experiment_tail_quad --schedule --seeds 100 --pattern normal
/tmp/aopt-smt/bin/python -m codex.experiment_tail_quad --schedule --horizon 8
/opt/homebrew/bin/python3.12 -m unittest codex.test_tail_quad codex.test_tail_speculation -v
/opt/homebrew/bin/python3.12 -m codex.run_submission_tests
/opt/homebrew/bin/python3.12 -m codex.export_tail_submission --quad
```

The next improvement needs an earlier start for the terminal selection, a
different hash lowering, or changes outside this fixed window. Repeating
the same eight-cycle scheduling query cannot solve the under-900 objective.

## Verified executable improvement: 984 cycles (2026-09-06)

**New best standalone submission: `codex/perf_takehome_984.py`.** The local
submission runner selects this file. The earlier 985 and 986 exports are
preserved. No frozen-machine or official-test changes were made.

`experiment_tail_early.py` changes the incoming dependency graph rather
than rerunning the previous scheduling problem:

- Reuse a completed output vector, scratch 431..438, for a cached-node
  difference. Replace one late FLOW leaf selection with a VALU MAD and
  advance its parent selection. Speculatively XOR the two parent candidates
  before the final selection, saving the following node-XOR packet.
- Advance the last group's preceding hash through both quad-table packets.
  Its hash result is now available when the second table executes.
- The second table XORs its last four candidate lanes while copying them.
  Two non-aligned vector XORs compute the first four useful lanes while
  writing four dead padding words. Destinations 683..690 and 939..946 do
  **not** overlap the table-written 691..694 and 947..950. The old hash's
  last four words are dead after parity extraction in that same packet.
- Selection moves to packet 974. A different output hash and its two late
  gathers are also advanced to relieve the terminal store stream. One
  output uses its shape-static address in a dead table register, removing
  an unnecessary shared-pointer increment dependency. Another completed
  store moves to packet 974.

The first attempt to advance the second hash failed frozen-machine testing
on input positions 133 and 135: their node-XOR operands were not ready when
the earlier MAD read them. Their gather addresses were already ready after
957, so the two gathers moved from 969 into idle LOAD slots at 962, and the
two node XORs moved from 970 to 969. Only the repaired version is exported.

The revised tail has 111 instructions and 195 register-dependency edges.
Its certified nine-cycle schedule starts at 975 and halts at 983, producing
**984 executed cycles**. `tail_early_schedule.json` stores that schedule;
export independently validates the graph hash, dependency lags, and engine
capacities. Solver diagnostics now include the longest dependency chain
and earliest store cycles, which identified the pointer and store-pressure
issues rather than treating UNSAT as a global optimality claim.

Validation completed:

- 100 full-uint32 random cases: all 256 outputs correct, 984 cycles.
- 100 normal random cases: all 256 outputs correct, 984 cycles.
- Original official suite on the standalone export: nine tests pass, 984.
- Exported-program boundary checks: zero, all-one, alternating bits, and
  three full-width random regression seeds, including the original failing
  seed; all correct at 984.
- Certificate/export equality, reproducible source, no-`__file__` execution,
  and disjoint destination checks for all 4096 second-table variants.

The standalone source is 134753 bytes and retains the 9504-packet static
program layout. Runtime requires only the standard library and the provided
`problem.DebugInfo`, not Z3 or any project-local module.

```sh
/tmp/aopt-smt/bin/python -m codex.experiment_tail_early --schedule --seeds 100
/tmp/aopt-smt/bin/python -m codex.experiment_tail_early --schedule --seeds 100 --pattern normal
/opt/homebrew/bin/python3.12 -m unittest codex.test_tail_early codex.test_tail_quad codex.test_tail_speculation -v
/opt/homebrew/bin/python3.12 -m codex.run_submission_tests
/opt/homebrew/bin/python3.12 -m codex.export_tail_submission --early
```

These changes demonstrate useful cross-packet speculation and dead-lane
reuse, but they still optimize the terminal latency. Reaching below 900
needs larger reductions in whole-kernel work or a substantially different
lookup strategy; 984 does not satisfy that objective.

## Verified executable improvement: 983 cycles (2026-09-06)

**Current best standalone submission: `codex/perf_takehome_983.py`.** The
local submission runner now selects it; earlier exports remain available.

Instead of further shortening the terminal hash, this iteration compresses
the straight-line prefix at static packets 938..969. The original window
contains 188 VALU and 295 ALU slots; 188 VALU slots cannot fit 31 cycles
(capacity 186). `experiment_prefix_window.py` scalarizes four late basic
vector operations into eight lane-wise ALU operations each, yielding 184
VALU and 327 ALU slots. Together with LOAD=50, STORE=8, FLOW=29, the window
can fit 31 cycles, including one real FLOW slot reserved for its exit jump.

The successful schedule has 598 instructions and 1766 dependency edges.
Packet 968 jumps to 970, skipping packet 969 without relocating downstream
code or table addresses. Actual execution drops from 984 to **983 cycles**.
Two initially found schedules passed frozen-machine tests. The final export
uses a newly solved certificate with stricter same-packet memory ordering.

Scheduler extensions:

- Data-only FLOW operations (`vselect`, `add_imm`) have explicit read/write
  sets. Control transfers remain disallowed inside the scheduled window.
- Memory ordering is conservative: store-to-load and store-to-store edges
  across packets, load-to-store WAR edges, and same-packet read-before-write
  plus store ordering. No disjoint-region assumption is required.
- Earliest/latest dependency windows reduce solver domains. Impossible
  engine-work totals are rejected before solving. The inserted exit reserves
  its actual FLOW slot, and replay checks capacity with that reservation.
- `prefix_schedule.json` freezes the resulting plan. Export revalidates the
  graph and every constraint using standard Python; it does not invoke Z3.

The zero-saving 32-cycle positive control also passed. A shorter 958..969
window cannot save a cycle under its fixed graph (dependency floor 12).
The 30-cycle full-window variant with eight scalarizations was UNSAT under
the tested graph. These are local results, not global optimality claims.

Final validation (after strengthening memory constraints and regenerating):

- 100 full-uint32 random cases and 100 normal random cases: all 256 output
  values correct, all at 983 cycles.
- Original official suite: nine passing tests, 983 cycles.
- Standalone boundary cases: zero, all-one, alternating bits, three offsets
  each, correct at 983 cycles.
- Exact certificate/export equality, reproducible source, no-`__file__`
  execution, and an aliasing load/store regression that confirms an input
  load observes old memory when load and store share a packet.

The file is 135006 bytes; its static program retains 9504 packets. The
sub-900 objective remains unmet. This pass offers a reusable window-level
scheduling mechanism, but whole-kernel work reduction is still required
for the full goal.

```sh
/opt/homebrew/bin/python3.12 -m codex.experiment_prefix_window --scalarize 4 --certificate codex/prefix_schedule.json --seeds 100
/opt/homebrew/bin/python3.12 -m codex.experiment_prefix_window --scalarize 4 --certificate codex/prefix_schedule.json --seeds 100 --pattern normal
/tmp/aopt-smt/bin/python -m unittest codex.test_prefix_window -v
/opt/homebrew/bin/python3.12 -m codex.run_submission_tests
/opt/homebrew/bin/python3.12 -m codex.export_tail_submission --prefix
```

## Executed-work audit and next structural cut (2026-09-06)

`profile_resource_gap.py` observes the frozen machine while it executes the
983 submission and verifies all outputs against the reference. It counts
executed packets only, never unreachable table variants. Ten full-uint32
random cases produced identical work; the complete report is saved in
`resource_profile_983.json`.

| Engine | Executed slots | Capacity at 899 | Excess with unchanged lowering |
| --- | ---: | ---: | ---: |
| ALU | 11469 | 10788 | 681 |
| VALU | 5770 | 5394 | 376 |
| LOAD | 1889 | 1798 | 91 |
| FLOW | 936 | 899 | 37 |
| STORE | 62 | 1798 | 0 |

The relaxed ALU/VALU lane-work bound is 961 cycles:
`ceil((11469 + 8*5770)/60)`. At 899 cycles, this lowering exceeds the
combined arithmetic budget by 3689 lane-equivalent operations. These are
bounds on current work, **not** lower bounds on every equivalent algorithm.
The per-engine excesses cannot simply be added because engine reassignment
changes them together.

There are exactly 1792 executed scalar tree gathers: 256 each from depths
4 through 10. The other 97 LOAD slots are constants and vector loads.
For vector-load depth categories, the report labels the first loaded word;
a vector can cross a level boundary. Most body windows saturate LOAD and
many also saturate ALU, VALU, and FLOW simultaneously. Consequently, applying
the recent scheduling-only prefix compression to those windows cannot save
cycles without removing work.

One concrete implementation restriction was revalidated: first-pass and
final-pass cache sets contain 16 and 15 entries, but their union has only
23 groups. Their eight overlapping groups are `3,4,5,7,9,11,12,29`.
The generalized experimental direct-table path currently keys records by
group/lane and permits just one selected round per group. This prevents
replacing all 31 cached lookup positions in a single candidate. The next
structural change should key such records by `(group, round, lanes)` and
keep dispatch state and identities distinct across both rounds.

An initial FLOW-budget screening explains why this is worth implementing:
if all 31 seven-select cached lookups could become four-FLOW quad-table
lookups, that would save 93 FLOW slots. Adding quad-table replacements for
12 gathered vectors would use 48 FLOW slots while removing 96 gathers.
The provisional totals would be FLOW=891 and LOAD=1793 before any additional
setup costs. This is a **design hypothesis**, not a built or verified score:
index construction, scratch lifetimes, arithmetic work, setup, and actual
schedulability still need to be accounted for. In particular it does not
by itself close the arithmetic deficit.

No submission change was made in this audit. The verified best remains
983 cycles, and the under-900 objective remains active.

```sh
/opt/homebrew/bin/python3.12 -m codex.profile_resource_gap --seeds 10 --full32
```

## Multi-round table identities implemented (2026-09-06)

The generalized experimental builder now supports the same group in both
traversals via `PAIRED_DIRECT_EXTRA_ROUNDS`. Record identities and emission
order use `(group, round, lanes)`. Index-register selection is round-specific;
the assembler matches successive uses of reused jump registers and searches
for the target after the corresponding jump. Extra-round support requires
exclusive chained paired dispatch. All new switches default off, and the
987 builder's emitted-program fingerprint still matches the baseline used
by the exported 983 submission.

`experiment_paired_base.py --also-final-group-ids` now accepts overlap with
`--group-ids`. Small executable validations:

- Group 3 in rounds 4 and 15, quad tables, vector packing: 100 full-uint32
  cases pass at 990 cycles, plus zero/ones/alternating boundaries.
- Same overlap with pair tables: 10 full-uint32 cases pass at 992 cycles.
- Group 2 in both rounds (without saved second-path bits): 10 full-uint32
  cases pass at 994 cycles, exercising different per-round index registers.
- Regression tests verify two dispatch records in each round and execute
  the shared-group case. Existing tail-export fingerprint/reproduction
  tests and the official 983 submission tests remain passing.

These results validate the new functionality; none improves the incumbent.

New large **DAG-only** resource screens, saved with exact commands in
`multiround_resource_candidates.json`:

| Candidate | ALU | VALU | LOAD | FLOW | Relaxed arithmetic bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| All 31 cached positions | 11423 | 5614 | 1877 | 846 | 939 |
| Cached 31 + gathered 12, balanced scalar/vector | 11287 | 5643 | 1781 | 894 | 941 |
| Cached 31 + gathered 12, reversed layout | 10483 | 5683 | 1813 | 897 | 933 |

These are instruction graphs, not assembled or executed large candidates.
The regular 43-position variant clears the raw LOAD/FLOW budget for 899 but
not the arithmetic budget. The reversed variant reduces arithmetic work
but exceeds the LOAD budget and has unbalanced engine demand. Post-assembly
overhead is not included in these screens.

An additional optional lowering, `HORNER_PAIRED_DIRECT_INDEX`, forms a
saved four-bit path as `((p0*2+p1)*2+p2)*2+p3` in three MADs instead of the
previous four vector operations. It works for both ordinary and prepacked
direct indexes. The shared-group case passes 100 full-uint32 tests, but its
teacher-priority schedule regresses from 990 to 992 cycles, so this is not a
score improvement. The large regular 43-position graph loses exactly 15
VALU operations (5643 -> 5628), giving relaxed arithmetic bound 939; it
still requires substantial further work reduction.

```sh
/opt/homebrew/bin/python3.12 -m unittest codex.test_multiround_tables codex.test_tail_speculation -v
/opt/homebrew/bin/python3.12 codex/experiment_paired_base.py --arity 4 --group-ids 3 --also-final-group-ids 3 --round 4 --seeds 100 --pattern full32 --teacher-weight 1024 --keep-final --compact-return --recycle-stride --relax-jump --no-packet-repair --vector-index
# Add --horner-index to test the three-MAD index construction.
```

The best executable submission remains `perf_takehome_983.py`; the sub-900
goal is not achieved. The next design must further cut arithmetic while
retaining the new multi-round lookup coverage, and must establish an actual
legal schedule before any graph bound is reported as a result.
